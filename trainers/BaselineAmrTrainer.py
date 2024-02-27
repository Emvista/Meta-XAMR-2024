from transformers import Seq2SeqTrainer
from torch.utils.data.dataloader import DataLoader
from settings import logger
from Evaluator import AmrEvaluator
from transformers import DataCollatorForSeq2Seq
from settings import TASK2PATH
import copy
import torch


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self,
                 prediction_out_dir,
                 eval_finetuning_dataset,
                 # test_finetuning_dataset,
                 max_len,
                 num_evaluations,
                 eval_finetune_lr,
                 finetuning_batch_size,
                 **kwargs):

        super().__init__(**kwargs)
        self.dev_finetuning_dataset = eval_finetuning_dataset
        self.num_evaluations = num_evaluations
        # self.test_finetuning_dataset = test_finetuning_dataset  # for consistency with old code

        # prepare kshot dataloaders
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                                    model=self.model,
                                                    padding='longest', # 'max_length' previously
                                                    max_length=max_len,
                                                    return_tensors='pt')

        self.prediction_out_dir = prediction_out_dir
        self.max_len = max_len
        self.eval_finetune_lr = eval_finetune_lr
        self.finetuning_batch_size = finetuning_batch_size

    def kshot_finetune(self, dataloader, adaptation_steps=1, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.args.learning_rate  # use the same learning rate as the trainer

        logger.info("kshot finetune rate : {}".format(learning_rate))
        temp_model = copy.deepcopy(self.model)
        temp_model.to(self.args.device)
        temp_optimizer = torch.optim.AdamW(temp_model.parameters(), learning_rate)
        grad_accum_steps = max(1, self.finetuning_batch_size // self.args.per_device_train_batch_size)

        for j in range(adaptation_steps):
            step_loss = 0

            for i, batch in enumerate(dataloader):
                batch.to(self.args.device)
                outputs = temp_model(**batch)
                loss = outputs.loss / grad_accum_steps
                loss.backward()
                step_loss += loss.item()

                if (i + 1) % grad_accum_steps == 0:
                    temp_optimizer.step()
                    temp_optimizer.zero_grad()

            step_loss /= (len(dataloader) // grad_accum_steps)
            print(f"{j + 1}/{adaptation_steps} adaptation step loss: {step_loss}")

        return temp_model

    def get_dataloader(self, k_dataset, shuffle=False):
        # load k examples for finetuning the model before evaluation
        data_collator = self.data_collator
        k_dataloader = DataLoader(batch_size=self.args.per_device_train_batch_size, shuffle=shuffle, dataset=k_dataset,
                                  collate_fn=data_collator)

        return k_dataloader

    def load_ref_for_eval(self, metric_key_prefix, eval_dataset):
        # load evaluation data for smatch calculation
        logger.info(f"Loading {metric_key_prefix} data for evaluation")
        if metric_key_prefix == "eval":
            dataloader = self.get_eval_dataloader()

        elif "test" in metric_key_prefix:
            dataloader = self.get_test_dataloader(eval_dataset)

        else:
            raise ValueError(f"metric_key_prefix {metric_key_prefix} not recognized")

        src_lang = dataloader.dataset.src_lang
        tgt_lang = dataloader.dataset.tgt_task

        split = metric_key_prefix if metric_key_prefix == "silvertest" else "test"  # ref files are for test
        sent_file, _, amr_file = TASK2PATH.get_path(name=f"{src_lang}-{tgt_lang}", split=split)

        return dataloader, sent_file, amr_file

    def verbose_early_stopping(self):
        if len(self.callback_handler.callbacks) > 3:
            early_stopping_counter = getattr(self.callback_handler.callbacks[3], 'early_stopping_patience_counter',
                                             None)
            early_stopping_patience = getattr(self.callback_handler.callbacks[3], 'early_stopping_patience', None)

            if early_stopping_counter is not None and early_stopping_patience is not None:
                logger.info(
                    "early_stopping_counter: {} / {}".format(str(early_stopping_counter), str(early_stopping_patience)))
            else:
                logger.info("Callback attributes not available.")
        else:
            logger.info("Not enough callbacks available.")

    def evaluate(self, eval_dataset=None,
                 finetuning_dataset=None,
                 ignore_keys=None,
                 metric_key_prefix="eval",
                 kshot_finetune=True,
                 adaptation_steps=1,
                 shuffle_k_dataset=False,
                 **gen_kwargs):
        """
        Evaluate the model and return the loss & smatch score on the evaluation set.
        :param eval_dataset:
        :param ignore_keys:
        :param metric_key_prefix:
        :param max_len:
        :param num_beams:
        :param prefix:
        :return: {"loss": 0.0, "smatch": 0.0, "epoch": 0}
        """
        # load k shot data and eval data
        dataloader, sent_file, amr_file = self.load_ref_for_eval(metric_key_prefix, eval_dataset)
        if metric_key_prefix == "eval":
            kshot_finetuning_dataloader = self.get_dataloader(self.dev_finetuning_dataset)
            model = self.kshot_finetune(kshot_finetuning_dataloader,
                                        adaptation_steps=adaptation_steps,
                                        learning_rate=self.eval_finetune_lr)

            evaluator = AmrEvaluator(tokenizer=self.tokenizer,
                                     eval_gold_file=amr_file,
                                     sent_path=sent_file,
                                     pred_save_dir=self.prediction_out_dir,
                                     model=model,
                                     dataloader=dataloader,
                                     src_lang=amr_file.stem.split("_")[0])

            loss, smatch = evaluator.run_eval(max_len=512, n_step=self.state.global_step)

            eval_metrics = {f'{metric_key_prefix}_loss': loss, f'{metric_key_prefix}_smatch': smatch,
                            "steps": self.state.global_step}

            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, eval_metrics)
            self.verbose_early_stopping()

            self.log(eval_metrics)
            return eval_metrics

        elif "test" in metric_key_prefix:
            kshot_finetuning_dataloader = self.get_dataloader(finetuning_dataset, shuffle=shuffle_k_dataset)

            src_lang = dataloader.dataset.src_lang
            tgt_lang = dataloader.dataset.tgt_task
            split = "silvertest" if "silvertest" in metric_key_prefix else "test"
            ref_path = TASK2PATH.get_path("{}-{}".format(src_lang, tgt_lang), split=split)

            model = self.model

            if kshot_finetune:
                logger.info("kshot finetuning model...")
                model = self.kshot_finetune(kshot_finetuning_dataloader,
                                            adaptation_steps=adaptation_steps,
                                            learning_rate=self.eval_finetune_lr)

            evaluator = AmrEvaluator(tokenizer=self.tokenizer,
                                     eval_gold_file=ref_path[2],
                                     sent_path=ref_path[0],
                                     pred_save_dir=self.prediction_out_dir,
                                     model=model,
                                     dataloader=dataloader,
                                     src_lang=src_lang, )

            loss, smatch = evaluator.run_eval(max_len=self.max_len, n_step=self.state.global_step,
                                              suffix=metric_key_prefix)

            # log metrics
            eval_metrics = {f'{metric_key_prefix}_loss': loss,
                            f'{metric_key_prefix}_smatch': smatch * 100,
                            "step": self.state.global_step}

            self.log(eval_metrics)
            return eval_metrics
