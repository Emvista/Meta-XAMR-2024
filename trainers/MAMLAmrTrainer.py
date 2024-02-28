from settings import TASK2PATH
import torch
import learn2learn as l2l
import transformers
from utils.torchutils import SaveBestModel, EarlyStopper
import numpy as np
from CheckpointHandler import CheckpointHandler
from Evaluator import AmrEvaluator
import wandb
from settings import PROJECT_DIR
from tqdm import tqdm
import random
from copy import deepcopy
import logging

def random_sample_tasks(tasks, num_tasks):
    return random.sample(tasks, num_tasks)

logger = logging.getLogger()
logfmt = '%(asctime)s - %(levelname)s - \t%(message)s'
logging.basicConfig(format=logfmt, datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

class MAMLAmrTrainer:
    def __init__(self, model, device, train_episode_samplers, eval_steps, max_length,
                 dev_finetuning_dataloaders, dev_eval_dataloaders, tokenizer,
                 save_steps, checkpoint_dir, inner_loop_lr,
                 outer_loop_lr, max_steps,
                 adaptation_steps, disable_tqdm, silent_amr_postprocessing,
                 patience, lr_scheduler_max_steps, save_total_limit, verbose, task_batch_size, eval_finetune_lr,
                 prediction_out_dir, finetuning_batch_size
                 ):
        self.save_total_limit = save_total_limit
        self.early_stop = EarlyStopper(patience=patience)
        self.device = device
        self.max_steps = max_steps # max number of steps for training
        self.lr_scheduler_max_steps = lr_scheduler_max_steps # max number of steps for lr scheduler
        self.outer_loop_lr = outer_loop_lr
        self.max_length = max_length

        self._model = model
        self._model.to(device=self.device)
        self.inner_loop_lr = inner_loop_lr
        self.maml = l2l.algorithms.MAML(self._model, lr=self.inner_loop_lr, first_order=True, allow_unused=True)
        self.tokenizer = tokenizer
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        self.silent_amr_postprocessing = silent_amr_postprocessing
        self.task_batch_size = task_batch_size

        self.disable_tqdm = disable_tqdm
        self.eval_steps = eval_steps
        self.save_steps = save_steps # save checkpoint every n steps
        self.checkpoint_dir = PROJECT_DIR / "models" / checkpoint_dir # dir to save checkpoints

        self.preds_save_dir = PROJECT_DIR / "predictions" / checkpoint_dir # dir to save predictions
        if prediction_out_dir is not None:
            self.preds_save_dir = self.preds_save_dir / prediction_out_dir

        self.train_episode_samplers = train_episode_samplers # [task_sampler1, task_sampler2, ...]
        # self.dev_tuple_dataloaders = dev_tuple_dataloaders # [(k_shot_dl, eval_dl), ...]
        self.dev_finetuning_dataloaders = dev_finetuning_dataloaders # dict {"es-amr": []...}
        self.dev_eval_dataloaders = dev_eval_dataloaders # dict {"es-amr": []...}

        self.global_step = 0
        self.checkpoint_handler = CheckpointHandler(save_total_limit=self.save_total_limit, checkpoint_dir=self.checkpoint_dir)
        self.verbose = verbose

        self.save_best_model = SaveBestModel(smaller_is_better=False)
        self.adaptation_steps = adaptation_steps
        self.eval_finetune_lr = eval_finetune_lr
        self.finetuning_batch_size = finetuning_batch_size

    def create_optimizer(self):
        """
        Create optimizer for the model (for now, does not suppport user defined optimizer)
        :return: optimizer
        """
        optimizer = torch.optim.AdamW(self.maml.parameters(), lr=self.outer_loop_lr)
        return optimizer

    def create_scheduler(self, warmup_steps=None, last_epoch=-1):
        """
        Create scheduler for the model (for now, does not suppport user defined optimizer)
        when resume from checkpoint, need to set last_epoch to the last epoch in the checkpoint
        :return: scheduler
        """
        if warmup_steps is None:
            warmup_steps = int(self.max_steps * 0.05) # 5% of train steps for warmup

        scheduler = transformers.get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                                 num_warmup_steps=warmup_steps,
                                                                 num_training_steps=self.lr_scheduler_max_steps,
                                                                 last_epoch=last_epoch)

        return scheduler

    def outer_step(self, episode_samplers):
        """
        One step of outer loop of MAML
        :return: training loss for a batch of tasks
        """
        num_tasks_per_step = len(episode_samplers)
        query_loss_over_tasks = 0.0
        random_episode_samplers = random_sample_tasks(episode_samplers, self.task_batch_size) # randomly choose n tasks for each step

        for episode_sampler in random_episode_samplers:
            task_type = episode_sampler.task_type
            support, query = episode_sampler.sample()
            learner = self.maml.clone()

            # fast adaptation step (inner loop)
            support_loss = self.inner_step(learner, query, task_type=task_type)

            # eval with query set
            outputs = learner(**query, task_type=task_type)
            query_loss = outputs.loss
            query_loss.backward()
            query_loss_over_tasks += query_loss.item()

            if self.verbose:
                src_lang = episode_sampler.src_lang
                tgt_task = episode_sampler.tgt_task
                logger.info(f"step {self.global_step} [{src_lang}-{tgt_task}] support loss: {support_loss:.4f}, query loss: {query_loss:.4f}")

        return query_loss_over_tasks / num_tasks_per_step # average loss over tasks

    def inner_step(self, learner, support, task_type):
        """
        One step of inner loop of MAML
        :param learner:
        :param support:
        :return: loss, learner
        """
        support_loss = 0.0

        for i in range(self.adaptation_steps):
            outputs = learner(**support, task_type=task_type)
            loss = outputs.loss
            learner.adapt(loss) # in-place adaptation of learner
            support_loss += loss.item()

        return support_loss / self.adaptation_steps # average loss over adaptation steps

    def load_latest_checkpoint(self):
         # find the latest checkpoint and load the model
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))

        if len(checkpoints) == 0:
            AssertionError("No checkpoints found")

        checkpoints = sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)
        latest_checkpoint = checkpoints[0]
        logger.info(f"Loading checkpoint {latest_checkpoint}")
        self.load_from_checkpoint(checkpoint_path=latest_checkpoint)

    def load_from_checkpoint(self, checkpoint_path=None, load_best_model=False):
        """
        :param checkpoint_path:
        :param load_best_model: True for eval at the end of training, False for resuming training
        update the model, optimizer, scheduler, and global step from checkpoint
        """
        logger.info(f"loading checkpoint from {checkpoint_path}")
        if checkpoint_path is None and not load_best_model:
            raise ValueError("Must provide checkpoint path or set load_best_model to True")

        if load_best_model:
            best_step = self.save_best_model.best_step
            checkpoint_path = self.checkpoint_dir / f"checkpoint-{best_step}.pt"
            logger.info(f"best model loaded from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.to(device=self.device)
        self.maml = l2l.algorithms.MAML(self._model, lr=self.inner_loop_lr, first_order=True, allow_unused=True)

        self.optimizer = self.create_optimizer()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.early_stop.counter = checkpoint['early_stop_counter']
        logger.info(f"load early stop counter to {self.early_stop.counter}" )
        self.save_best_model.best_val = checkpoint['best_smatch']

        #TODO: temporary try and except for old scripts
        try:
            self.save_best_model.best_step = checkpoint['best_step']
            logger.info(f"load best step at {self.save_best_model.best_step}" )
        except:
            logger.warning("best step not found")

        if not load_best_model:
            # if resuming from the last checkpoint, update the global step and scheduler
            # if loading best model, do not update the global step and scheduler
            self.global_step = checkpoint['global_step'] + 1
            self.scheduler = self.create_scheduler(last_epoch=self.global_step)

        logger.info("checkpoint successfully loaded")

    def save_current_checkpoint(self):

        checkpoint_path = self.checkpoint_dir / f"checkpoint-{self.global_step}.pt"
        logger.info(f"saving checkpoint at {checkpoint_path}")

        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_smatch': self.save_best_model.best_val,
            'early_stop_counter': self.early_stop.counter,
            'best_step': self.save_best_model.best_step
        }, checkpoint_path)

    def kshot_finetune(self, dataloader, finetune_lr=1e-4):
        logger.info("copying the model")
        model = deepcopy(self._model)
        model.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)
        grad_accum_steps = self.finetuning_batch_size // dataloader.batch_size
        
        for j in range(self.adaptation_steps):
            step_loss = []

            for i, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps
                step_loss.append(loss.item())
                loss.backward()

                if (i + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            ada_step_loss = np.mean(step_loss)
            logger.info(f"eval finetuning {j+1}/{self.adaptation_steps} loss : {ada_step_loss:.4f}")

        return model

    def train(self, resume_from_checkpoint=False):
        """
        over training steps, compute loss (outer step) and update model parameters
        :return:
        """
        self.maml.train()
        num_tasks_per_step = len(self.train_episode_samplers)

        if resume_from_checkpoint:
            self.load_latest_checkpoint() # load the latest checkpoint

            for task_sampler in self.train_episode_samplers:
                task_sampler.fast_forward_to_step(self.global_step)

        # start training
        for step in tqdm(range(self.global_step + 1, self.max_steps + 1), disable=self.disable_tqdm):
            self.global_step = step

            logger.info(f"step {step}")
            avg_query_loss_over_tasks = self.outer_step(self.train_episode_samplers)
            # Average the accumulated gradients and optimize
            for name, p in self.maml.named_parameters():
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / num_tasks_per_step)
                else:
                    pass #TODO: add user warning with logger

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Log training loss
            wandb.log({"train/lr": self.scheduler.get_lr()[0]})  # log learning rate
            wandb.log({"train/loss": avg_query_loss_over_tasks, "step": step})
            logger.info(f"step {step} avg loss: {avg_query_loss_over_tasks:.4f}")

            # Evaluate on dev set
            if step % self.eval_steps == 0:
                metric_key_prefix="dev"

                for task, dataloader in self.dev_eval_dataloaders.items():
                    # for eval dataset, there is only one dataloader per task, so we can just take the first one
                    eval_metrics = self.evaluate(finetuning_dataloader=self.dev_finetuning_dataloaders[task],
                                                 eval_dataloader=dataloader,
                                                 metric_key_prefix=metric_key_prefix,
                                                 split="test")

                    # split is test because we want to evaluate on Spanish test set, and finetuning on Spanish eval set
                    if "amr" in task: # if task is amr, save best model based on smatch
                        self.save_best_model(current_val=eval_metrics[f"{metric_key_prefix}_smatch"],
                                             current_step=step,
                                             model=self._model,
                                             optimizer=self.optimizer,
                                             save_to=self.checkpoint_dir / f"checkpoint-{step}.pt")

                    self.maml.train() # back to train mode

                if step > self.lr_scheduler_max_steps * 0.25: # start early stop counting after 25% of max steps reached
                    increase_counter = True if self.save_best_model.best_step != step else False
                    if self.early_stop(increase_counter=increase_counter):
                        logger.info(f"Early stopping at step {step}")
                        return  # stop training

            # Save checkpoint
            if step % self.save_steps == 0:
                torch.cuda.empty_cache()
                if self.save_best_model.best_step != step:
                    # save current checkpoint only if it is not already saved as best model
                    self.save_current_checkpoint()
                # Delete old checkpoints (except for the best ckpt) to save disk space
                best_model_checkpoint = self.checkpoint_dir / f"checkpoint-{self.save_best_model.best_step}.pt"
                self.checkpoint_handler.rotate_checkpoints(use_mtime=True,
                                                           best_model_checkpoint=best_model_checkpoint)


    def eval_log(self, metrics, prefix):
        """
        log metrics with logger and wandb
        :param metrics:
        :return:
        """

        def remove_prefix(input_string, prefix):
            if input_string.startswith(prefix):
                return input_string[len(prefix)+1:]
            else:
                return input_string

        simple_metric = {}

        for key, value in metrics.items(): # delete the prefix (ex. dev_loss => loss) for logging
            new_key = remove_prefix(key, prefix)
            simple_metric[new_key] = value

        wandb.log({f"{prefix}/{k}": v for k, v in simple_metric.items()})
        logger.info(" ".join([f"{prefix}_{k}: {v:.2f}" for k, v in simple_metric.items()]))
        # logger.info(f"step {self.global_step} {prefix}_loss: {simple_metric['loss']:.4f}, {prefix}_smatch: {simple_metric['smatch']:.4f}")


    def evaluate(self, finetuning_dataloader, eval_dataloader , metric_key_prefix, split="test", lr=-1, kshot_finetune=True):
        """
        evaluate on a set of tasks
        :param finetuning_dataloader: dataloader for kshot finetuning
        :param eval_dataloader: dataloader for evaluation
        :param metric_key_prefix: prefix for metric keys in wandb
        :param split: target split to find sent file / gold amr file
        :return: eval_metrics (dict of metrics)
        """
        # total_loss = []
        # total_smatch = []

        if "test" in split and lr > 0:
            logger.info(f"INFORMATION: setting lr to {lr} for finetuning..")
            self.maml.lr = lr # set lr for kshot finetuning

        src_lang = eval_dataloader.dataset.src_lang
        tgt_lang = eval_dataloader.dataset.tgt_task
        ref_path = TASK2PATH.get_path("{}-{}".format(src_lang, tgt_lang), split=split)
        model = self._model

        if kshot_finetune:
            logger.info("kshot finetuning before evaluation")
            model = self.kshot_finetune(finetuning_dataloader, finetune_lr=self.eval_finetune_lr)

        evaluator = AmrEvaluator(tokenizer=self.tokenizer,
                                 eval_gold_file=ref_path[2],
                                 sent_path=ref_path[0],
                                 pred_save_dir=self.preds_save_dir,
                                 model=model,
                                 dataloader=eval_dataloader,
                                 src_lang=src_lang,
                                 silent_amr_postprocessing=self.silent_amr_postprocessing)

        loss, smatch = evaluator.run_eval(max_len=self.max_length, n_step=self.global_step, suffix=metric_key_prefix)

        eval_metrics = {f'{metric_key_prefix}_loss': loss,
                        f'{metric_key_prefix}_smatch': smatch * 100,
                        "step": self.global_step}

        self.eval_log(eval_metrics, prefix=metric_key_prefix)
        torch.cuda.empty_cache()  # to avoid OOM after evaluation

        return eval_metrics


