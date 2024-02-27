from transformers import Seq2SeqTrainingArguments
from argparse import ArgumentParser
import numpy as np
import pprint
from transformers import MBartForConditionalGeneration
from transformers import AutoTokenizer
from torch.utils.data import ConcatDataset
from settings import logger
from transformers import DataCollatorForSeq2Seq
from settings import TASK2PATH, PARSING_TASKS, PROJECT_DIR
from custom_datasets.CustomDataset import AmrSqueezedTokDataset
from transformers import EarlyStoppingCallback
from train_maml_xlingual import split_to_subset
from trainers.BaselineAmrTrainer import CustomSeq2SeqTrainer


def load_tokenized_dataset(task_name, tokenizer, split, max_len, num_examples=None):
    dataset_path = TASK2PATH.get_path(task_name, split)  # returns a tuple of (sent_file_path, amr_file_path, penman file path)
    dataset = AmrSqueezedTokDataset(tokenizer=tokenizer, datapath=dataset_path, max_len=max_len, size=num_examples)
    return dataset


def load_model(trainer, checkpoint_path):
    checkpoint = PROJECT_DIR / "models" / checkpoint_path
    logger.info("Loading model from checkpoint:", checkpoint)
    trainer.model = MBartForConditionalGeneration.from_pretrained(checkpoint)
    trainer.model.to(trainer.args.device)
    logger.info("trainer model device on:", str(trainer.model.device))


def train_model(trainer, checkpoint_path):
    trainer.train(resume_from_checkpoint=checkpoint_path)


def evaluate_model(trainer, test_dataset, test_finetuning_datasets, num_evaluations, task_prefix, kshot_finetune, adaptation_steps):
    logger.info("Evaluating model...")
    eval_metric = {}

    for i in range(num_evaluations):
        logger.info(f"Running {task_prefix} evaluation {i + 1}/{num_evaluations}")
        test_finetuning_dataset = test_finetuning_datasets[i]
        metric = trainer.evaluate(eval_dataset=test_dataset,
                                  finetuning_dataset=test_finetuning_dataset,
                                  metric_key_prefix=f"{task_prefix}{i + 1}",
                                  kshot_finetune=kshot_finetune,
                                  adaptation_steps=adaptation_steps)
        eval_metric.update(metric)

    mean_smatch = np.mean([eval_metric[f"{task_prefix}{i + 1}_smatch"] for i in range(num_evaluations)])
    mean_std = np.std([eval_metric[f"{task_prefix}{i + 1}_smatch"] for i in range(num_evaluations)])
    trainer.log({f"{task_prefix}_smatch": mean_smatch, f"{task_prefix}_std": mean_std})


def main(args):
    # 1. load tokenizer and model
    model_name = args.model_name
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    default_special_tokens = tokenizer.additional_special_tokens
    extended_special_tokens = default_special_tokens + PARSING_TASKS
    tokenizer.add_special_tokens({'additional_special_tokens': extended_special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    output_dir = PROJECT_DIR / "models" / args.output_dir

    # 2. set up trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           model=model,
                                           padding='max_length',
                                           max_length=args.max_length,
                                           return_tensors='pt')

    training_args = Seq2SeqTrainingArguments(output_dir=output_dir,
                                             evaluation_strategy="steps",
                                             do_eval=args.do_eval,
                                             eval_steps=args.eval_steps,
                                             per_device_train_batch_size=args.per_device_train_batch_size,
                                             per_device_eval_batch_size=args.per_device_eval_batch_size,
                                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                                             eval_accumulation_steps=args.eval_accumulation_steps,
                                             learning_rate=args.learning_rate,
                                             max_steps=args.max_steps,
                                             logging_strategy="steps",
                                             logging_steps=args.logging_steps,
                                             save_strategy="steps",
                                             save_steps=args.save_steps,
                                             load_best_model_at_end=True, # if False=> last checkpoint, if True => best checkpoint
                                             seed=args.seed,
                                             lr_scheduler_type="linear",
                                             warmup_ratio=0.05,
                                             metric_for_best_model="smatch",
                                             greater_is_better=True,
                                             save_total_limit=args.save_total_limit,)

    # 3. load dataset
    train_datasets = []
    for train_task in args.train_tasks:
        train_datasets.append(load_tokenized_dataset(train_task, tokenizer, "train", args.max_length))
    train_dataset = ConcatDataset(train_datasets)

    eval_dataset = load_tokenized_dataset(task_name=args.eval_task, tokenizer=tokenizer, split="test", max_len=512)
    test_dataset = load_tokenized_dataset(task_name=args.test_task, tokenizer=tokenizer, split="test", max_len=512)

    eval_finetuning_dataset = load_tokenized_dataset(task_name=args.eval_task, tokenizer=tokenizer, split="dev",
                                                     max_len=512, num_examples=args.dev_k_size)
    test_finetuning_dataset = load_tokenized_dataset(task_name=args.test_task, tokenizer=tokenizer, split="dev_shuffled",
                                                     max_len=512, num_examples=args.test_k_size * args.num_evaluations)
    test_finetuning_datasets = split_to_subset(test_finetuning_dataset, args.num_evaluations)

    prediction_out_dir = PROJECT_DIR / "predictions" / args.output_dir
    if args.prediction_sudo_name is not None : # if not specified, save to output_dir
        prediction_out_dir = prediction_out_dir / args.prediction_sudo_name

    # 4. train model
    trainer = CustomSeq2SeqTrainer(prediction_out_dir=prediction_out_dir,
                                   model=model,
                                   args=training_args,
                                   train_dataset=train_dataset,
                                   eval_dataset=eval_dataset,
                                   data_collator=data_collator,
                                   tokenizer=tokenizer,
                                   eval_finetuning_dataset=eval_finetuning_dataset,
                                   max_len=args.max_length,
                                   num_evaluations=args.num_evaluations,
                                   callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
                                   eval_finetune_lr=args.eval_finetune_lr,
                                   finetuning_batch_size=args.finetuning_batch_size)

    if args.max_steps > 0:
        train_model(trainer, args.resume_from_checkpoint)
    else: # if max_steps is 0, evaluate model
        load_model(trainer, args.resume_from_checkpoint)

    # 5. evaluate model
    kshot_finetune = True if args.test_k_size > 0 else False
    evaluate_model(trainer,
                   test_dataset,
                   test_finetuning_datasets,
                   args.num_evaluations,
                   "test",
                   kshot_finetune,
                   args.adaptation_steps)



if __name__ == '__main__':

    argparse = ArgumentParser()
    argparse.add_argument("--model_name", type=str, default="facebook/mbart-large-50")
    argparse.add_argument("--per_device_train_batch_size", type=int, default=2)
    argparse.add_argument("--per_device_eval_batch_size", type=int, default=4)
    argparse.add_argument("--gradient_accumulation_steps", type=int, default=2)
    argparse.add_argument("--eval_accumulation_steps", type=int, default=2)
    argparse.add_argument("--learning_rate", type=float, default=3e-5)
    argparse.add_argument("--max_steps", type=int, default=10)
    argparse.add_argument("--max_length", type=int, default=512)
    argparse.add_argument("--logging_steps", type=int, default=1)
    argparse.add_argument("--save_steps", type=int, default=10)
    argparse.add_argument("--seed", type=int, default=42)
    argparse.add_argument("--do_eval", type=bool, default=True)
    argparse.add_argument("--eval_steps", type=int, default=10)
    argparse.add_argument("--train_tasks", default=["en-amr", "de-amr"], nargs='*', help="task task names")
    argparse.add_argument("--eval_task", type=str, default="es-amr")
    argparse.add_argument("--test_task", type=str, default="fr-amr")
    argparse.add_argument("--output_dir", type=str, default="test_trainer")
    argparse.add_argument("--resume_from_checkpoint", type=bool, default=False)
    argparse.add_argument("--dev_k_size", type=int, default=16, help="number of examples to use for kshot eval, if 0, use standard eval without finetuning")
    argparse.add_argument("--test_k_size", type=int, default=16, help="number of examples to use for kshot eval, if 0, use standard eval without finetuning")
    argparse.add_argument("--num_evaluations", type=int, default=1, help="number of times to do kshot-finetuning eval")
    argparse.add_argument("--early_stopping_patience", type=int, default=8, help="patience for early stopping")
    argparse.add_argument("--eval_finetune_lr", type=float, default=1e-5, help="kshot finetune learning rate for test")
    argparse.add_argument("--save_total_limit", type=int, default=4, help="number of checkpoints to save")
    argparse.add_argument("--adaptation_steps", type=int, default=2, help="number of adaptation steps for kshot finetuning")
    argparse.add_argument("--prediction_sudo_name", type=str, default=None, help="directory to save predictions to distinguish between different runs")
    argparse.add_argument("--finetuning_batch_size", type=int, default=4, help="batch size to optimize finetuning loss")
    args = argparse.parse_args()

    if args.test_k_size == 0:
        logger.info("test_k_size is 0, set num_evaluations to 1")
        args.num_evaluations = 1

    pp  = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    main(args)
