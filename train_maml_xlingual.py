from argparse import ArgumentParser
from transformers import AutoTokenizer, MBartConfig
from settings import PARSING_TASKS
from torch.utils.data.dataloader import DataLoader
from modeling.MBart4MultiTask import MBart4MultiTask
import random
from trainers.MAMLAmrTrainer import MAMLAmrTrainer
from TaskSampler import EpisodeSampler
import torch
import wandb
import numpy as np
from load_data import load_dataset, load_dataloader
from settings import logger, TASK2PATH
import pprint
from torch.utils.data import Subset


def split_to_subset(dataset, num_subsets) -> list:
    """
    split a dataset into num_subsets
    """

    if len(dataset) == 0:
        return [dataset]

    assert len(dataset) >= num_subsets, f"dataset size {len(dataset)} is smaller than num_subsets {num_subsets}"

    subset_size = len(dataset) // num_subsets
    subset_indexes = [range(i * subset_size, (i + 1) * subset_size) for i in range(num_subsets)]
    subsets = [Subset(dataset, indexes) for indexes in subset_indexes]

    return subsets


def main(args):
    # 0. Init some variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. load tokenizer
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    default_special_tokens = tokenizer.additional_special_tokens

    if args.is_old_version:
        extended_special_tokens = default_special_tokens + ["amr", "ucca"]
    else:
        extended_special_tokens = default_special_tokens + PARSING_TASKS
    tokenizer.add_special_tokens({'additional_special_tokens': extended_special_tokens})
    # 2. load datasets
    dev_task = args.dev_task
    test_task = args.test_task

    train_datasets = {task: load_dataset(task, "train", max_len=args.max_length, tokenizer=tokenizer,
                                         use_cached_data=args.use_cached_data) for task in args.train_tasks}
    dev_dataset = {dev_task: load_dataset(dev_task, "test", max_len=args.max_length, tokenizer=tokenizer,
                                          use_cached_data=args.use_cached_data)}
    test_dataset = {test_task: load_dataset(test_task, "test", max_len=args.max_length, tokenizer=tokenizer,
                                            use_cached_data=args.use_cached_data)}

    # load k examples for finetuning before evaluation (shuffling first and save cache for reproduction)
    logger.info("loading finetuning datasets")
    dev_k_finetune_dataset = {
        dev_task: load_dataset(dev_task, "dev", args.max_length, shuffle=True, tokenizer=tokenizer,
                               num_data=args.dev_k_size, use_cached_data=args.use_cached_data)}
    test_k_finetune_dataset = {
        test_task: load_dataset(test_task, "dev_shuffled", args.max_length, shuffle=False, tokenizer=tokenizer,
                                num_data=args.test_k_size * args.num_evaluations, use_cached_data=args.use_cached_data)}

    # divide k finetune datasets into subsets
    test_k_finetune_dataset = {test_task: split_to_subset(dataset, args.num_evaluations) for test_task, dataset in
                               test_k_finetune_dataset.items()}

    # load model
    logger.info("Loading model")
    if args.random_initialize_mbart:
        # use randomly initialized mdoel
        config = MBartConfig()
        model = MBart4MultiTask(config)
    else:
        model = MBart4MultiTask.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # create dataloader
    logger.info("create train dataloaders")
    train_dataloaders = {
        task: load_dataloader(task, dataset, args.per_device_train_batch_size * 2, model=model, tokenizer=tokenizer,
                              max_length=args.max_length) for task, dataset in train_datasets.items()}
    # create episode sampler for train dataset

    logger.info("create episode samplers")
    train_episode_samplers = {task: EpisodeSampler(dataloader, device=device) for task, dataloader in
                              train_dataloaders.items()}

    # prepare dev / test dataloaders
    logger.info("create dataloaders for test / finetuning sets..")
    dev_eval_dataloader = {
        dev_task: load_dataloader(dev_task, dev_dataset[dev_task], args.per_device_eval_batch_size, split="test",
                                  max_length=args.max_length)}
    test_eval_dataloader = {
        test_task: load_dataloader(test_task, test_dataset[test_task], args.per_device_eval_batch_size, split="test",
                                   max_length=args.max_length)}
    dev_finetuning_dataloader = {
        dev_task: load_dataloader(dev_task, dev_k_finetune_dataset[dev_task], args.per_device_train_batch_size,
                                  split="dev",
                                  max_length=args.max_length)}
    test_finetuning_dataloaders = {test_task: [
        load_dataloader(test_task, test_k_finetune_dataset, args.per_device_train_batch_size, split="dev",
                        max_length=args.max_length)
        for test_k_finetune_dataset in test_k_finetune_dataset[test_task]]}

    # 4. create trainer
    logger.info("Creating trainer")
    trainer = MAMLAmrTrainer(model=model,
                             device=device,
                             train_episode_samplers=list(train_episode_samplers.values()),
                             dev_finetuning_dataloaders=dev_finetuning_dataloader,
                             dev_eval_dataloaders=dev_eval_dataloader,
                             tokenizer=tokenizer,
                             eval_steps=args.eval_steps,
                             save_steps=args.save_steps,
                             checkpoint_dir=args.checkpoint_dir,
                             inner_loop_lr=args.inner_loop_lr,
                             outer_loop_lr=args.outer_loop_lr,
                             max_steps=args.max_steps,
                             adaptation_steps=args.adaptation_steps,
                             max_length=args.max_length,
                             disable_tqdm=args.disable_tqdm,
                             silent_amr_postprocessing=args.silent_amr_postprocessing,
                             patience=args.patience,
                             lr_scheduler_max_steps=args.lr_scheduler_max_steps,
                             save_total_limit=args.save_total_limit,
                             verbose=args.verbose,
                             task_batch_size=args.task_batch_size,
                             eval_finetune_lr=args.test_finetune_lr,
                             prediction_out_dir=args.prediction_sudo_name,
                             finetuning_batch_size=args.finetuning_batch_size,
                             )

    # 5. train a model
    logger.info("Training model starts")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if args.do_test:
        # 6. evaluate the model
        kshot_finetune = True if args.test_k_size > 0 else False  # if kshot_size is 0, do not finetune before evaluation
        trainer.load_from_checkpoint(load_best_model=True)
        eval_metric = {}
        for i in range(args.num_evaluations):
            finetune_dataloader = test_finetuning_dataloaders[test_task][i]
            logger.info(f"Running test evaluation {i + 1}/{args.num_evaluations}")
            metric = trainer.evaluate(finetuning_dataloader=finetune_dataloader,
                                      eval_dataloader=test_eval_dataloader[test_task],
                                      metric_key_prefix=f"test{i + 1}",
                                      split="test",
                                      lr=args.test_finetune_lr,
                                      kshot_finetune=kshot_finetune)
            eval_metric.update(metric)

        mean_smatch = np.mean([eval_metric[f"test{i + 1}_smatch"] for i in range(args.num_evaluations)])
        smatch_std = np.std([eval_metric[f"test{i + 1}_smatch"] for i in range(args.num_evaluations)])
        trainer.eval_log({"test_smatch": mean_smatch, "test_std": smatch_std}, prefix="test")



if __name__ == '__main__':

    argparse = ArgumentParser()
    argparse.add_argument("--model_name_or_path", type=str, default="facebook/mbart-large-50")
    argparse.add_argument("--per_device_train_batch_size", type=int, default=2)
    argparse.add_argument("--per_device_eval_batch_size", type=int, default=4)
    argparse.add_argument("--inner_loop_lr", type=float, default=1e-5)
    argparse.add_argument("--outer_loop_lr", type=float, default=3e-5)
    argparse.add_argument("--max_steps", type=int, default=10)
    argparse.add_argument("--lr_scheduler_max_steps", type=int, default=-1,
                          help="max_steps for lr scheduler, give negative value to ignore this argument")
    argparse.add_argument("--max_length", type=int, default=512)
    argparse.add_argument("--save_steps", type=int, default=10)
    argparse.add_argument("--seed", type=int, default=42)
    argparse.add_argument("--dropout", type=float, default=0.2)
    argparse.add_argument("--eval_steps", type=int, default=5)
    argparse.add_argument("--train_tasks", default=["en-amr", "de-amr"], nargs='*', help="task task names")
    argparse.add_argument("--dev_task", default="es-amr", help="dev task names")
    argparse.add_argument("--test_task", default="fr-amr", help="test task names")
    argparse.add_argument("--checkpoint_dir", type=str, default="test_maml4multitask")
    argparse.add_argument("--resume_from_checkpoint", type=bool, default=False)
    argparse.add_argument("--test_k_size", type=int, default=16,
                          help="number of examples to use for kshot eval, if set to 0, standard eval without finetuning")
    argparse.add_argument("--adaptation_steps", type=int, default=2, help="number of inner loop steps")
    argparse.add_argument("--disable_tqdm", type=bool, default=False)
    argparse.add_argument("--silent_amr_postprocessing", type=bool, default=False,
                          help="whether to logger.info logs for amr processing or not")
    argparse.add_argument("--do_test", type=bool, default=True, help="whether to do test eval or not")
    argparse.add_argument("--num_evaluations", type=int, default=3, help="number of evaluations to run")
    argparse.add_argument("--test_finetune_lr", type=float, default=-1,
                          help="inner loop lr for test evaluation, give negative value to ignore this argument")
    argparse.add_argument("--patience", type=int, default=20, help="patience for early stopping")
    argparse.add_argument("--save_total_limit", type=int, default=None, help="number of checkpoints to save")
    argparse.add_argument("--use_cached_data", type=bool, default=False,
                          help="when true, use cached data when available")
    argparse.add_argument("--verbose", type=bool, default=True,
                          help="whether of not to show detailed loss for each task")
    argparse.add_argument("--task_batch_size", type=int, default=None,
                          help="number of tasks per step, if None, use all tasks per step")
    argparse.add_argument("--dev_k_size", type=int, default=None, help="number of k_size for dev")
    argparse.add_argument("--prediction_sudo_name", type=str, default=None,
                          help="directory to save predictions (optional)")
    argparse.add_argument("--finetuning_batch_size", type=int, default=4, help="batch size to optimize finetuning loss")
    argparse.add_argument("--random_initialize_mbart", type=bool, default=False, help="whether to use pretrained mbart weights or not")
    argparse.add_argument("--is_old_version", type=bool, default=False, help="(temp) whether to add cstp as special token or not")
    args = argparse.parse_args()

    logger.info("Initializing wandb")
    wandb.init(project="maml-amr", config=vars(args), name=args.checkpoint_dir)

    is_invalid_task_batch_size = args.task_batch_size is not None and args.task_batch_size > len(args.train_tasks)

    # set default values when not specified
    if args.lr_scheduler_max_steps < 0:
        logger.info("lr_scheduler_max_steps is not specified, set to max_steps")
        args.lr_scheduler_max_steps = args.max_steps  # set to max_steps if not specified

    if args.test_finetune_lr < 0:
        logger.info("test_finetune_lr is not specified, set to inner_loop_lr")
        args.test_finetune_lr = args.inner_loop_lr  # set to inner_loop_lr if not specified

    if args.task_batch_size is None or is_invalid_task_batch_size:
        logger.info("task_batch_size is not specified or invalid, set to len(train_tasks)")
        args.task_batch_size = len(args.train_tasks)  # use all tasks per step if not specified

    if args.dev_k_size is None:
        logger.info("dev_k_size is not specified, set to per_device_train_batch_size")
        args.dev_k_size = args.per_device_train_batch_size

    if args.test_k_size == 0:
        logger.info("kshot_size is 0, set num_evaluations to 1")
        args.num_evaluations = 1

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))

    main(args)
