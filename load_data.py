from pathlib import Path
from settings import MBART_LANG_CODEMAP, MULTI_LANGS, PARSING_TASKS
import pickle
from custom_datasets.CustomDataset import TokenizedSeq2seqDataset, Seq2seqIterDataset, Seq2seqDataset
from torch.utils.data import Subset
from settings import TASK2PATH, SEQ2SEQ_TASKS, TOKEN_CLS_TASKS
from torch.utils.data import DataLoader
from custom_datasets.custom_collate_fn import truncate_to_max_input, truncate_to_max_input_n_label
# from transformers import DataCollatorForSeq2Seq
from custom_datasets.custom_collate_fn import Seq2SeqCollator
from copy import deepcopy
from torch.utils.data import IterableDataset


def load_dataloader(task_name, dataset, batch_size, split="train", model=None, tokenizer=None, max_length=512, shuffle=None):
    """
    Depending on the task type (ner, seq2seq, wsd, etc) and split (train, dev, test),
    return the dataloader for the dataset
    :param task_name: "en-ner", "en-es", "en-ucca", "en-amr", etc
    :param dataset: torch Dataset
    :param batch_size:
    :param split: "train", "dev", "test"
    :param model: model for seq2seq / train task (for decoder input)
    :param tokenizer: tokenizer for seq2seq / train task
    :param max_length: the number of tokens to truncate
    :return: DataLoader
    """
    # token classification task train set
    if task_name in TOKEN_CLS_TASKS and split=="train":
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=truncate_to_max_input, num_workers=0)

    # token classification task dev / test set
    elif task_name in TOKEN_CLS_TASKS and split!="train":
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=truncate_to_max_input, num_workers=0)

    # seq2seq train set
    elif task_name in SEQ2SEQ_TASKS and split=="train":
        assert tokenizer is not None and model is not None, "tokenizer and model must be given for seq2seq train task"
        tokenizer = deepcopy(tokenizer)

        src_lang = MBART_LANG_CODEMAP[task_name.split("-")[0]]
        tgt_lang = MBART_LANG_CODEMAP[task_name.split("-")[1]]

        # set tokenizer's src and tgt lang for each data type
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        collator = Seq2SeqCollator(tokenizer, model=model, max_length=max_length, return_tensors="pt")

        # if dataset is IterableDataset, we don't need to shuffle it (since it's already shuffled)
        if isinstance(dataset, IterableDataset):
            return DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=0)

        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=0)

    # seq2seq dev / test set or finetuning set
    elif task_name in SEQ2SEQ_TASKS and split!="train":
        shuffle = False if shuffle is None else shuffle  # False for dev/test set, True for finetuning set
        return DataLoader(batch_size=batch_size, dataset=dataset, collate_fn=truncate_to_max_input_n_label, shuffle=shuffle, num_workers=0)

    else:
        AssertionError, "task name must be one of the following: {}".format(TOKEN_CLS_TASKS + SEQ2SEQ_TASKS)

def load_cached_data(cache_filename: Path):
    if cache_filename.exists() :
        with open(cache_filename, 'rb') as f:
            try:
                cached = pickle.load(f)

            except: cached = None
        return cached

    else:
        return None

def save_data_to_cache(cache_filename: Path, data):
    cache_filename.parent.mkdir(parents=True, exist_ok=True) # Create cache directory if it doesn't exist

    with open(cache_filename, 'wb') as f:
        pickle.dump(data, f)
        print(f"save cached dataset to {cache_filename}")

def load_dataset(task_name, split, max_len, tokenizer=None, use_cached_data=True, shuffle=False, num_data=None):
    """
    Depending on the task type (ner, seq2seq, wsd, etc) and split (train, dev, test),
    load the dataset and return it
    :param task_name: name of the task (ex) en-es, en-ucca, en-amr ...)
    :param split: train, dev, test
    :param max_len: tokenizer max length
    :param tokenizer: transformers tokenizer
    :param use_cached_data: whether to use cached data or not
    :param shuffle: whether to shuffle the dataset or not (only for finetuning data)
    :param num_data: size of the dataset to use (only for finetuning data)
    :return: dataset
    """

    src_lang = task_name.split("-")[0]
    tgt_task = task_name.split("-")[1]
    dataset_path = TASK2PATH.get_path(task_name, split)
    is_translation = (src_lang in MULTI_LANGS and tgt_task in MULTI_LANGS)
    is_parsing = (src_lang in PARSING_TASKS or tgt_task in PARSING_TASKS)

    # check if the dataset is cached, if so, load it
    cached_dataset_path = dataset_path[1].parent / f"cached_{task_name}.pkl"
    dataset = load_cached_data(cached_dataset_path)

    if dataset is not None and use_cached_data:   # use cache if
        # only for seq2seq dev, test set
        print(f"use cached data from {cached_dataset_path}")
        return dataset

    # translation train set
    elif is_translation and split == "train":
        src_path = dataset_path[0].parent / "shuffled" / dataset_path[0].name
        tgt_path = dataset_path[1].parent / "shuffled" / dataset_path[1].name
        dataset = Seq2seqIterDataset(src_path=src_path, tgt_path=tgt_path)

    elif is_parsing and split == "train":
        dataset = Seq2seqDataset(src_path=dataset_path[0], tgt_path=dataset_path[1])

    # "seq2seq" dev/test set + finetuning set
    elif is_parsing and split != "train":
        dataset = TokenizedSeq2seqDataset(src_path=dataset_path[0],
                                          tgt_path=dataset_path[1],
                                          tokenizer=tokenizer,
                                          max_len=max_len,
                                          shuffle=shuffle)

    # if num_data is given, we only use the subset of the dataset
    if num_data is not None:
        dataset = Subset(dataset, list(range(num_data)))

    # save the dataset to cache
    if not is_translation: # don't save multi-lingual dataset to cache (file stream)
        save_data_to_cache(cached_dataset_path, dataset)

    return dataset