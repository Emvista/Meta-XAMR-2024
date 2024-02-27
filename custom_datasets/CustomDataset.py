import torch
import random
from torch.utils.data import DataLoader
from utils.amr_utils import get_amr_pairs
from settings import MBART_LANG_CODEMAP

class TokenizedSeq2seqDataset(torch.utils.data.Dataset):
    """
    Reads in the whole txt file and tokenzier it (for small dataset e.g. finetuning dataset)
    """
    def __init__(self, src_path, tgt_path, tokenizer, shuffle=False, max_len=512):

        self.src_lang = src_path.suffix[1:]
        self.tgt_task = tgt_path.suffix[1:]
        self.split = src_path.parent.name

        src = open(src_path, 'r').readlines()
        tgt = open(tgt_path, 'r').readlines()

        if shuffle:  # shuffle the data for fine tuning sets
            src, tgt = self.shuffle(src, tgt)
        tokenizer.src_lang = MBART_LANG_CODEMAP[self.src_lang]
        tokenizer.tgt_lang = self.tgt_task

        self.tok_data = tokenizer(text=src, text_target=tgt, return_tensors="pt", padding='max_length', truncation=True, max_length=max_len)

        # replace padding token id (1) with -100 for labels
        self.tok_data["labels"][self.tok_data["labels"] == 1] = -100

        # sanity check to make sure the data is aligned
        if self.tok_data['input_ids'].shape[0] != self.tok_data['labels'].shape[0]:
            print("WARNING : input_ids and labels should have the same number of lines")

    def __len__(self):
        return self.tok_data['input_ids'].shape[0]

    def __getitem__(self, idx):

        return  {'input_ids': self.tok_data['input_ids'][idx],
                'attention_mask': self.tok_data['attention_mask'][idx],
                'labels': self.tok_data['labels'][idx]}

    def shuffle(self, src, tgt):

        indices = list(range(len(src)))
        random.shuffle(indices)

        shuffled_src = [src[i] for i in indices]
        shuffled_tgt = [tgt[i] for i in indices]

        return shuffled_src, shuffled_tgt


class Seq2seqDataset(torch.utils.data.Dataset):
    """
    Reads a file stream to tokenizer it on the fly (for large dataset e.g. training set)
    """
    def __init__(self, src_path, tgt_path):
        self.src_lang = src_path.suffix[1:]
        self.tgt_task = tgt_path.suffix[1:]
        self.split = src_path.parent.name

        self.src = open(src_path, 'r').readlines()
        self.tgt = open(tgt_path, 'r').readlines()

        if len(self.src) != len(self.tgt):
            print("WARNING : source and target should have the same number of lines")

    def __getitem__(self, idx):
        return self.src[idx].rstrip(), self.tgt[idx].rstrip()

    def __len__(self):
        return len(self.src)


class Seq2seqIterDataset(torch.utils.data.IterableDataset):
    """
    Reads a file stream to tokenizer it on the fly (for large dataset e.g. training set)
    """
    def __init__(self, src_path, tgt_path):
        self.src_lang = src_path.suffix[1:]
        self.tgt_task = tgt_path.suffix[1:]
        self.split = src_path.parent.name

        self.src = open(src_path, 'r')
        self.tgt = open(tgt_path, 'r')

    def __iter__(self):

        for src, tgt in zip(self.src, self.tgt):
            yield src.rstrip(), tgt.rstrip()


class AmrTokDataset(torch.utils.data.Dataset):
    """
    AmrBaseline train inputs: random k examples from mixed datasets [en-amr, de-amr, ...]
    => data should be tagged with src_lang and tgt_lang beforehand (not during the data loading)
    """
    def __init__(self, tokenizer, datapath, max_len):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset = self.get_preprocessed_dataset(datapath)

    def get_preprocessed_dataset(self, datapath):
        src_lang, tgt_lang = datapath[0].stem.split('-')
        amr_pairs = get_amr_pairs(datapath)  # pair of (src, tgt) strings
        model_inputs = [self._preprocess_graphs(src_lang, tgt_lang, amr_pair) for amr_pair in amr_pairs]
        return model_inputs


    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def _preprocess_graphs(self, src_lang, tgt_lang, amr_pair):
        tokenizer = self.tokenizer
        tokenizer.src_lang = MBART_LANG_CODEMAP[src_lang]
        tokenizer.tgt_task = MBART_LANG_CODEMAP[tgt_lang]
        src, tgt = amr_pair
        model_inputs = tokenizer(text=src, text_target=tgt,  truncation=True, return_tensors='pt', max_length=self.max_len)
        return model_inputs


class AmrSqueezedTokDataset(AmrTokDataset):
    """
    Similar to AmrDataset but it returns un-nested tensors
    e.g. tokenizer returns nested tensors like:
    {'input_ids': tensor([[  0,  10,  20,  30,  40,  50,  60,  70,  80,  90]])
    this class returns:
    {'input_ids': tensor([  0,  10,  20,  30,  40,  50,  60,  70,  80,  90])}
    """
    def __init__(self, tokenizer, datapath, max_len, random_shuffle=False, size=None):
        super().__init__(tokenizer, datapath, max_len)
        self.src_lang = datapath[0].stem.split('-')[0]
        self.tgt_task = datapath[0].stem.split('-')[1]

        if random_shuffle:
            import random
            random.shuffle(self.dataset)

        if size is not None:
            self.dataset = self.dataset[:size]

    def _preprocess_graphs(self, src_lang, tgt_lang, amr_pair):

        tokenizer = self.tokenizer
        tokenizer.src_lang = MBART_LANG_CODEMAP[src_lang]
        tokenizer.tgt_lang = MBART_LANG_CODEMAP[tgt_lang]

        src, tgt = amr_pair
        model_inputs = tokenizer(text=src, text_target=tgt,  truncation=True, return_tensors='pt', max_length=self.max_len)
        model_inputs  = {key: value.squeeze(0) for key, value in model_inputs.items()}

        return model_inputs


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from settings import TASK2PATH
    #
    # task = TASK2PATH.get_path("{}-{}".format("en", "es"), "dev")
    # tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
    # ds = SequenceDataset(src_path=task[0], tgt_path=task[1], tokenizer=tokenizer, max_len=128)
    # print(ds[1])