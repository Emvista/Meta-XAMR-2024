from torch.utils.data import IterableDataset
from itertools import cycle

class EpisodeSampler:

    def __init__(self, dataloader, device):

        self.dataloader = dataloader
        self.device = device
        self.random_swap_input_labels = False
        self.src_lang = dataloader.dataset.src_lang
        self.tgt_task = dataloader.dataset.tgt_task

        if self.tgt_task in ["wsd", "ner"]:
            self.task_type = self.tgt_task

        else:
            self.task_type = "seq2seq"

        # if iterable dataset, make it cycle
        if isinstance(dataloader.dataset, IterableDataset):
            self.dataloader = cycle(self.dataloader)

    def fast_forward_to_step(self, curr_step):
        print("Fast forwarding dataloader to step {}".format(curr_step))
        # for i in range(curr_step):
        #     next(self.dataloader)
        i = 0
        for _ in self.dataloader:
            i += 1

            if i == curr_step:
                break

    def sample(self):
        # divide sampled batch to support and query
        samples = next(iter(self.dataloader))

        num_inputs = samples['input_ids'].shape[0]
        support = {key: value[:num_inputs//2].to(self.device) for key, value in samples.items()}
        query = {key: value[num_inputs//2:].to(self.device) for key, value in samples.items()}

        return support, query

