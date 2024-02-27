import torch

def truncate_to_max_input(batch):
    """
    Truncate tokenized input and labels to the maximum length in the batch
    sine the whole dataset is padded to the same length, we truncate each batch
    to the maximum length in the batch to save momory and speed up training
    """

    # Convert the 'input_ids', 'attention_mask', and 'labels' to PyTorch tensors
    input_ids = torch.stack([example['input_ids'] for example in batch])
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    labels = torch.stack([example['labels'] for example in batch])

    # Find the maximum length across the batch using the attention_mask
    max_len = attention_mask.sum(dim=1).max().item()

    # Truncate each example in the batch to the maximum length
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]
    labels = labels[:, :max_len]

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def truncate_to_max_input_n_label(batch):
    """
    Truncate tokenized input and labels to the maximum length in the batch
    sine the whole dataset is padded to the same length, we truncate each batch
    to the maximum length in the batch to save momory and speed up training
    :param batch:
    :return:
    """
    # Convert the 'input_ids', 'attention_mask', and 'labels' to PyTorch tensors
    input_ids = torch.stack([example['input_ids'] for example in batch])
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    labels = torch.stack([example['labels'] for example in batch])

    # Find the maximum length across the batch using the attention_mask
    input_max_len = attention_mask.sum(dim=1).max().item()
    label_max_len = (labels != -100).sum(dim=1).max().item() # 1 = pad token

    # Truncate each example in the batch to the maximum length
    input_ids = input_ids[:, :input_max_len]
    attention_mask = attention_mask[:, :input_max_len]
    labels = labels[:, :label_max_len]

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


class Seq2SeqCollator:
    """
    Receive a batch of untokenized inputs and labels and
    tokenize and pad them for seq2seq task (max length of inputs))
    """
    def __init__(self, tokenizer, model, max_length, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.model = model
        self.return_tensors = return_tensors
        self.max_length = max_length


    def __call__(self, batch):
        inputs = [example[0] for example in batch]
        labels = [example[1] for example in batch]

        tok_inputs = self.tokenizer(inputs,
                                    text_target=labels,
                                    return_tensors=self.return_tensors,
                                    padding="longest", truncation=True,
                                    max_length=self.max_length)

        # replace padding token id (1) with -100 for labels
        tok_inputs["labels"][tok_inputs["labels"] == 1] = -100

        # prepare decoder_input_ids
        if (
                tok_inputs["labels"] is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=tok_inputs["labels"])
            tok_inputs["decoder_input_ids"] = decoder_input_ids

        return tok_inputs






