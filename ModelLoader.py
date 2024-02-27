from transformers import AutoTokenizer
from transformers import MBartForConditionalGeneration
from transformers import M2M100ForConditionalGeneration
from settings import PARSING_TASKS
import torch
import learn2learn as l2l
from torch import nn, optim


class AmrTrainerLoader:

    @staticmethod
    def load_model_n_tokenizer(model_name, extra_tokens=PARSING_TASKS):
        print(f"=============Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"=============Loading model: {model_name}")
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        print(f"=============Adding special tokens: {extra_tokens}")
        tokenizer.add_special_tokens({'additional_special_tokens': extra_tokens},
                                     replace_special_tokens=False)
        model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    @staticmethod
    def load_best_model(model_name, checkpoint, device):
        model, tokenizer = AmrTrainerLoader.load_model_n_tokenizer(model_name)
        print(f"==================loading model to {device}==================")
        model.to(device)

        print(f"==================loading checkpoint from {checkpoint}==================")
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, tokenizer

    @staticmethod
    def load_checkpoint(model_name, checkpoint, learning_rate, device):
        model, tokenizer = AmrTrainerLoader.load_model_n_tokenizer(model_name)
        print("======== Loading model from checkpoint: {} ================".format(checkpoint))
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        try:
            current_step = checkpoint['global_step']
        except:
            current_step = "test"

        return model, tokenizer, optimizer, current_step

