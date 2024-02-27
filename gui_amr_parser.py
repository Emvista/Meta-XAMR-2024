from consumer import FrenchAmrParser
import gradio as gr
from pathlib import Path
import torch
from transformers import MBartForConditionalGeneration, AutoConfig, AutoTokenizer
from AMR.restoreAMR import restore_amr as pp # postprocessing
from AMR.amr_utils import reverse_tokenize, tokenize_line, get_default_amr
import penman
import logging


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
class FrenchAmrParser:
    def __init__(self):
        """
        French to AMR parsing model
        """
        # load a randomly initialized model
        self.config = AutoConfig.from_pretrained('facebook/mbart-large-50')
        self.model = MBartForConditionalGeneration(self.config)

        # load a tokenizer
        special_tokens_dict = {'additional_special_tokens': ["amr", "cstp", "ucca"]}
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50', src_lang="fr_XX")
        self.tokenizer.add_special_tokens(special_tokens_dict=special_tokens_dict,
                                          replace_additional_special_tokens=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = './checkpoint/checkpoint-17500.pt'
        self.load_model(checkpoint) # load the trained model

    def load_model(self, path : str):
        logging.info("Loading model from %s", path)
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file {path} not found.")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("Model loaded successfully.")
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def parse_amr(text: str):
        # parse linearized amr graph with penman
            amr = penman.parse(text)
            amr_penman = penman.format(amr)
            return amr_penman

    def postprocess(self, prediction: str, coreference="index"):
        try:
            line = pp.preprocess(prediction, coreference)

            # Make sure char and word-level are in a similar representation now
            line = reverse_tokenize(tokenize_line(line))

            # Do some extra steps to fix some easy to fix problems
            line = pp.do_extra_steps(line)

            # Output of neural models can have non-finished edges, remove
            line = pp.remove_dangling_edges(line)

            # Extra step to make sure digits are not added to arguments
            line = pp.add_space_when_digit(line)

            # Restore variables here, also fix problems afterwards if there are any
            line = pp.convert(line)

            # The digit problem might reoccur again here
            line = pp.add_space_when_digit(line)

        except:
            logging.info("Error in postprocessing, returning a basic linearized AMR.")
            line = get_default_amr()

        return line

    def run(self, text: str):
        # set generation parameters
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(["amr"])[0]
        num_beams = 5

        # tokenize inputs
        tok = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # generate prediction
        with torch.no_grad():
            model_input = {k: v.to(self.device) for k, v in tok.items()}
            prediction = self.model.generate(**model_input,
                                             num_beams=num_beams,
                                             forced_bos_token_id=forced_bos_token_id,)
            prediction = self.tokenizer.decode(prediction[0], skip_special_tokens=True)

            # mainly restore variables from the linearized amr graph
            post_processed = self.postprocess(prediction)

            # parse linearized amr graph with penman and return it
            return str(FrenchAmrParser.parse_amr(post_processed))


if __name__ == '__main__':
    amr_parser = FrenchAmrParser()
    gr.Interface(fn=amr_parser.run, inputs="text", outputs="text").launch()
