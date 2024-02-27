from transformers import MBartForConditionalGeneration
from transformers.modeling_outputs import TokenClassifierOutput, Seq2SeqLMOutput
from torch import nn
from typing import Optional, Tuple, Union
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.utils import add_start_docstrings_to_model_forward, logging
from transformers.models.mbart.modeling_mbart import MBART_INPUTS_DOCSTRING, MBART_GENERATION_EXAMPLE
from transformers.utils import replace_return_docstrings, add_end_docstrings

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/mbart-large-50"
_CONFIG_FOR_DOC = "MBartConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

MBART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mbart-large-50",
    # See all MBART models at https://huggingface.co/models?filter=mbart
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens


class MBart4MultiTask(MBartForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.wsd_num_labels = None
        self.wsd_classifier = None
        self.ner_num_labels = None
        self.ner_classifier = None
        self.dropout = nn.Dropout(self.config.classifier_dropout)

    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(MBART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task_type: Optional[str] = None,
    ) -> Union[Seq2SeqLMOutput, TokenClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # mbart forward
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(input_ids=input_ids,
                                                 attention_mask=attention_mask,
                                                 head_mask=head_mask,
                                                 inputs_embeds=inputs_embeds,
                                                 output_attentions=output_attentions,
                                                 output_hidden_states=output_hidden_states,
                                                 return_dict=return_dict,)

        # this part of code is obsolete and should be removed in the future
        if task_type in ['wsd', 'ner']:
            sequence_output = self.dropout(encoder_outputs[0])
            logits = self.wsd_classifier(sequence_output) if task_type == 'wsd' else self.ner_classifier(sequence_output)
            num_labels = self.wsd_num_labels if task_type == 'wsd' else self.ner_num_labels

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

            if not return_dict:
                output = (logits,) + encoder_outputs[1:]  #TOOD: [2:] ??
                return ((loss,) + output) if loss is not None else output

            return TokenClassifierOutput(loss=loss,
                                         logits=logits,
                                         hidden_states=encoder_outputs.hidden_states,
                                         attentions=encoder_outputs.attentions,
                                         )

        else:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

            decoder_outputs = self.model.decoder(input_ids=decoder_input_ids,
                                                 attention_mask=decoder_attention_mask,
                                                 encoder_hidden_states=encoder_outputs[0], # is it encoder_outputs[0] or encoder_outputs.last_hidden_state?
                                                 encoder_attention_mask=attention_mask,
                                                 head_mask=decoder_head_mask,
                                                 cross_attn_head_mask=cross_attn_head_mask,
                                                 past_key_values=past_key_values,
                                                 inputs_embeds=decoder_inputs_embeds,
                                                 use_cache=use_cache,
                                                 output_attentions=output_attentions,
                                                 output_hidden_states=output_hidden_states,
                                                 return_dict=return_dict,
                                                 )

            lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias

            masked_lm_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

