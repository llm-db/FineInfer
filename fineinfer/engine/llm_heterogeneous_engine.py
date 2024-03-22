from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import PreTrainedModel
import torch
from transformers.generation.logits_process import (
    LogitsProcessorList
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList
)


def generate_step(
    self: PreTrainedModel,
    unfinished_sequences: torch.LongTensor,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    f_input_ids: torch.LongTensor = None,
    f_attention_mask: Optional[torch.Tensor] = None,
    f_position_ids: Optional[torch.LongTensor] = None,
    f_labels: Optional[torch.LongTensor] = None,
    **model_kwargs,
) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, Dict[str, Any]]:
    #init values
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

    # prepare model inputs
    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # forward pass to get next token
    f_loss, f_logits, outputs = self.forward_heterogeneous(
        **model_inputs,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
        f_input_ids=f_input_ids,
        f_attention_mask=f_attention_mask,
        f_position_ids=f_position_ids,
        f_labels=f_labels,
    )

    next_token_logits = outputs.logits[:, -1, :]

    # pre-process distribution
    next_tokens_scores = logits_processor(input_ids, next_token_logits)

    # argmax
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)

    # finished sentences should have their next token be a padding token
    if eos_token_id is not None:
        if pad_token_id is None:
            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

    # update generated ids, model inputs, and length for next step
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

    model_kwargs = self._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
    )

    if eos_token_id_tensor is not None:
        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )

    return f_loss, f_logits, unfinished_sequences, input_ids, model_kwargs
