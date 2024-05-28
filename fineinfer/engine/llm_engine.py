import copy
from dataclasses import dataclass
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from transformers import PreTrainedModel
from transformers.cache_utils import StaticCache
from transformers.utils import ModelOutput, is_torchdynamo_compiling, logging
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList
)


logger = logging.get_logger(__name__)

NEED_SETUP_CACHE_CLASSES_MAPPING = {
    "static": StaticCache,
}


@dataclass
class PrepareOutput(ModelOutput):
    input_ids: Optional[torch.LongTensor] = None
    logits_processor: Optional[LogitsProcessorList] = None
    logits_warper: Optional[LogitsProcessorList] = None
    stopping_criteria: Optional[StoppingCriteriaList] = None
    generation_config: Optional[GenerationConfig] = None
    synced_gpus: Optional[bool] = None
    streamer: Optional["BaseStreamer"] = None
    model_kwargs: Optional[Dict[str, Any]] = None

@dataclass
class BatchMeta(ModelOutput):
    prompt_lens: Optional[torch.LongTensor] = None
    gen_lens: Optional[torch.LongTensor] = None
    cur_lens: Optional[torch.LongTensor] = None

@torch.no_grad()
def prepare_inputs_and_config(
    self: PreTrainedModel,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> PrepareOutput:

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()
    tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
    generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
    self._validate_model_kwargs(model_kwargs.copy())

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    # decoder-only models must use left-padding for batched generation.
    if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    if generation_config.cache_implementation is not None and model_kwargs.get("past_key_values") is not None:
        raise ValueError(
            "Passing both `cache_implementation` (used to initialize certain caches) and `past_key_values` (a "
            "Cache object) is unsupported. Please use only one of the two."
        )
    elif generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
        if not self._supports_cache_class:
            raise ValueError(
                "This model does not support the `cache_implementation` argument. Please check the following "
                "issue: https://github.com/huggingface/transformers/issues/28981."
            )
        if generation_config.cache_implementation == "static":
            if not self._supports_static_cache:
                raise ValueError(
                    "This model does not support `cache_implementation='static'`. Please check the following "
                    "issue: https://github.com/huggingface/transformers/issues/28981"
                )
            model_kwargs["past_key_values"] = self._get_static_cache(batch_size, generation_config.max_length)

    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare distribution pre_processing samplers
    prepared_logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    # 9. prepare stopping criteria
    prepared_stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs   
    )

    # 10. go into different generation modes
    # 11. prepare logits warper
    prepared_logits_warper = (
        self._get_logits_warper(generation_config) if generation_config.do_sample else None
    )

    # 12. expand input_ids with `num_return_sequences` additional sequences per batch
    input_ids, model_kwargs = self._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_return_sequences,
        is_encoder_decoder=self.config.is_encoder_decoder,
        **model_kwargs,
    )

    return PrepareOutput(
        input_ids = input_ids,
        logits_processor = prepared_logits_processor,
        logits_warper = prepared_logits_warper,
        stopping_criteria = prepared_stopping_criteria,
        generation_config = generation_config,
        synced_gpus = synced_gpus,
        streamer = streamer,
        model_kwargs = model_kwargs
    )

@torch.no_grad()
def generate_step(
    self: PreTrainedModel,
    unfinished_sequences: torch.LongTensor,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList],
    stopping_criteria: Optional[StoppingCriteriaList],
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    **model_kwargs,
) -> Tuple[torch.LongTensor, torch.LongTensor, Dict[str, Any]]:
    #init values
    pad_token_id = generation_config.pad_token_id
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
        raise ValueError(
            "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
            f"{logits_warper})."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # prepare model inputs
    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # forward pass to get next token
    outputs = self(
        **model_inputs,
        return_dict=True,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    next_token_logits = outputs.logits[:, -1, :]

    # pre-process distribution
    next_token_scores = logits_processor(input_ids, next_token_logits)
    if do_sample:
        next_token_scores = logits_warper(input_ids, next_token_scores)

    # token selection
    if do_sample:
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_tokens = torch.argmax(next_token_scores, dim=-1)

    # finished sentences should have their next token be a padding token
    if has_eos_stopping_criteria:
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

    # update generated ids, model inputs, and length for next step
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    model_kwargs = self._update_model_kwargs_for_generation(
        outputs,
        model_kwargs,
        is_encoder_decoder=self.config.is_encoder_decoder,
    )

    unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)

    return unfinished_sequences, input_ids, model_kwargs

@torch.no_grad()
def add_new_request(
    unfinished_sequences: torch.LongTensor,
    input_ids: torch.LongTensor,
    model_kwargs: Dict[str, Any],
    new_unfinished_sequences: torch.LongTensor,
    new_input_ids: torch.LongTensor,
    new_model_kwargs: Dict[str, Any],
)  -> Tuple[torch.LongTensor, torch.LongTensor, Dict[str, Any]]:
    device = torch.cuda.current_device()

    unfinished_sequences = torch.cat((unfinished_sequences, new_unfinished_sequences), dim=0)

    cur_input_ids = torch.zeros(new_input_ids.shape[0], input_ids.shape[1], dtype=torch.long, device=device)
    cur_input_ids[:, -new_input_ids.shape[1]:] = new_input_ids
    input_ids = torch.cat((input_ids, cur_input_ids), dim=0)

    cur_attention_mask = torch.zeros(new_input_ids.shape[0], input_ids.shape[1], dtype=torch.long, device=device)
    cur_attention_mask[:, -new_input_ids.shape[1]:] = new_model_kwargs['attention_mask']
    model_kwargs['attention_mask'] = torch.cat((model_kwargs['attention_mask'], cur_attention_mask), dim=0)

    if model_kwargs['use_cache']:
        model_kwargs['past_key_values'] = list(model_kwargs['past_key_values'])
        for layer_idx, past_key_value in enumerate(model_kwargs['past_key_values']):
            past_key, past_value = past_key_value
            new_past_key, new_past_value = new_model_kwargs['past_key_values'][layer_idx]

            cur_past_key = torch.zeros_like(past_key, dtype=past_key.dtype, device=device)[:new_input_ids.shape[0]]
            cur_past_value = torch.zeros_like(past_value, dtype=past_value.dtype, device=device)[:new_input_ids.shape[0]]
            cur_past_key[:, :, -new_past_key.shape[2]:, :] = new_past_key
            cur_past_value[:, :, -new_past_value.shape[2]:, :] = new_past_value

            new_past_key = torch.cat((past_key, cur_past_key), dim=0)
            new_past_value = torch.cat((past_value, cur_past_value), dim=0)
            model_kwargs['past_key_values'][layer_idx] = (new_past_key, new_past_value)
        model_kwargs['past_key_values'] = tuple(model_kwargs['past_key_values'])

    return unfinished_sequences, input_ids, model_kwargs

@torch.no_grad()
def remove_old_request(
    unfinished_sequences: torch.LongTensor,
    input_ids: torch.LongTensor,
    model_kwargs: Dict[str, Any],
    batch_meta: BatchMeta,
    output_ids: List[torch.LongTensor],
) :
    device = torch.cuda.current_device()
    masks = torch.ones(unfinished_sequences.shape, dtype=torch.bool, device=device)
    for _id in range(len(batch_meta.cur_lens)):
        if batch_meta.cur_lens[_id] == (batch_meta.prompt_lens[_id] + batch_meta.gen_lens[_id]):
            masks[_id] = False

    unfinished_sequences = unfinished_sequences[masks]
    batch_meta.prompt_lens = batch_meta.prompt_lens[masks]
    batch_meta.gen_lens = batch_meta.gen_lens[masks]
    batch_meta.cur_lens = batch_meta.cur_lens[masks]

    for _id, _mask in enumerate(masks):
        if _mask == False:
            output_ids.append(input_ids[_id])
    input_ids = input_ids[masks]

    if len(input_ids):
        max_sequence_length = torch.max(batch_meta.cur_lens)
        input_ids = input_ids[:, -max_sequence_length:]
        model_kwargs['attention_mask'] = model_kwargs['attention_mask'][masks][:, -max_sequence_length:]
        if model_kwargs['use_cache']:
            model_kwargs['past_key_values'] = list(model_kwargs['past_key_values'])
            for layer_idx, past_key_value in enumerate(model_kwargs['past_key_values']):
                past_key, past_value = past_key_value
                model_kwargs['past_key_values'][layer_idx] = (past_key[masks][:, :, 1-max_sequence_length:, :], \
                    past_value[masks][:, :, 1-max_sequence_length:, :])

            model_kwargs['past_key_values'] = tuple(model_kwargs['past_key_values'])

    return unfinished_sequences, input_ids, model_kwargs, batch_meta, output_ids
