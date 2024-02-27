import copy
from dataclasses import dataclass
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList
)


@dataclass
class PrepareOutput(ModelOutput):
    input_ids: Optional[torch.LongTensor] = None
    logits_processor: Optional[LogitsProcessorList] = None
    stopping_criteria: Optional[StoppingCriteriaList] = None
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[Union[int, List[int]]] = None
    output_scores: Optional[bool] = None
    return_dict_in_generate: Optional[bool] = None
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

    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()

    # priority: `generation_config` argument > `model.generation_config` (the default generation config)
    if generation_config is None:
        # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
        # two conditions must be met
        # 1) the generation config must have been created from the model config (`_from_model_config` field);
        # 2) the generation config must have seen no modification since its creation (the hash is the same).
        if self.generation_config._from_model_config and self.generation_config._original_object_hash == hash(
            self.generation_config
        ):
            new_generation_config = GenerationConfig.from_model_config(self.config)
            if new_generation_config != self.generation_config:
                warnings.warn(
                    "You have modified the pretrained model configuration to control generation. This is a"
                    " deprecated strategy to control generation and will be removed soon, in a future version."
                    " Please use and modify the model generation configuration (see"
                    " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                )
                self.generation_config = new_generation_config
        generation_config = self.generation_config

    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    generation_config.validate()
    self._validate_model_kwargs(model_kwargs.copy())

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not self.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if generation_config.max_new_tokens is not None:
        if not has_default_max_length and generation_config.max_length is not None:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length
    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. determine generation mode
    generation_mode = self._get_generation_mode(generation_config, assistant_model)

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
    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    # 9. prepare stopping criteria
    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )

    return PrepareOutput(
        input_ids = input_ids,
        logits_processor = logits_processor,
        stopping_criteria = stopping_criteria,
        pad_token_id = generation_config.pad_token_id,
        eos_token_id = generation_config.eos_token_id,
        output_scores = generation_config.output_scores,
        return_dict_in_generate = generation_config.return_dict_in_generate,
        synced_gpus = synced_gpus,
        streamer = streamer,
        model_kwargs = model_kwargs
    )

@torch.no_grad()
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
    **model_kwargs,
) -> Tuple[torch.LongTensor, torch.LongTensor, Dict[str, Any]]:
    #init values
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

    # prepare model inputs
    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # forward pass to get next token
    outputs = self(
        **model_inputs,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
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
