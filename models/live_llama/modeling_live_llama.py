import torch
from torch import nn
from transformers import LlamaForCausalLM, Cache
from transformers.activations import GELUActivation
from transformers.utils import logging

from .configuration_live_llama import LiveLlamaConfig
from ..modeling_live import build_live, LiveMixin, find_vision_indices, find_narration_indices
from models.clustering import CLAM

logger = logging.get_logger(__name__)


def find_second_last_before(v_indice, m_indices):
    if not m_indices:
        return
    v_start, v_end = v_indice
    valid_intervals = [interval for interval in m_indices if interval[1] < v_start]

    # Check if we have at least two intervals before 'b_start'
    if len(valid_intervals) >= 2:
        # Get the second last interval
        return valid_intervals[-2]


def create_custom_attention_mask(input_tensor, attention_mask, v_id, m_id, n_id, separator_id, dtype, last_k_narration=None):
    # Convert sequence to a tensor and calculate sequence length
    sequence_length = input_tensor.shape[1]

    # Adapted from modeling_llama.py self._update_causal_mask line 982
    device = input_tensor.device
    if dtype.is_floating_point:
        min_dtype = torch.finfo(dtype).min
    else:  # must be integer
        min_dtype = torch.iinfo(dtype).min
    causal_mask = torch.full(
        (sequence_length, sequence_length), fill_value=min_dtype, dtype=dtype,
        device=device
    )
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

    # Get vision (and memory) indices
    indices = find_vision_indices(input_tensor, v_id, separator_id)
    if m_id is not None:
        m_indices = find_vision_indices(input_tensor, m_id, separator_id=None)
    else:
        m_indices = None
    if n_id is not None:
        n_indices = find_vision_indices(input_tensor, n_id, separator_id=None)
    else:
        n_indices = None

    # Get narration indices
    narration_indices = None
    if last_k_narration is not None:
        narration_indices = find_narration_indices(input_tensor, start_id=72803, end_id=128009)  # TODO, this may only work for Llama

    for b in range(len(indices)):
        cur_indices = indices[b]
        cur_m_indices = m_indices[b] if m_indices is not None else None
        cur_n_indices = n_indices[b] if n_indices is not None else None
        for i, (s, e) in enumerate(cur_indices):
            if not i:
                continue

            # mask out last vision tokens
            last_s, last_e = cur_indices[i - 1]
            causal_mask[b, :, s:, last_s: last_e + 1] = min_dtype

            # mask out second last memory tokens (we need the last memory tokens)
            m_to_be_maksed = find_second_last_before((s, e), cur_m_indices)
            if m_to_be_maksed:
                m_s, m_e = m_to_be_maksed
                causal_mask[b, :, s:, m_s: m_e + 1] = min_dtype

            # mask out second last narration memory tokens
            n_to_be_maksed = find_second_last_before((s, e), cur_n_indices)
            if n_to_be_maksed:
                n_s, n_e = n_to_be_maksed
                causal_mask[b, :, s:, n_s: n_e + 1] = min_dtype

        # We need to mask previous narrations as well
        if narration_indices is not None:
            cur_narration_indices = narration_indices[b]
            for i, (s, e) in enumerate(cur_narration_indices):
                last_i = i - last_k_narration - 1
                if last_i < 0:
                    continue
                last_s, last_e = cur_narration_indices[last_i]
                causal_mask[b, :, s:, last_s: last_e + 1] = min_dtype

    # Deal with padding mask
    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )

    return causal_mask


class LiveLlamaForCausalLM(LlamaForCausalLM, LiveMixin):
    config_class = LiveLlamaConfig
    _keys_to_ignore_on_load_missing = ['vision_encoder', 'connector', 'clustering']

    def __init__(self, config: LiveLlamaConfig):
        super().__init__(config)
        self.connector = torch.nn.Sequential(
            torch.nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True),
            GELUActivation(config.hidden_size),
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        )

        if config.enable_vision_memory:
            if config.clustering_type == 'GLA':
                self.clustering = CLAM(
                    d_model=config.vision_hidden_size, n_head=8,
                    n_clusters=config.num_m_tokens, dropout=0.1,
                    learnable_tgt=config.learnable_memory_tgt,
                )
            else:
                raise NotImplementedError

        if config.enable_narration_memory:
            if config.clustering_type == 'GLA':
                self.narration_clustering = CLAM(
                    d_model=config.hidden_size, n_head=8,
                    n_clusters=config.num_n_tokens, dropout=0.1,
                    learnable_tgt=config.learnable_memory_tgt,
                )
            else:
                raise NotImplementedError

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        frames: torch.FloatTensor = None,
        all_frames: torch.FloatTensor = None,
        mem_frame_indices: torch.FloatTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cache_position: torch.LongTensor = None,
        mask_out_previous_vision: bool = False,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames, all_frames, mem_frame_indices)

        if self.config.enable_narration_memory:
            inputs_embeds = self.update_narration_memory_embed(input_ids, inputs_embeds)

        if mask_out_previous_vision or (self.training and self.config.vision_mask):
            # In training, we generate quadratic attention mask to avoid attention on previous vision tokens
            attention_mask = create_custom_attention_mask(
                input_ids, attention_mask, v_id=self.config.v_placeholder_id,
                m_id=self.config.m_placeholder_id,
                n_id=self.config.n_placeholder_id,
                separator_id=self.config.frame_token_interval_id,
                dtype=self.dtype,
                last_k_narration=self.config.last_k_narration,
            )

        outputs = super().forward(
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            # labels
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            cache_position=cache_position,
        )

        loss = None
        if labels is not None:
            logits = outputs[0]
            v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
            weight = v_mask * self.config.stream_loss_weight + ~v_mask
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none') * weight
            loss = loss.sum() / (labels >= 0).sum()

        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs
    
        outputs.loss = loss
        return outputs

    def generate_after_embed(self, input_ids, frames, **kwargs):
        return super().generate(inputs_embeds=self.joint_embed(input_ids, frames), **kwargs)

def build_live_llama(**kwargs):
    return build_live(config_class=LiveLlamaConfig, model_class=LiveLlamaForCausalLM, **kwargs)

if __name__ == '__main__':
    from ..arguments_live import LiveOnePlusTrainingArguments
    print(LiveOnePlusTrainingArguments().to_dict())
    model, tokenizer = build_live_llama(is_training=True, **LiveOnePlusTrainingArguments().to_dict())
    print(model.config, tokenizer)
