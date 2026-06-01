import json
import os
from typing import Dict, Union, Any, Optional, List, Tuple
import torch
from torch import nn
from transformers import Trainer

from models import fast_greedy_generate, find_vision_indices, LiveTrainingArguments
from models.live_llama.modeling_live_llama import LiveLlamaForCausalLM


class StreamGenerator:
    def __init__(
            self,
            model: LiveLlamaForCausalLM,
            tokenizer,
            args: LiveTrainingArguments,
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

        # visual
        self.frame_num_tokens = self.model.config.frame_num_tokens
        self.frame_v_placeholder = self.model.config.v_placeholder * self.frame_num_tokens
        self.frame_token_interval_id = self.model.config.frame_token_interval_id
        self.frame_placeholder_ids = torch.tensor(
            self.model.config.v_placeholder_id).repeat(
            self.model.config.frame_num_tokens).reshape(1, -1)

        # generation
        self.system_prompt = args.system_prompt
        self.eos_token_id = self.model.config.eos_token_id

        self.decoding_strategy = args.decoding_strategy
        self.skip_thresh_high = args.decoding_threshold  # high -> more narrations
        self.skip_thresh_low = args.decoding_threshold_low  # low -> fewer narrations
        self.avg_duration = 8  # 3.831 +- 7.951s  2fps -> 7.662

    def reset(self, device):
        self.past_key_values = None
        self.last_ids = None
        self.narration_times = []
        self.output_ids = []
        self.inplace_output_ids = torch.zeros(1, 100, device=device, dtype=torch.long)
        self._added_stream_prompt_ids = self.tokenizer.apply_chat_template(
            [{}], add_stream_prompt=True, return_tensors='pt').to(device)  # '\n['
        self._added_generation_ids = self.tokenizer.apply_chat_template(
            [{}], add_generation_prompt=True, return_tensors='pt').to(device)  # '\nAssistant:'
        self._added_stream_generation_ids = self.tokenizer.apply_chat_template(
            [{}], add_stream_generation_prompt=True, return_tensors='pt').to(device)  # ']\nAssistant:'

        # Position ids
        self.last_pos = None
        self.pos_v_start = None
        self.pos_v_end = None

        # Narration indices, used to trim cache
        self.narration_indices = []

    def generate_narration(self, is_begin=False):
        self.last_ids = self._added_generation_ids if is_begin else self._added_stream_generation_ids
        inputs_embeds = self.model.get_input_embeddings()(self.last_ids)

        if self.last_pos is not None:
            nar_start = self.last_pos + 2 if is_begin else self.last_pos_trimmed + 2  # Assistant

        output_ids, self.past_key_values, self.last_pos = fast_greedy_generate(
            model=self.model, inputs_embeds=inputs_embeds,
            past_key_values=self.past_key_values,
            eos_token_id=self.eos_token_id,
            inplace_output_ids=self.inplace_output_ids,
            last_pos=self.last_pos,
        )
        self.last_ids = output_ids[:, -1:]
        self.output_ids.append(output_ids.clone())

        if self.last_pos is not None:
            nar_end = self.past_key_values[0][0].shape[-2]  # we donot -1, as we want to include eot_id as well.
            self.narration_indices.append((nar_start, nar_end))

    def trim_past_key_values(self, past_key_values, start, stop, device):
        # Element corresponding to stop will be removed !!!
        # Use mask to trim corresponding elements in past_key_values
        len_cache = past_key_values[0][0].shape[-2]
        select = torch.ones(len_cache, dtype=torch.bool, device=device)
        select[start:stop + 1] = False
        return [[past_keys[:,:,select], past_values[:,:,select]] for past_keys, past_values in past_key_values]

    def update_narration_indices_after_trim(self, trimmed_len: int):
        # Trim only affects the last (most current) narration
        start, end = self.narration_indices.pop()
        start, end = start - trimmed_len, end - trimmed_len
        self.narration_indices.append((start, end))

    def get_threshold(self, frame_idx):
        if self.decoding_strategy == "single_threshold":
            thresh = self.skip_thresh_high
        elif self.decoding_strategy == "two_threshold":
            if frame_idx - self.narration_times[-1] > self.avg_duration:
                thresh = self.skip_thresh_high  # encourage decoding
            else:
                thresh = self.skip_thresh_low
        else:
            raise NotImplementedError("self.decoding_strategy should be either single_threshold or two_threshold")
        return thresh

    def __call__(self, input_ids, frame_embeds, *args, **kwargs):
        if self.args.enable_vision_memory:
            assert self.args.vision_mask
            if self.args.last_k_narration is not None:
                cls = self._call_vision_memory_last_k_narration
            else:
                cls = self._call_vision_memory
        else:
            cls = self._call_base

        return cls(input_ids, frame_embeds, *args, **kwargs)

    def _call_base(self, input_ids, frame_embeds, *args, **kwargs):
        # Some preparation
        self.reset(input_ids.device)

        # 1. Replace <v> with the real frame (This applies only for the first frame.)
        inputs_embeds = self.model.joint_embed(input_ids, frame_embeds[0])
        D = inputs_embeds.shape[-1]
        frame_embeds = self.model.connector(frame_embeds)

        # 2. We always generate a narration at the beginning, as done in training.
        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=self.past_key_values)
        self.past_key_values = outputs.past_key_values
        self.generate_narration(is_begin=True)
        self.narration_times.append(0)

        # 3. Make frame-wise prediction
        for i, frame_embed in enumerate(frame_embeds[1:], start=1):

            if self.last_ids == self.eos_token_id:
                self.last_ids = torch.cat([self.last_ids, self._added_stream_prompt_ids], dim=1)

            inputs_embeds = torch.cat([
                self.model.get_input_embeddings()(self.last_ids).view(1, -1, D),
                frame_embed.view(1, -1, D),
            ], dim=1)

            outputs = self.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=self.past_key_values)
            self.past_key_values = outputs.past_key_values

            # 3.1. We check if we should skip the current frame,
            # if the "skip" probability is smaller than the predefined threshold
            # then the argmax id should be ]\n, meaning we should generate a narration.
            next_score = outputs.logits[:, -1:].softmax(dim=-1)

            thresh = self.get_threshold(frame_idx=i)

            if next_score[:, :, self.frame_token_interval_id] < thresh:
                next_score[:, :, self.frame_token_interval_id].zero_()
            self.last_ids = next_score.argmax(dim=-1)
            if (self.last_ids != self.frame_token_interval_id) or (i == len(frame_embeds) - 1):
                # We generate narrations either if the argmax id is ]\n or if the frame is the last frame
                if i != len(frame_embeds) - 1:
                    assert self.last_ids == 933, f'{self.last_ids} != 933'  # HACK, 933 = ]\n
                self.generate_narration()
                self.narration_times.append(i)

        return self.narration_times, self.output_ids

    def _call_vision_memory(self, input_ids, frames, *args, **kwargs):
        # Some preparation
        device = input_ids.device
        self.reset(device)
        rank = kwargs.get('rank')
        end_stream_id = torch.tensor([[933]]).to(input_ids)  # ]\n, TODO
        left_square_bracket_id = torch.tensor([[58]]).to(input_ids)  # [ TODO

        # 0. Only compute vision and memory mask if required
        self.pos_v_start, self.pos_v_end = find_vision_indices(
            input_ids, v_id=self.model.config.v_placeholder_id,
            separator_id=self.model.config.frame_token_interval_id
        )[0][0]

        # 1. Replace <v> with the real frame (This applies only for the first frame.)
        inputs_embeds = self.model.joint_embed(input_ids, frames[0])
        D = inputs_embeds.shape[-1]
        frame_embeds = self.model.connector(frames)
        # 1.1 Get memory embeds
        mem_frame_indices = torch.arange(len(frames), device=device, dtype=torch.int)
        memory_embeds = self.model.memory_embed(frames, mem_frame_indices).view(len(frames), -1, D)

        # 2. We always generate a narration at the beginning, as done in training.
        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=self.past_key_values)
        self.past_key_values = outputs.past_key_values
        self.last_pos = self.past_key_values[0][0].shape[-2] - 1

        self.generate_narration(is_begin=True)
        self.past_key_values = self.trim_past_key_values(self.past_key_values, self.pos_v_start, self.pos_v_end, device)
        self.last_pos_trimmed = self.past_key_values[0][0].shape[-2] - 1

        self.narration_times.append(0)

        # 3. Make frame-wise prediction
        for i, frame_embed in enumerate(frame_embeds[1:], start=1):

            if self.last_ids == self.eos_token_id:
                self.last_ids = torch.cat([self.last_ids, self._added_stream_prompt_ids], dim=1)
                self.pos_v_start = self.last_pos_trimmed + self.last_ids.size(1)  # [

                # <eos> \n[ mem ]\n [ frame
                inputs_embeds = torch.cat([
                    self.model.get_input_embeddings()(self.last_ids).view(1, -1, D),
                    memory_embeds[i-1].view(1, -1, D),
                    self.model.get_input_embeddings()(end_stream_id).view(1, -1, D),
                    self.model.get_input_embeddings()(left_square_bracket_id).view(1, -1, D),
                    frame_embed.view(1, -1, D),
                ], dim=1)

            else:
                inputs_embeds = torch.cat([
                    self.model.get_input_embeddings()(self.last_ids).view(1, -1, D),
                    frame_embed.view(1, -1, D),
                ], dim=1)

            position_ids = self.last_pos + 1 + torch.arange(inputs_embeds.size(1), device=inputs_embeds.device, dtype=torch.int)
            self.last_pos += inputs_embeds.size(1)  # Update last pos
            self.last_pos_trimmed += inputs_embeds.size(1)  # Update last pos
            outputs = self.model(inputs_embeds=inputs_embeds, position_ids=position_ids[None], use_cache=True, past_key_values=self.past_key_values)
            self.past_key_values = outputs.past_key_values

            # 3.1. We check if we should skip the current frame,
            # if the "skip" probability is smaller than the predefined threshold
            # then the argmax id should be ]\n, meaning we should generate a narration.
            next_score = outputs.logits[:, -1:].softmax(dim=-1)

            thresh = self.get_threshold(frame_idx=i)

            if next_score[:, :, self.frame_token_interval_id] < thresh:
                next_score[:, :, self.frame_token_interval_id].zero_()
            self.last_ids = next_score.argmax(dim=-1)
            if (self.last_ids != self.frame_token_interval_id) or (i == len(frame_embeds) - 1):
                # We generate narrations either if the argmax id is ]\n or if the frame is the last frame
                if i != len(frame_embeds) - 1:
                    assert self.last_ids == 933, f'{self.last_ids} != 933'  # HACK, 933 = ]\n
                self.pos_v_end = self.last_pos_trimmed + 1  # ]\n
                self.generate_narration()
                self.past_key_values = self.trim_past_key_values(self.past_key_values, self.pos_v_start, self.pos_v_end, device)
                self.last_pos_trimmed = self.past_key_values[0][0].shape[-2] - 1
                self.narration_times.append(i)

        return self.narration_times, self.output_ids

    def _call_vision_memory_last_k_narration(self, input_ids, frames, *args, **kwargs):
        # Some preparation
        device = input_ids.device
        self.reset(device)
        rank = kwargs.get('rank')
        end_stream_id = torch.tensor([[933]]).to(input_ids)  # ]\n TODO
        left_square_bracket_id = torch.tensor([[58]]).to(input_ids)  # [  TODO
        k = self.args.last_k_narration

        # 0. Only compute vision and memory mask if required
        self.pos_v_start, self.pos_v_end = find_vision_indices(
            input_ids, v_id=self.model.config.v_placeholder_id,
            separator_id=self.model.config.frame_token_interval_id
        )[0][0]

        # 1. Replace <v> with the real frame (This applies only for the first frame.)
        inputs_embeds = self.model.joint_embed(input_ids, frames[0])
        D = inputs_embeds.shape[-1]
        frame_embeds = self.model.connector(frames)
        # 1.1 Get memory embeds
        mem_frame_indices = torch.arange(len(frames), device=device, dtype=torch.int)
        memory_embeds = self.model.memory_embed(frames, mem_frame_indices).view(len(frames), -1, D)

        # 2. We always generate a narration at the beginning, as done in training.
        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=self.past_key_values)
        self.past_key_values = outputs.past_key_values
        self.last_pos = self.past_key_values[0][0].shape[-2] - 1

        self.generate_narration(is_begin=True)
        self.past_key_values = self.trim_past_key_values(self.past_key_values, self.pos_v_start, self.pos_v_end, device)
        self.last_pos_trimmed = self.past_key_values[0][0].shape[-2] - 1
        # update narration indices after cache trim
        self.update_narration_indices_after_trim(trimmed_len=self.pos_v_end - self.pos_v_start + 1)

        self.narration_times.append(0)

        # 3. Make frame-wise prediction
        for i, frame_embed in enumerate(frame_embeds[1:], start=1):

            if self.last_ids == self.eos_token_id:
                self.last_ids = torch.cat([self.last_ids, self._added_stream_prompt_ids], dim=1)
                self.pos_v_start = self.last_pos_trimmed + self.last_ids.size(1)  # [

                # <eos> \n[ mem ]\n [ frame
                inputs_embeds = torch.cat([
                    self.model.get_input_embeddings()(self.last_ids).view(1, -1, D),
                    memory_embeds[i-1].view(1, -1, D),
                    self.model.get_input_embeddings()(end_stream_id).view(1, -1, D),
                    self.model.get_input_embeddings()(left_square_bracket_id).view(1, -1, D),
                    frame_embed.view(1, -1, D),
                ], dim=1)

            else:
                inputs_embeds = torch.cat([
                    self.model.get_input_embeddings()(self.last_ids).view(1, -1, D),
                    frame_embed.view(1, -1, D),
                ], dim=1)

            position_ids = self.last_pos + 1 + torch.arange(inputs_embeds.size(1), device=inputs_embeds.device, dtype=torch.int)
            self.last_pos += inputs_embeds.size(1)  # Update last pos
            self.last_pos_trimmed += inputs_embeds.size(1)  # Update last pos
            outputs = self.model(inputs_embeds=inputs_embeds, position_ids=position_ids[None], use_cache=True, past_key_values=self.past_key_values)
            self.past_key_values = outputs.past_key_values

            # 3.1. We check if we should skip the current frame,
            # if the "skip" probability is smaller than the predefined threshold
            # then the argmax id should be ]\n, meaning we should generate a narration.
            next_score = outputs.logits[:, -1:].softmax(dim=-1)

            thresh = self.get_threshold(frame_idx=i)

            if next_score[:, :, self.frame_token_interval_id] < thresh:
                next_score[:, :, self.frame_token_interval_id].zero_()
            self.last_ids = next_score.argmax(dim=-1)
            if (self.last_ids != self.frame_token_interval_id) or (i == len(frame_embeds) - 1):
                # We generate narrations either if the argmax id is ]\n or if the frame is the last frame
                if i != len(frame_embeds) - 1:
                    assert self.last_ids == 933, f'{self.last_ids} != 933'  # HACK, 933 = ]\n
                self.pos_v_end = self.last_pos_trimmed + 1  # ]\n
                self.generate_narration()
                self.past_key_values = self.trim_past_key_values(self.past_key_values, self.pos_v_start, self.pos_v_end, device)
                # update narration indices after cache trim
                self.update_narration_indices_after_trim(trimmed_len=self.pos_v_end - self.pos_v_start + 1)

                # We need to trim narrations if the number of all past narrations exceeds k
                if len(self.narration_indices) > k:
                    nar_start, nar_end = self.narration_indices.pop(0)
                    self.past_key_values = self.trim_past_key_values(self.past_key_values, nar_start, nar_end, device)
                    trimmed_len = nar_end - nar_start + 1
                    # Update narration indices
                    self.narration_indices = [(start - trimmed_len, end - trimmed_len) for start, end in self.narration_indices]

                self.last_pos_trimmed = self.past_key_values[0][0].shape[-2] - 1
                self.narration_times.append(i)

        return self.narration_times, self.output_ids


class TrainerStreamGenerator(Trainer):
    def __init__(self, stream_generator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_generator = stream_generator

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad(), self.compute_loss_context_manager():
            inputs = self._prepare_inputs(inputs)
            sample_idxs = inputs.pop('sample_idxs')

            path = os.path.join(self.args.output_dir, f'narrations_{sample_idxs[0].item():04d}.json')
            if os.path.exists(path):
                print(f'Skipping {path}...')
                return None, None, None

            narration_times, output_ids = self.stream_generator(inputs['input_ids'], inputs['frames'], rank=self.args.process_index)

        # Save to disk
        decoded_text = [self.tokenizer.decode(out[0]) for out in output_ids]
        narrations = {t: text for t, text in zip(narration_times, decoded_text)}
        with open(path, 'w') as file:
            json.dump(narrations, file, indent=4)

        output_ids = torch.cat(output_ids, dim=1)
        return (None, output_ids.reshape(1, -1), sample_idxs)