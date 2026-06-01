import torch
from functools import partial
from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother


def data_collator(batch: list[list], *, tokenizer: PreTrainedTokenizer, **kwargs):
    batch = list(zip(*batch))
    batch_text, batch_frames, batch_all_frames, batch_mem_frame_indices, batch_learn_ranges, batch_sample_idx, batch_evaluation_kwargs = batch
    batch = tokenizer(batch_text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", padding=True)
    batch_labels = torch.full_like(batch.input_ids, LabelSmoother.ignore_index, dtype=torch.long)
    for text, labels, input_ids, offset_mapping, learn_range in zip(
        batch_text, batch_labels, batch.input_ids, batch.offset_mapping, batch_learn_ranges
    ):
        for learn_r in learn_range:
            start = torch.nonzero(offset_mapping[:,0] == learn_r.start).item()
            if offset_mapping[:,0][-1] >= learn_r.stop:
                stop = torch.nonzero(offset_mapping[:,0] == learn_r.stop).item()
            else: # the last eos token
                stop = len(input_ids)
            labels[start-1:stop-1] = input_ids[start:stop]
            # NOTE: input_ids may out of boundary of len(tokenizer) - 1. (1 is the added vision placeholder)
            # this is because some frames has v_placeholder_id target. so replace it with eos token.
            labels[labels >= len(tokenizer) - 1] = tokenizer.eos_token_id
    batch['labels'] = batch_labels
    batch.pop('offset_mapping')
    batch['frames'] = torch.cat(batch_frames)
    batch['all_frames'] = torch.cat(batch_all_frames)
    batch['mem_frame_indices'] = torch.cat(batch_mem_frame_indices)
    batch['sample_idxs'] = torch.tensor(batch_sample_idx)
    if batch_evaluation_kwargs[0]:
        batch['evaluation_kwargs'] = batch_evaluation_kwargs[0] # evaluation only supports bs = 1, so its okay

    return batch


def stream_generate_data_collator(batch: list[list], *, tokenizer: PreTrainedTokenizer, **kwargs):
    batch = list(zip(*batch))
    batch_text, batch_frames, batch_sample_names, batch_sample_idx = batch
    batch = tokenizer(batch_text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", padding=True)
    batch['sample_idxs'] = torch.tensor(batch_sample_idx)
    batch['frames'] = torch.cat(batch_frames)
    # batch['sample_names'] = batch_sample_names
    return batch


def get_data_collator(stream_generate=False, **kwargs):
    if stream_generate:
        return partial(stream_generate_data_collator, **kwargs)
    return partial(data_collator, **kwargs)
