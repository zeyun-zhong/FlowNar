import torch
import os
import json

from tqdm import tqdm
from transformers import PreTrainedTokenizer

from data.utils import DictWithTo


class Ego4DSegmentSummary(torch.utils.data.Dataset):
    evaluation_kwargs = DictWithTo(evaluator='stream_generate')

    def get_metadata(self):
        return json.load(open(os.path.join(self.root, 'features_metadata.json')))

    def __init__(self, frame_fps: int, system_prompt: str, tokenizer: PreTrainedTokenizer, **kwargs):
        super().__init__()
        self.root = kwargs.get('data_root', '')
        self.anno_root = os.path.join(self.root, 'annotations')
        self.frame_fps = frame_fps
        local_debug = kwargs.get('local_debug', False)
        self.local_debug = local_debug
        self.metadata = self.get_metadata() if not local_debug else None

        self.system_prompt = system_prompt
        self.tokenizer = tokenizer

        annos = json.load(open(os.path.join(self.anno_root, 'segment_summaries.json')))
        self.annos = []
        for video_uid, annotation_uid_summaries in tqdm(annos.items(), desc='Constructing dataset for real stream generation...'):
            if self.local_debug:
                self.metadata = {
                    video_uid: {"duration": 10000000, "path": f"{video_uid}_tmp.pt"}}

            video_path = os.path.join(self.root, "features", "full_scale_2fps_max384_1+3x3_google--siglip-large-patch16-384", self.metadata[video_uid]['path'])
            for load_range in annotation_uid_summaries:
                # Convert string to float
                start, end = load_range.split(', ')
                start, end = float(start), float(end)
                self.annos.append((video_path, int(start * frame_fps), int(end * frame_fps) + 1))

    def __getitem__(self, idx, add_generation_prompt=False, **kwargs):
        video_path, start, end = self.annos[idx]
        if 'tmp' in video_path:
            frames = torch.randn(end, 10, 1024).bfloat16()
            frames = frames[start:end]
        else:
            frames = torch.load(video_path, weights_only=True)[start:end]
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {'role': 'stream', 'num_frames': 1, 'learn': False},  # Prompts always come after the current frame.
            {"role": "user", "content": "Please concisely narrate the video in real time."}
        ]
        video_name = os.path.basename(video_path).split('.')[0]
        sample_name = f'{video_name}_{start}_{end}'

        text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=add_generation_prompt)
        return text, frames, sample_name, idx

    def __len__(self):
        return len(self.annos)


def build_ego4d_segment_summary_val(**kwargs):
    return Ego4DSegmentSummary(**kwargs)

