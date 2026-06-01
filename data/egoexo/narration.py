import os, torch, json, tqdm, collections, random
from transformers import EvalPrediction

from data.stream import StreamMixIn
from data.utils import ceil_time_by_fps, DictWithTo

class EgoExo4DNarrationStream(StreamMixIn):
    instructions = [{"role": "user", "content": "Please concisely narrate the video in real time. Use the tag 'C' to denote the camera wearer, and other letter tags, such as 'X', to denote other individuals in the scene."}]
    evaluation_kwargs = DictWithTo(evaluator='stream_evaluate')

    def __len__(self):
        return len(self.annos)

    def get_metadata(self):
        return json.load(open(os.path.join(self.root, 'features_metadata.json')))

    def get_annos(self, split: str) -> dict[str, dict[str, list]]:
        assert split in ['train', 'val']
        anno_path = os.path.join(self.anno_root, f'refined_egocentric_atomic_descriptions_gpt_{split}.json')
        narration_streams = json.load(open(anno_path))
        return narration_streams

    def __init__(self, *, split: str, frame_fps: int, is_training: bool, augmentation: bool, local_debug: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, augmentation=augmentation, is_training=is_training, local_debug=local_debug, **kwargs)
        self.root = kwargs.get('data_root', '')
        self.anno_root = os.path.join(self.root, 'annotations')
        self.frame_fps = frame_fps
        self.local_debug = local_debug
        self.metadata = self.get_metadata() if not local_debug else None

        self.is_training = is_training
        enable_vision_memory = kwargs.get('enable_vision_memory', False)
        enable_narration_memory = kwargs.get('enable_narration_memory', False)
        learn_stream = kwargs.get('learn_stream', True)

        annos = self.get_annos(split)
        self.annos = []
        for video_uid, _annotation_uid_narrations in tqdm.tqdm(annos.items(), desc=f'narration_stream_{split}...'):

            if self.local_debug:
                self.metadata = {video_uid: {"duration": 10000000, "path": "tmp"}}  # temporary solution added by X

            duration = self.metadata[video_uid]['duration']  # in seconds
            for annotation in _annotation_uid_narrations:
                narrations = annotation['descriptions']
                if not narrations:
                    continue
                start_time = ceil_time_by_fps(narrations[0]['timestamp'], frame_fps, min_time=0, max_time=duration)
                conversation = []
                last_time = start_time - 1 / frame_fps
                last_text = None
                num_total_frames = 0
                for narration in narrations:
                    if not narration['ego_visible']:
                        continue
                    if last_time >= duration:
                        break
                    text = narration['text']
                    if text == last_text:
                        continue
                    time = ceil_time_by_fps(narration['timestamp'], frame_fps, min_time=0, max_time=duration)
                    if time == last_time: # since we have sorted and ceiled, so directly replace, this time is more close
                        conversation[-1]['content'] = text
                    else: # time > last_time
                        num_frames = int((time - last_time) * frame_fps)
                        num_total_frames += num_frames

                        conversation.extend([
                            {"role": "stream", 'num_frames': num_frames, 'learn': learn_stream},
                            {"role": "assistant", "content": text, 'learn': True},
                        ])

                        if enable_narration_memory:
                            conversation.append({"role": "narration_memory"})

                        if enable_vision_memory:
                            conversation.append({"role": "memory"})

                    last_time = time
                    last_text = text
                if not conversation:
                    continue
                self.add_to_annos(video_uid, conversation, start_time, last_time, frame_fps)

    def add_to_annos(self, video_uid, conversation, start_time, end_time, frame_fps):
        while conversation[-1]["role"] in ["memory", "narration_memory"]:
            conversation.pop()
        video_path = os.path.join(self.root, "features", "egocentric_2fps_max384_1+3x3_google--siglip-large-patch16-384", self.metadata[video_uid]['path'])
        self.annos.append({
            'conversation': conversation,
            'load_ranges': {video_path: range(
                int(start_time * frame_fps), int(end_time * frame_fps) + 1)}
        })

    def preprocess_conversation(self, conversation):
        assert conversation[0]['role'] == 'stream' and conversation[0]['num_frames'] == 1
        conversation[0]['learn'] = False
        return conversation[:1] + [random.choice(self.instructions)] + conversation[1:] # first is stream

    def __getitem__(self, index):
        anno = self.annos[index]
        return *super().__getitem__(
            conversation=self.preprocess_conversation(anno['conversation']),
            load_ranges=anno['load_ranges'],
        ), index, self.evaluation_kwargs


    def compute_metrics(self, eval_predictions: EvalPrediction, *args, **kwargs):
        lm_ppl, frame_diff, fluency, lm_correctness = torch.from_numpy(eval_predictions.predictions).mean(dim=0).tolist()
        return {
            f'lm_ppl': lm_ppl,
            f'time_diff': frame_diff / self.frame_fps,
            f'fluency': fluency,
            f'lm_correctness': lm_correctness
        }

def build_egoexo4d_narration_stream_train(**kwargs):
    return EgoExo4DNarrationStream(split='train', **kwargs)

def build_egoexo4d_narration_stream_val(**kwargs):
    return EgoExo4DNarrationStream(split='val', **kwargs)

class EgoExo4DRefinedNarrationStream(EgoExo4DNarrationStream):
    instructions = [
        {"role": "user", "content": "Please concisely narrate the video in real time."},
        {"role": "user", "content": "Help me to illustrate my view in short."},
        {"role": "user", "content": "Please simply describe what do you see."},
        {"role": "user", "content": "Continuously answer what you observed with simple text."},
        {"role": "user", "content": "Do concise real-time narration."},
        {"role": "user", "content": "Hey assistant, do you know the current video content? Reply me concisely."},
        {"role": "user", "content": "Simply interpret the scene for me."},
        {"role": "user", "content": "What can you tell me about? Be concise."},
        {"role": "user", "content": "Use simple text to explain what is shown in front of me."},
        {"role": "user", "content": "What is the action now? Please response in short."},
    ]

def build_egoexo4d_refined_narration_stream_train(**kwargs):
    return EgoExo4DRefinedNarrationStream(split='train', **kwargs)

def build_egoexo4d_refined_narration_stream_val(**kwargs):
    return EgoExo4DRefinedNarrationStream(split='val', **kwargs)

