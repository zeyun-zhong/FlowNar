from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class LiveTrainingArguments(TrainingArguments):
    live_version: str = 'live1+'
    system_prompt: str = (
        "A multimodal AI assistant is helping users with some activities."
        " Below is their conversation, interleaved with the list of video frames received by the assistant."
    )
    train_datasets: list[str] = None
    eval_datasets: list[str] = None
    stream_loss_weight: float = 1.0
    llm_pretrained: str = 'meta-llama/Meta-Llama-3-8B-Instruct'
    vision_pretrained: str = 'google/siglip-large-patch16-384'
    lora_modules: str = "model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$"
    lora_r: int = 128
    lora_alpha: int = 256
    finetune_modules: list[str] = field(default_factory=lambda: ['connector'])
    frame_fps: int = 2 # for training. inference can be 10
    frame_token_cls: bool = None
    frame_token_pooled: list[int] = None
    frame_resolution: int = 384
    frame_token_interval: str  = None
    frame_token_interval_threshold: float = 0.0
    augmentation: bool = False
    attn_implementation: str = 'flash_attention_2'
    output_attentions: bool = False
    output_dir: str = 'outputs/debug'
    max_num_frames: int = None
    local_debug: bool = False
    data_root: str = ""
    fix_llm: bool = False
    vision_mask: bool = False
    enable_vision_memory: bool = False
    num_m_tokens: int = None
    sample_max_frames: int = 1000
    clustering_type: str = 'GLA'
    learnable_memory_tgt: bool = False
    learn_stream: bool = True
    last_k_narration: int = None
    enable_narration_memory: bool = False
    num_n_tokens: int = None
    finetune_downstream: bool = False
    decoding_threshold: float = 0.8  # threshold for skip token
    decoding_threshold_low: float = 0.5
    decoding_strategy: str = 'two_threshold'

@dataclass
class LiveOneTrainingArguments(LiveTrainingArguments):
    live_version: str = 'live1'
    frame_token_cls: bool = True
    frame_num_tokens: int = 1
    frame_token_interval: str  = ','
    embed_mark: str = '2fps_max384_1'
    max_num_frames: int = 7200 # 1h, 2fps, 7200 frames

@dataclass
class LiveOnePlusTrainingArguments(LiveTrainingArguments):
    live_version: str = 'live1+'
    frame_token_cls: bool = True
    frame_token_pooled: list[int] = field(default_factory=lambda: [3,3])
    frame_num_tokens: int = 10 # 1+3x3
    embed_mark: str = '2fps_max384_1+3x3'
    frame_token_interval: str = ','
    max_num_frames: int = 1200 # 10min, 2fps, 1200 frames

def get_args_class(live_version: str):
    if live_version == 'live1':
        return LiveOneTrainingArguments
    elif live_version == 'live1+':
        return LiveOnePlusTrainingArguments
    raise NotImplementedError
