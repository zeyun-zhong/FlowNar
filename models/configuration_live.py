
from transformers import PretrainedConfig

class LiveConfigMixin(PretrainedConfig):
    def __init__(self, *, vision_pretrained: str = None,
        frame_resolution: int = None, frame_token_cls: bool = None, frame_token_pooled: list[int] = None, frame_num_tokens: int = None,
        v_placeholder: str = '<v>', frame_token_interval: str = None, v_placeholder_id: int = None, frame_token_interval_id: int = None,
        stream_loss_weight: float = 1.0, vision_hidden_size=1024, vision_mask=False,
        m_placeholder: str = '<m>', enable_vision_memory: bool = False, m_placeholder_id: int = None, num_m_tokens: int = None,
        clustering_type: str = 'GLA', learnable_memory_tgt: bool = False, last_k_narration=None,
        enable_narration_memory: bool = False, n_placeholder_id: int = None, num_n_tokens: int = None, n_placeholder: str = '<n>',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_pretrained = vision_pretrained
        self.frame_resolution = frame_resolution
        self.frame_token_cls = frame_token_cls
        self.frame_token_pooled = frame_token_pooled
        self.frame_num_tokens = frame_num_tokens
        self.vision_hidden_size = vision_hidden_size
        self.stream_loss_weight = stream_loss_weight
        self.v_placeholder = v_placeholder
        self.frame_token_interval = frame_token_interval
        self.v_placeholder_id = v_placeholder_id
        self.frame_token_interval_id = frame_token_interval_id
        self.vision_mask = vision_mask
        self.m_placeholder = m_placeholder
        self.enable_vision_memory = enable_vision_memory
        self.m_placeholder_id = m_placeholder_id
        self.num_m_tokens = num_m_tokens
        self.clustering_type = clustering_type
        self.learnable_memory_tgt = learnable_memory_tgt
        self.last_k_narration = last_k_narration
        self.enable_narration_memory = enable_narration_memory
        self.n_placeholder_id = n_placeholder_id
        self.num_n_tokens = num_n_tokens
        self.n_placeholder = n_placeholder
