"""
Unified Configuration for Event Camera Semantic Segmentation
Optimized for memory efficiency with aggressive caching enabled by default
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import yaml


@dataclass
class DataConfig:
    """Data configuration with memory optimization"""
    data_root: str = "./data"
    train_path: str = "train"
    val_path: str = "val"
    test_path: str = "test"

    # Fixed spatial dimensions to prevent zero shape error
    voxel_size: List[float] = field(default_factory=lambda: [2.0, 2.0, 2.0])
    # spatial_size: List[int] = field(default_factory=lambda: [352, 288, 256])  # Fixed reasonable size
    spatial_size: List[int] = field(default_factory=lambda: [[88, 72, 64]])  # Fixed reasonable size
    time_window: float = 0.1

    # Event processing (reduced for memory efficiency)
    max_events_num: int = 30000  # Reduced from 50000
    min_events_per_sample: int = 1000
    input_channels: int = 4  # (x, y, t, p)

    # Data augmentation
    augment: bool = True
    flip_prob: float = 0.5
    scale_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    rotation_range: float = 0.2

    # Dataloader settings (optimized for memory)
    batch_size: int = 1  # Very small batch size to minimize memory
    num_workers: int = 0  # Single thread to avoid memory overhead
    shuffle_train: bool = True
    pin_memory: bool = False  # Disabled to save memory
    drop_last: bool = True

    # Cache settings (ENABLED BY DEFAULT)
    use_cache: bool = True  # Always use cache
    cache_dir: str = "voxel_cache"  # Cache directory name


@dataclass
class ModelConfig:
    """Model configuration"""
    width: int = 4
    num_classes: int = 1  # Binary segmentation

    # Mamba configuration
    use_mamba_encoder: bool = True
    use_mamba_decoder: bool = False
    use_mamba_attention: bool = False

    encoder_guidance_type: str = "spatial"
    decoder_guidance_type: str = "learned"
    attention_guidance_type: str = "temporal"

    mamba_d_state: int = 8
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_radius: float = 1.0

    pretrained_path: Optional[str] = None


@dataclass
class TrainConfig:
    """Training configuration with memory optimizations"""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Optimizer
    optimizer: str = "adam"
    momentum: float = 0.9
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])

    # Scheduler
    scheduler: str = "step"
    step_size: int = 10
    gamma: float = 0.1
    warmup_epochs: int = 5

    # Loss
    loss_type: str = "stc"
    k: int = 10  # STCLoss parameter
    t: float = 0.07  # STCLoss temperature

    # Training strategy (memory optimized)
    gradient_clip: float = 1.0
    mixed_precision: bool = False  # Disabled to avoid potential memory issues
    gradient_accumulation_steps: int = 4  # Increased for smaller batches

    # Checkpointing
    save_interval: int = 5
    save_best: bool = True
    model_save_root: str = "./checkpoints"


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    eval_interval: int = 1
    metrics: List[str] = field(default_factory=lambda: ["miou", "accuracy", "precision", "recall"])

    # ROC evaluation
    roc: bool = True
    pd_detT: float = 0.01
    correct_thresh: float = 0.5

    # Thresholds
    confidence_thresholds: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    default_threshold: float = 0.5


@dataclass
class MemoryConfig:
    """Memory optimization configuration"""
    use_cache: bool = True  # Always enabled
    gradient_accumulation_steps: int = 4  # Increased for memory efficiency
    max_memory_gb: int = 6  # Conservative memory limit
    clear_cache_frequency: int = 3  # More frequent cache clearing

    # Additional memory optimizations
    enable_memory_monitoring: bool = True
    aggressive_gc: bool = True  # Enable aggressive garbage collection
    cache_compression: bool = True  # Enable cache compression


@dataclass
class Config:
    """Main configuration with memory optimizations"""
    # Experiment
    exp_name: str = "event_seg_mamba_optimized"
    exp_dir: str = "./experiments"
    seed: int = 37

    # Device
    device: str = "cuda:0"
    gpu_ids: List[int] = field(default_factory=lambda: [0])

    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Logging
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "event_segmentation"

    # Additional paths (for compatibility)
    root: str = "./data"
    whole_t: float = 0.1
    res: List[int] = field(default_factory=lambda: [346, 260])

    def __post_init__(self):
        """Create directories and setup paths"""
        self.exp_path = os.path.join(self.exp_dir, self.exp_name)
        self.checkpoint_dir = os.path.join(self.exp_path, "checkpoints")
        self.log_dir = os.path.join(self.exp_path, "logs")
        self.result_dir = os.path.join(self.exp_path, "results")

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.train.model_save_root, exist_ok=True)

        # Ensure cache is always enabled
        self.data.use_cache = True
        self.memory.use_cache = True

        # Sync memory settings with training
        self.train.gradient_accumulation_steps = self.memory.gradient_accumulation_steps

    def save(self, path: str):
        """Save configuration to yaml file"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str):
        """Load configuration from yaml file"""
        with open(path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**config_dict)

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary"""
        return {
            'input_channels': self.data.input_channels,
            'width': self.model.width,
            'num_classes': self.model.num_classes,
            'mamba_config': {
                'use_mamba_encoder': self.model.use_mamba_encoder,
                'use_mamba_decoder': self.model.use_mamba_decoder,
                'use_mamba_attention': self.model.use_mamba_attention,
                'encoder_config': {
                    'guidance_type': self.model.encoder_guidance_type,
                    'd_state': self.model.mamba_d_state,
                    'd_conv': self.model.mamba_d_conv,
                    'expand': self.model.mamba_expand,
                    'radius': self.model.mamba_radius,
                },
                'decoder_config': {
                    'guidance_type': self.model.decoder_guidance_type,
                    'd_state': self.model.mamba_d_state,
                    'd_conv': self.model.mamba_d_conv,
                    'expand': self.model.mamba_expand,
                    'radius': self.model.mamba_radius,
                },
                'attention_config': {
                    'guidance_type': self.model.attention_guidance_type,
                    'd_state': self.model.mamba_d_state,
                    'd_conv': self.model.mamba_d_conv,
                    'expand': self.model.mamba_expand,
                    'radius': self.model.mamba_radius,
                },
            }
        }

    def create_cache_config(self):
        """Create cache precomputation configuration"""
        return {
            'data_root': self.data.data_root,
            'cache_dir': self.data.cache_dir,
            'max_events_num': self.data.max_events_num,
            'spatial_shape': self.data.spatial_size,
            'compression': self.memory.cache_compression,
            'batch_size': 3,  # Small batch for cache computation
        }


def create_config(config_path: Optional[str] = None, **kwargs) -> Config:
    """Create configuration from file or kwargs"""
    if config_path and os.path.exists(config_path):
        config = Config.load(config_path)
    else:
        config = Config()

    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def create_memory_optimized_config() -> Config:
    """Create a memory-optimized configuration"""
    config = Config()

    # Ultra conservative memory settings
    config.data.batch_size = 1
    config.data.max_events_num = 20000
    config.data.num_workers = 0
    config.data.pin_memory = False

    # Training optimizations
    config.train.gradient_accumulation_steps = 8
    config.train.mixed_precision = False
    config.memory.gradient_accumulation_steps = 8
    config.memory.max_memory_gb = 4
    config.memory.clear_cache_frequency = 2

    return config