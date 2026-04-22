from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    hf_dataset: str = "huggan/anime-faces"
    hf_split: str = "train"
    cache_dir: Path = Path("cache")
    force_rebuild_cache: bool = False
    max_samples: int = 20000
    output_dir: Path = Path("outputs")
    image_size: int = 64
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    latent_dim: int = 32
    beta: float = 4.0
    recon_loss: str = "l1"
    kl_warmup_epochs: int = 0
    num_workers: int = 2
    seed: int = 42
    device: str = "cuda"


@dataclass
class EvalConfig:
    checkpoint: Path
    hf_dataset: str = "huggan/anime-faces"
    hf_split: str = "train"
    cache_dir: Path = Path("cache")
    force_rebuild_cache: bool = False
    max_samples: int = 20000
    output_dir: Path = Path("outputs/eval")
    image_size: int = 64
    batch_size: int = 16
    latent_dim: int = 32
    recon_loss: str = "l1"
    interpolation_steps: int = 8
    num_samples: int = 16
    num_workers: int = 2
    device: str = "cuda"
