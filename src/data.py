from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class HFImageTransformDataset(Dataset):
    def __init__(self, dataset, image_size: int) -> None:
        self.dataset = dataset
        self.transform = build_transforms(image_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        image = item["image"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        return self.transform(image)


class ShardedTensorDataset(Dataset):
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        with (cache_path / "metadata.json").open("r", encoding="utf-8") as fp:
            self.metadata = json.load(fp)

        self.shards = self.metadata["shards"]
        self.cumulative_sizes: list[int] = []
        total = 0
        for shard in self.shards:
            total += shard["count"]
            self.cumulative_sizes.append(total)

        self._loaded_shard_index: int | None = None
        self._loaded_images: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.metadata["num_images"]

    def _load_shard(self, shard_index: int) -> torch.Tensor:
        shard_path = self.cache_path / self.shards[shard_index]["file"]
        if self._loaded_shard_index != shard_index:
            self._loaded_images = torch.load(shard_path, map_location="cpu", weights_only=False)
            self._loaded_shard_index = shard_index
        return self._loaded_images

    def __getitem__(self, index: int):
        shard_index = bisect_right(self.cumulative_sizes, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_sizes[shard_index - 1]
        local_index = index - shard_start
        images = self._load_shard(shard_index)
        image = images[local_index].float().div(255.0)
        return image, 0


def build_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def _cache_dir_path(
    cache_dir: Path,
    dataset_name: str,
    split: str,
    image_size: int,
    max_samples: int | None,
) -> Path:
    dataset_key = dataset_name.replace("/", "__")
    sample_suffix = "all" if max_samples is None else str(max_samples)
    return cache_dir / f"{dataset_key}_{split}_{image_size}_{sample_suffix}"


def ensure_hf_tensor_cache(
    dataset_name: str,
    split: str,
    image_size: int,
    cache_dir: Path,
    max_samples: int | None = 20000,
    force_rebuild: bool = False,
    preprocess_batch_size: int = 512,
    preprocess_workers: int = 4,
) -> Path:
    cache_path = _cache_dir_path(cache_dir, dataset_name, split, image_size, max_samples)
    metadata_path = cache_path / "metadata.json"
    if metadata_path.exists() and not force_rebuild:
        return cache_path

    cache_path.mkdir(parents=True, exist_ok=True)
    for shard_file in cache_path.glob("shard_*.pt"):
        shard_file.unlink()
    if metadata_path.exists():
        metadata_path.unlink()

    dataset = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    preprocess_dataset = HFImageTransformDataset(dataset=dataset, image_size=image_size)
    preprocess_loader = DataLoader(
        preprocess_dataset,
        batch_size=preprocess_batch_size,
        shuffle=False,
        num_workers=preprocess_workers,
        pin_memory=False,
    )

    shards = []
    total_images = 0
    for shard_index, batch in enumerate(
        tqdm(preprocess_loader, desc=f"Caching {dataset_name} [{split}] at {image_size}x{image_size}")
    ):
        images_uint8 = batch.mul(255).to(torch.uint8)
        shard_name = f"shard_{shard_index:05d}.pt"
        torch.save(images_uint8, cache_path / shard_name)
        batch_count = int(images_uint8.shape[0])
        total_images += batch_count
        shards.append({"file": shard_name, "count": batch_count})

    metadata = {
        "dataset_name": dataset_name,
        "split": split,
        "image_size": image_size,
        "max_samples": max_samples,
        "num_images": total_images,
        "shards": shards,
    }
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    return cache_path


def read_cache_metadata(cache_path: Path) -> dict:
    with (cache_path / "metadata.json").open("r", encoding="utf-8") as fp:
        return json.load(fp)


def describe_cache(
    dataset_name: str,
    split: str,
    image_size: int,
    cache_dir: Path,
    max_samples: int | None,
) -> tuple[Path, dict | None]:
    cache_path = _cache_dir_path(cache_dir, dataset_name, split, image_size, max_samples)
    metadata_path = cache_path / "metadata.json"
    if not metadata_path.exists():
        return cache_path, None
    return cache_path, read_cache_metadata(cache_path)


def build_hf_dataloader(
    dataset_name: str,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    split: str = "train",
    cache_dir: Path = Path("cache"),
    max_samples: int | None = 20000,
    force_rebuild_cache: bool = False,
) -> DataLoader:
    preprocess_workers = max(1, num_workers)
    preprocess_batch_size = min(max(batch_size, 256), 2048)
    cache_path = ensure_hf_tensor_cache(
        dataset_name=dataset_name,
        split=split,
        image_size=image_size,
        cache_dir=cache_dir,
        max_samples=max_samples,
        force_rebuild=force_rebuild_cache,
        preprocess_batch_size=preprocess_batch_size,
        preprocess_workers=preprocess_workers,
    )
    wrapped_dataset = ShardedTensorDataset(cache_path=cache_path)
    return DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
