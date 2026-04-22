from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from .config import EvalConfig
from .data import build_hf_dataloader
from .models import BetaVAE
from .utils import ensure_dir, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained beta-VAE.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hf-dataset", type=str, default="huggan/anime-faces", help="Hugging Face dataset name.")
    parser.add_argument("--hf-split", type=str, default="train", help="Split name when using --hf-dataset.")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"), help="Directory used for preprocessed tensor caches.")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Rebuild the local tensor cache even if it already exists.")
    parser.add_argument("--max-samples", type=int, default=20000, help="Maximum number of images to preprocess and evaluate from.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--interpolation-steps", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_model(checkpoint_path: Path, latent_dim: int, device: torch.device) -> BetaVAE:
    model = BetaVAE(latent_dim=latent_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def save_reconstructions(model: BetaVAE, dataloader, device: torch.device, output_dir: Path) -> None:
    images, _ = next(iter(dataloader))
    images = images.to(device)
    reconstructions, _, _ = model(images)
    grid = make_grid(torch.cat([images[:8], reconstructions[:8]], dim=0), nrow=8)
    save_image(grid, output_dir / "reconstructions.png")


@torch.no_grad()
def save_samples(model: BetaVAE, num_samples: int, latent_dim: int, device: torch.device, output_dir: Path) -> None:
    z = torch.randn(num_samples, latent_dim, device=device)
    samples = model.decode(z)
    grid = make_grid(samples, nrow=max(1, min(4, num_samples)))
    save_image(grid, output_dir / "samples.png")


@torch.no_grad()
def save_interpolation(
    model: BetaVAE,
    dataloader,
    interpolation_steps: int,
    device: torch.device,
    output_dir: Path,
) -> None:
    images, _ = next(iter(dataloader))
    images = images[:2].to(device)
    mu, _ = model.encode(images)
    z0, z1 = mu[0], mu[1]
    weights = torch.linspace(0.0, 1.0, steps=interpolation_steps, device=device)
    latent_path = torch.stack([(1 - alpha) * z0 + alpha * z1 for alpha in weights], dim=0)
    decoded = model.decode(latent_path)
    grid = make_grid(decoded, nrow=interpolation_steps)
    save_image(grid, output_dir / "interpolation.png")


def main() -> None:
    args = parse_args()
    config = EvalConfig(
        checkpoint=args.checkpoint,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        cache_dir=args.cache_dir,
        force_rebuild_cache=args.force_rebuild_cache,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        interpolation_steps=args.interpolation_steps,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        device=args.device,
    )

    output_dir = ensure_dir(config.output_dir)
    device = resolve_device(config.device)

    dataloader = build_hf_dataloader(
        dataset_name=config.hf_dataset,
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        split=config.hf_split,
        cache_dir=config.cache_dir,
        max_samples=config.max_samples,
        force_rebuild_cache=config.force_rebuild_cache,
    )
    model = load_model(config.checkpoint, config.latent_dim, device)

    save_reconstructions(model, dataloader, device, output_dir)
    save_samples(model, config.num_samples, config.latent_dim, device, output_dir)
    save_interpolation(model, dataloader, config.interpolation_steps, device, output_dir)


if __name__ == "__main__":
    main()
