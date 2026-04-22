from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from .config import TrainConfig
from .data import build_hf_dataloader, describe_cache
from .models import BetaVAE
from .utils import ensure_dir, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a beta-VAE on anime images.")
    parser.add_argument("--hf-dataset", type=str, default="huggan/anime-faces", help="Hugging Face dataset name.")
    parser.add_argument("--hf-split", type=str, default="train", help="Split name when using --hf-dataset.")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"), help="Directory used for preprocessed tensor caches.")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Rebuild the local tensor cache even if it already exists.")
    parser.add_argument("--max-samples", type=int, default=20000, help="Maximum number of images to preprocess and train on.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--recon-loss", type=str, default="l1", choices=["l1", "mse", "bce"], help="Reconstruction loss.")
    parser.add_argument("--kl-warmup-epochs", type=int, default=0, help="Linearly increase beta during the first N epochs.")
    parser.add_argument("--beta-sweep", type=float, nargs="*", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def save_training_curves(history: list[dict[str, float]], output_path: Path, beta: float) -> None:
    epochs = [entry["epoch"] for entry in history]
    losses = [entry["loss"] for entry in history]
    recon_losses = [entry["recon_loss"] for entry in history]
    kl_losses = [entry["kl_loss"] for entry in history]
    effective_betas = [entry["effective_beta"] for entry in history]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, label="Total loss", linewidth=2)
    plt.plot(epochs, recon_losses, label="Reconstruction loss", linewidth=2)
    plt.plot(epochs, kl_losses, label="KL loss", linewidth=2)
    plt.plot(epochs, effective_betas, label="Effective beta", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training curves for beta={beta}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_one_beta(config: TrainConfig) -> None:
    seed_everything(config.seed)
    device = resolve_device(config.device)

    run_name = f"beta_{str(config.beta).replace('.', '_')}"
    run_dir = ensure_dir(config.output_dir / run_name)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")

    cache_path, cache_metadata = describe_cache(
        dataset_name=config.hf_dataset,
        split=config.hf_split,
        image_size=config.image_size,
        cache_dir=config.cache_dir,
        max_samples=config.max_samples,
    )
    if config.force_rebuild_cache:
        print(f"[cache] Rebuilding cache at {cache_path}")
    elif cache_metadata is not None:
        print(
            f"[cache] Reusing cache at {cache_path} "
            f"({cache_metadata['num_images']} images, {len(cache_metadata['shards'])} shards)"
        )
    else:
        print(f"[cache] Building cache at {cache_path}")

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
    if cache_metadata is None or config.force_rebuild_cache:
        _, cache_metadata = describe_cache(
            dataset_name=config.hf_dataset,
            split=config.hf_split,
            image_size=config.image_size,
            cache_dir=config.cache_dir,
            max_samples=config.max_samples,
        )
    if cache_metadata is not None:
        print(
            f"[data] Using {cache_metadata['num_images']} images from {len(cache_metadata['shards'])} shards"
        )
    print(f"[device] Training on {device} with batch_size={config.batch_size}, num_workers={config.num_workers}")
    print(f"[loss] Using reconstruction loss: {config.recon_loss}")
    if config.kl_warmup_epochs > 0:
        print(f"[warmup] KL warm-up enabled for {config.kl_warmup_epochs} epochs toward beta={config.beta}")
    else:
        print(f"[warmup] KL warm-up disabled, using fixed beta={config.beta}")

    model = BetaVAE(latent_dim=config.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        metrics = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
        if config.kl_warmup_epochs > 0:
            warmup_progress = min(epoch / config.kl_warmup_epochs, 1.0)
            effective_beta = config.beta * warmup_progress
        else:
            effective_beta = config.beta

        progress = tqdm(dataloader, desc=f"beta={config.beta} epoch {epoch}/{config.epochs}")
        for images, _ in progress:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)

            reconstruction, mu, logvar = model(images)
            loss, batch_metrics = model.loss_function(
                reconstruction=reconstruction,
                target=images,
                mu=mu,
                logvar=logvar,
                beta=effective_beta,
                recon_loss_type=config.recon_loss,
            )
            loss.backward()
            optimizer.step()

            for key, value in batch_metrics.items():
                metrics[key] += value
            progress.set_postfix(
                loss=batch_metrics["loss"],
                recon=batch_metrics["recon_loss"],
                kl=batch_metrics["kl_loss"],
                beta=effective_beta,
            )

        num_batches = len(dataloader)
        epoch_metrics = {key: value / num_batches for key, value in metrics.items()}
        epoch_metrics["epoch"] = epoch
        epoch_metrics["effective_beta"] = effective_beta
        history.append(epoch_metrics)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": vars(config),
            "epoch": epoch,
            "metrics": epoch_metrics,
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if epoch_metrics["loss"] < best_loss:
            best_loss = epoch_metrics["loss"]
            torch.save(checkpoint, checkpoint_dir / "best.pt")

    with (run_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2, default=str)
    save_training_curves(history, run_dir / "training_curves.png", config.beta)


def main() -> None:
    args = parse_args()
    betas = args.beta_sweep if args.beta_sweep else [args.beta]
    for beta in betas:
        config = TrainConfig(
            hf_dataset=args.hf_dataset,
            hf_split=args.hf_split,
            cache_dir=args.cache_dir,
            force_rebuild_cache=args.force_rebuild_cache,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            latent_dim=args.latent_dim,
            beta=beta,
            recon_loss=args.recon_loss,
            kl_warmup_epochs=args.kl_warmup_epochs,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
        )
        train_one_beta(config)


if __name__ == "__main__":
    main()
