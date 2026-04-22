# Beta-VAE on Anime Images

This repository turns the mini-report into a concrete course project:

- reproduce a convolutional `beta-VAE`;
- train it on anime-style images resized to `64x64`;
- compare several values of `beta`;
- evaluate reconstructions, random samples, and latent interpolations.

## How the report maps to the code

- `src/models/beta_vae.py`: convolutional encoder/decoder and VAE loss.
- `src/train.py`: training loop and beta comparison runs.
- `src/evaluate.py`: reconstruction, sampling, and interpolation outputs.
- `src/data.py`: dataset loading from Hugging Face with a reusable local tensor cache.

The implementation follows the standard VAE formulation from Auto-Encoding Variational Bayes and adds the `beta` coefficient from beta-VAE:

`loss = reconstruction_loss + beta * kl_divergence`

The default reconstruction loss is now `L1`, which is often less blurry than `MSE` for image reconstruction.
You can also enable KL warm-up to gradually increase the KL weight during the first epochs.

## Dataset caching

The first training run preprocesses the Hugging Face images once and stores a tensor cache under `cache/`.
Later runs reuse the cached shards instead of decoding thousands of PNG files again.
By default, the project limits preprocessing to the first `20000` images for faster experimentation.
Training now prints cache status messages so you can see whether it is rebuilding or reusing the cache.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --hf-dataset huggan/anime-faces --beta 4.0 --epochs 20
```

For Colab, a small amount of dataloader parallelism is enabled by default with `num_workers=2`.
If you want to put the cache on local Colab storage for better I/O stability, pass `--cache-dir /content/cache`.
The default image cap is `--max-samples 20000`.

To compare several beta values in separate runs:

```bash
python -m src.train --hf-dataset huggan/anime-faces --beta-sweep 1.0 2.0 4.0 8.0
```

You can switch the reconstruction loss if needed:

```bash
python -m src.train --hf-dataset huggan/anime-faces --beta 1.0 --recon-loss l1
python -m src.train --hf-dataset huggan/anime-faces --beta 1.0 --recon-loss mse
python -m src.train --hf-dataset huggan/anime-faces --beta 1.0 --recon-loss bce
```

You can enable KL warm-up like this:

```bash
python -m src.train --hf-dataset huggan/anime-faces --beta 1.0 --recon-loss l1 --kl-warmup-epochs 10
```

Outputs are written under `outputs/`.

## Evaluate

Generate reconstructions, random samples, and latent interpolations from a checkpoint:

```bash
python -m src.evaluate --checkpoint outputs/beta_4_0/checkpoints/best.pt --hf-dataset huggan/anime-faces
```

## Colab example

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/GEAI
!pip install -r requirements.txt
!python -m src.train --hf-dataset huggan/anime-faces --cache-dir /content/cache --beta-sweep 1.0 2.0 4.0 8.0 --epochs 20
!python -m src.evaluate --checkpoint outputs/beta_4_0/checkpoints/best.pt --hf-dataset huggan/anime-faces --cache-dir /content/cache
```

The dataset is downloaded by the `datasets` library directly from Hugging Face when the script runs.

## Important note about your API URL

This endpoint:

```bash
https://datasets-server.huggingface.co/first-rows?dataset=huggan%2Fanime-faces&config=default&split=train
```

is a preview endpoint that returns the first rows of a dataset, not the full training download API. Hugging Face documents `/first-rows` as a dataset viewer endpoint for previewing up to the first 100 rows, which is useful for schema inspection but not for fetching the whole dataset for training. Sources:

- [Hugging Face dataset viewer docs](https://huggingface.co/docs/dataset-viewer/en/first_rows)
- [huggan/anime-faces dataset card](https://huggingface.co/datasets/huggan/anime-faces)

## What to include in your final report

- brief reminder of VAE and why beta-VAE changes the objective;
- dataset description and preprocessing;
- model architecture and training settings;
- visual comparison of reconstructions for different `beta`;
- visual comparison of random samples for different `beta`;
- interpolation examples in latent space;
- short discussion of the trade-off: higher `beta` often improves latent structure but can hurt reconstruction quality.
