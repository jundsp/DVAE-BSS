# DVAE-BSS

Dynamical variational auto-encoding for unsupervised blind sound separation.

PyTorch code for unsupervised blind source separation of audio with dynamical (sequential) variational auto-encoders. The model learns to separate mixtures into sources from mixtures alone. It is trained without isolated source signals.

This is the research code behind Chapter 6 of my Ph.D. dissertation; the tagged release `dissertation-2024` matches it. It extends [VAE-BSS](https://github.com/jundsp/VAE-BSS), the non-dynamical model from the EUSIPCO 2021 paper below, with temporal (recurrent) latent dynamics.

## Contents

| Path | Purpose |
|---|---|
| `model.py` | Models: sequential VAE (`SVAE`), dilated variant (`DSVAE`), convolutional encoders/decoders, likelihoods |
| `train.py`, `argparser.py` | Training (supports multi-GPU via `torch.distributed`); run `python train.py --help` for options |
| `dataset.py`, `data_augmentations.py` | Loaders for RWC, NSynth, VCTK, MUSDB, MUMS, MNIST, and synthetic two-note mixtures |
| `lossfun.py`, `utils/` | Losses, STFT/iSTFT, permutation-invariant evaluation helpers, dataset preparation scripts |
| `evaluate*.py`, `evaluation_*.py`, `infer.py` | Evaluation (BSS-eval) and inference on audio files |
| `baseline_*.py`, `train_mixit.py` | Baselines: NMF, ideal binary/ratio masks, MixIT |
| `summarize/` | Scripts that produce the result figures (compression, number of sources, likelihood variance, dilated vs. strided) |

Scripts expect datasets under `data/` and write to `output/`; adjust paths at the top of each script or via `--data-directory` / `--output-directory`. Datasets are not included.

## Setup

```bash
pip install -r requirements.txt
```

Developed with Python 3.8 and PyTorch 1.10, trained on Compute Canada GPU clusters (`utils/run.sh` is an example Slurm script).

## References

- J. Neri, *Bayesian Hierarchical Modeling and Inference of Mixed Audio Sources*, Ph.D. dissertation, McGill University, 2024.
- J. Neri, R. Badeau, and P. Depalle, "Unsupervised Blind Source Separation with Variational Auto-Encoders," in *Proc. 29th European Signal Processing Conference (EUSIPCO)*, Dublin, Ireland, 2021. Code: [VAE-BSS](https://github.com/jundsp/VAE-BSS).
- J. Neri, P. Depalle, and R. Badeau, "Dynamical Variational Auto-Encoding Unsupervised Blind Sound Separation," in preparation.

## License

GPL-3.0. See [LICENSE](LICENSE).
