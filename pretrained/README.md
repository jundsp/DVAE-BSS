# Pretrained model: `dvae_rwc_k5.pt`

The DVAE behind the RWC and NSynth results of the dissertation (Ch. 6), converted to the
package's checkpoint format. `dvae-separate` uses it when no `--model` is given.

```python
from dvae_bss import load_model

model, stft = load_model("pretrained/dvae_rwc_k5.pt")
```

## Model

| | |
|---|---|
| Architecture | strided convolutional encoder/decoder (32-64-128-128-128 channels), bi-LSTM inference network, LSTM dynamical prior with 512 units, latent dimension 256 |
| Outputs | `K = 5` sources |
| Front end | 16 kHz mono, STFT with 512-point Hann window and hop 128, magnitude compression `c = 0.5` |
| Likelihood | scale-invariant Normal, variance `σ² = 0.1` |
| Parameters | 11.7 M, stored in float16 (23 MB) and loaded as float32 |
| Origin | run `231017-22161497` of the code at tag `dissertation-2024` |

Storing the weights in float16 changes separated waveforms by at most about 1e-3 and the mean
SI-SDR on the examples below by 0.001 dB.

## Training

Trained without any isolated sources, on mixtures of two streams generated on the fly from
single notes of the RWC Musical Instrument Sound Database (training split). Each stream is a
sequence of random notes with random gaps, about 8 s long (1025 STFT frames). AdamW, learning
rate 1e-4, weight decay 2e-5, until the evidence lower bound converged (about 500 epochs).

## Evaluation

SI-SDR, SIR and SAR in dB, averaged over all true sources of a test set. Each true source is
matched to the closest of the five outputs.

| Test set | SI-SDR | SIR | SAR | Source |
|---|---:|---:|---:|---|
| RWC test (two-note mixtures) | 13.26 | 36.07 | 14.61 | dissertation |
| NSynth test (two-note mixtures, not seen in training) | 9.58 | 35.16 | 11.64 | dissertation |
| TwoNote (1326 synthetic two-note mixtures) | 11.35 | 26.70 | 13.43 | re-measured with this file and `dvae-evaluate` |

The TwoNote re-measurement matches the original code's results for this run per mixture
(95 % of values within 0.01 dB, same mean). On 16 NSynth examples saved by the original code,
this package reproduces the original estimates to within about 1e-6.

## Intended use and limitations

- Research on unsupervised source separation, and a starting point for fine-tuning.
- Trained on isolated, mostly harmonic notes of acoustic instruments at 16 kHz. It works best
  on mixtures of a few such notes and generalises less well to other sounds: see the lower
  NSynth score, which includes synthesised and electronic sounds.
- Not trained on speech, full songs, percussion-heavy material or reverberant recordings.
- Outputs are unordered. A mixture of two sources usually fills two of the five outputs, and the
  others stay near silent. Notes that start together at the same pitch are often not separated.

## Licence

The weights are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), matching the licence of the
training data. The code in this repository is GPL-3.0.

Training data: M. Goto, H. Hashiguchi, T. Nishimura, and R. Oka, "RWC Music Database: Music
Genre Database and Musical Instrument Sound Database," in *Proc. ISMIR*, 2003. Re-released in
2026 under CC BY-NC 4.0: [RWC Instruments Database](https://doi.org/10.5281/zenodo.17170844).
