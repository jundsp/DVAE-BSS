# Experiments

Scripts that reproduce the baselines and figures of the dissertation (Ch. 6) from trained
models. They are not part of the installed package; run them from the repository root after

```bash
pip install -e ".[experiments]"
```

Every script has `--help`, takes its paths as arguments, and writes into an output folder.
`dvae-evaluate` and the baselines all write the same `results.json` (one record per mixture and
source, with SI-SDR/SIR/SAR and the test set's metadata), so any figure script can read any of them.

## Workflow

```bash
# 1. a pre-mixed test set from isolated sounds (or bring your own, see PremixedDataset)
python experiments/data_prep/make_test_set.py --audio data/test_notes --out data/test_set -n 500

# 2. score the model and the baselines
dvae-evaluate -m runs/k5/model.pt -d data/test_set -o results/dvae
python experiments/baselines/ideal_masks.py -d data/test_set -o results/ideal      # mix, IRM, IBM
python experiments/baselines/ica_nmf.py -d data/test_set -o results/classical       # ICA, NMF

# 3. figures
python experiments/figures/breakdown.py results/dvae/results.json -o figures/breakdown
python experiments/figures/compare_runs.py results/c1/results.json results/c05/results.json \
    --labels 1 0.5 --xlabel "compression" -o figures/compression
```

## What reproduces what

| Script | Output | Dissertation |
|---|---|---|
| `dvae-evaluate`, `baselines/ideal_masks.py` | `summary.json` per method | tables of BSS-eval on RWC and NSynth (DVAE, mix, IBM, IRM) |
| `figures/compare_runs.py` | box and line plots over models trained with different settings | compression, number of sources and variance results |
| `figures/breakdown.py` | SI-SDR by instrument, instrument pair, time offset, pitch difference | RWC / NSynth / two-note result figures |
| `figures/qualitative.py` | mixture, true sources and estimates as spectrograms, plus wav files | qualitative separation examples |
| `figures/model_internals.py` | sources, latent states, prior samples, encoder and decoder feature maps | feature and latent-state figures |
| `baselines/ica_nmf.py`, `baselines/mixit.py` | classical and MixIT baselines | not in the dissertation's dynamical tables (see below) |
| `data_prep/*` | resampling and splitting of sounds, file lists, test-set builder | dataset preparation |

To compare settings, train one model per setting (`dvae-train --compression 0.3 ...`), evaluate
each, and pass the `results.json` files with `--labels` to `compare_runs.py`.

## Notes and differences from the original scripts

- **Ideal masks** are computed from the true, unscaled sources. The original script rescaled the
  sources by the mixture's gain before building the masks (it reused one variable), and also
  stored the mixture, IRM and IBM results in one shared dictionary. Numbers can therefore differ
  slightly from the dissertation's IBM/IRM columns.
- **MixIT** was never scored in the original repository (`baseline_mixit.py` only plotted
  outputs). `mixit.py eval` groups the model's `K` outputs into the reference sources with the
  oracle split that maximises SI-SDR, the usual way to evaluate MixIT. Treat it as a new
  implementation of the baseline, not as the code behind any published number.
- **NMF** needs the [NMF Toolbox](https://github.com/groupmm/NMFtoolbox) (not on PyPI) on
  `PYTHONPATH`. It is imported only when requested and is not exercised by the test suite.
- **Not ported:** the spectrum-average plot of the compression study (its data-producing code was
  already missing from the repository) and Griffin-Lim audio of prior samples.
- The `place` column of `results.json` is the rank of a source by SI-SDR within its mixture
  (0 = worst). `compare_runs.py` shows it as "estimate" (1 = best).
