# Experiments

Baselines and figure scripts for Chapter 6 of the [dissertation](https://mcgill.scholaris.ca/server/api/core/bitstreams/2334758f-542d-45cc-897d-37b6dd949f12/content). Install with `pip install -e ".[experiments]"` and
run from the repository root. Every script has `--help`. `dvae-evaluate` and the baselines all
write the same `results.json`, so any figure script can read any of them.

| Script | [Dissertation](https://mcgill.scholaris.ca/server/api/core/bitstreams/2334758f-542d-45cc-897d-37b6dd949f12/content) |
|---|---|
| `dvae-evaluate`, `baselines/ideal_masks.py` | BSS-eval tables (DVAE, mix, IBM, IRM) |
| `figures/compare_runs.py` | compression, number of sources and variance studies |
| `figures/breakdown.py` | SI-SDR by instrument, pair, time offset, pitch difference |
| `figures/qualitative.py` | separation examples |
| `figures/model_internals.py` | feature maps and latent states |
| `baselines/ica_nmf.py`, `baselines/mixit.py` | classical and MixIT baselines |
| `figures/make_demo_page.py` | the examples page in `docs/` |
| `data_prep/` | dataset preparation and the test-set builder |
| `data_prep/make_nsynth_examples.py` | the NSynth mixtures on the demo page and in `examples/` |
| `dvae-make-twonote` (`src/dvae_bss/datagen/twonote.py`) | the TwoNote test set (rebuilt) |
| `dvae-make-synthetic` (`src/dvae_bss/datagen/synthetic.py`) | synthetic notes and test mixtures, no licence needed |

Differences from the original scripts:

- The ideal masks are computed from the unscaled true sources. The original rescaled them by
  mistake, so the IBM/IRM numbers can differ slightly from the dissertation.
- MixIT was never scored in the original code, so `mixit.py eval` is new.
- NMF needs the [NMF Toolbox](https://github.com/groupmm/NMFtoolbox) on `PYTHONPATH`.
