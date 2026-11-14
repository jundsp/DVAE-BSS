# Pretrained model

`dvae_rwc_k5.pt` is a `K = 5` model from the [dissertation's](https://mcgill.scholaris.ca/server/api/core/bitstreams/2334758f-542d-45cc-897d-37b6dd949f12/content) experiments (run `230122-13058078`,
the `K = 5` model of the number-of-sources study), converted to the current checkpoint format
and stored in float16. It was trained on two-stream mixtures of RWC instrument notes at 16 kHz,
with `c = 0.5` and `σ² = 0.1`.

Which saved checkpoint produced the dissertation's tables is not recorded. This one was chosen
because it scored best among the `K = 5` RWC checkpoints that survive from that work, in
SI-SDR (dB) on test sets anyone can regenerate:

| Test set | Median | Mean |
|---|---:|---:|
| TwoNote (`dvae-make-twonote`, 1326 mixtures) | 13.1 | 10.9 |
| Synthetic notes (`dvae-make-synthetic`, 500 mixtures) | 11.1 | 12.7 |

The model shipped before November 2026 (run `231017-22161497`) gets 13.0 / 11.3 on TwoNote and
9.7 / 11.8 on the synthetic set. Neither model separates two notes of the same pitch.

The weights are licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
following the licence of the
[RWC Instruments Database](https://doi.org/10.5281/zenodo.17170844) (Goto et al., ISMIR 2003).
