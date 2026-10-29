# Pretrained model

`dvae_rwc_k5.pt` is the model behind the RWC and NSynth results in Chapter 6 of the
dissertation (run `231017-22161497` at tag `dissertation-2024`), converted to the current
checkpoint format and stored in float16. It was trained on two-stream mixtures of RWC
instrument notes at 16 kHz, with `K = 5`, `c = 0.5` and `σ² = 0.1`. On the TwoNote test set it
gets 11.35 dB SI-SDR, the same as the original code.

The weights are licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
following the licence of the
[RWC Instruments Database](https://doi.org/10.5281/zenodo.17170844) (Goto et al., ISMIR 2003).
