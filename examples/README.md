# Example

`mixture.wav` is mixture 365 of the TwoNote test set used in the dissertation: two synthesised
harmonic notes at 16 kHz, 3.2 s. A sustained note starts first, and a decaying note 8 semitones
higher starts 0.5 s later. The pretrained model puts each note in its own output (about 16 dB
SI-SDR each) and leaves the other three outputs silent.

```bash
dvae-separate examples/mixture.wav --plot
```

The audio was made by the author and is released with the code under GPL-3.0.
