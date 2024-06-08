
-   sr: 32000, window_size: 2048, hop_size: 1024, fmin: 200, fmax: 14000, mel_bins: 224
-   sr: 32000, window_size: 1024, hop_size: 512, fmin: 50, fmax: 14000, mel_bins: 128

https://stackoverflow.com/questions/62584184/understanding-the-shape-of-spectrograms-and-n-mels

When computing an [STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform), you compute the FFT for a number of short segments. These segments have the length `n_fft`. Usually these segments overlap (in order to avoid information loss), so the distance between two segments is often not `n_fft`, but something like `n_fft/2`. The name for this distance is `hop_length`. It is also defined in samples.

So when you have 1000 audio samples, and the hop_length is 100, you get 10 features frames (note that, if `n_fft` is greater than hop_length, you may need to pad).

In your example, you are using the default `hop_length` of 512. So for audio sampled at 22050 Hz, you get a feature frame rate of

```python
frame_rate = sample_rate/hop_length = 22050 Hz/512 = 43 Hz
```

Again, padding may change this a little.

So for 10s of audio at 22050 Hz, you get a spectrogram array with the dimensions `(128, 430)`, where 128 is the number of Mel bins and 430 the number of features (in this case, Mel spectra).