
## Learnings from the high scoring entries
Will complete this some other time when I'm less busy.  Apparently using Chirp or Birdnet to generate embeddings is going to be an important part of any future competition.

## Learnings from my own work

### Coding
- `pd.set_option('display.max_colwidth', None)`  Stops notebook chopping pd dataframes too narrow
- `torch.set_flush_denormal(True)`  Prevents really small training weights from blowing out the inference time
- `Config.PRECISION = 16 if accelerator == 'gpu' else 32` Allows mixed precision training when the GPU is in use.
- `skf =StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)` Good practice even if I only use one fold, since it ensures an even class distribution between training and validation.

### Audio
- I suspect chopping my spectrogram in half and stacking vertically was a good idea, it just didn't help me for other reasons.  See next point below.
- Use an SED model, evaluate 10-15 second samples, and move the window by 5 seconds each time, instead of just doing 5 second samples.  Should reduce the incidence of false positives in the training dataset, and get a better result by averaging three windows.
- Use time and freqency masking
- Not much to gain from random cropping, since it reduces the amount of time in the sample.  Instead consider rotating one end of the sample to the beginning.
- Use ogg files.   Any theoretical gain from storing data in .npy or .wav wasn't worth the hassle of the huge files.

### ML
- Ensemble 2-3 models next time!   Even if it means using smaller networks to meet process time requirements. Persevering with single-models and focusing on good principles is very noble, but it's also like taking a knife to a pistol fight.
- Use mixup, and if appropriate, cutmix
- Try label-smoothing, but note that this is mainly a strategy against over-training, rather than a way to get a direct performance boost.
- Try undersampling and over-sampling to deal with class imbalance.  
- Macro-average precision is a very useful metric.  Implement this in my other work.  It gives a better measure of model quality as all classes are treated equally, and there is no effect of threshold choice.




