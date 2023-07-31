# Kaggle_BirdCLEF_2023
Code and notes from BirdCLEF 2023

My Strategy this year was to build a training set out of the first and last 8 seconds of each training sample, then randomly crop from 5 seconds within that (where they were long enough to do so), otherwise I padded the samples out and took just one sub-sample.  I then used my model to go over the middle portion and create a second dataset where the samples were detected.   My motivation for this was that the first and last seconds of the clip were somewhat more likely to contain the primary bird, as the clips are cropped from longer recordings.  Also by doing this I had a uniform sample length to batch into my dataloader for efficient processing.

I don't think this approach actually worked very well, and I went backwards in my final placing compared to previous years.  I didn't put any time into testing this sampling strategy as my focus for this year was more on the audio processing and image classification parts.  I'm hypothesizing here that any false positives in my initial dataset would have just flowed through into the second one.  And by using only 5 second samples (instead of say 10 seconds tranformed into a single spectrogram), I increased the chance of having false positives in the first place.

Next year I will go back to using an SED model,  and also plan to use longer training samples to see if that helps.   With some post processing to combine overlaping samples in their target 5 second inference window.   
