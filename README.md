# Kaggle_BirdCLEF_2023

I learned a lot from this year though I didn't do so well in the leaderboard.  It did help me immensely with future projects, both in and outside of Kaggle.

I struggled to find a CV scheme that bore any relationship with the leaderboard position.  It is clear with hind sight that this is one of the main challenge with BirdCLEF, as the models perform so differently on the Xeno-Canto training samples compared to real soundscapes. 

I played about this year with augmentation, and spectrogram generation methods, and also tried breaking my spectrogram up into two halfs, and stacking them on top of each other to make a square image with better temporal resolution.  This didn't seem to help in the end.  I also tried to tackle the weakly labeled nature of the dataset by a threshold based pseudolabelling scheme on the training data with the first iteration of the model.
