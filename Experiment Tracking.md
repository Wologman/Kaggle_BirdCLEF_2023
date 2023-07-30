
Baseline: Efficientnet_b0_ns [from here](https://www.kaggle.com/code/nischaydnk/birdclef-2023-pytorch-lightning-training-w-cmap)

epoch 4 validation loss 0.3892030119895935
epoch 4 validation C-MAP score pad 5 0.7863115654380066
epoch 4 validation C-MAP score pad 3 0.7272426590726732
epoch 4 validation AP score 0.7434433454616223

epoch 7 validation loss 0.4265593886375427
epoch 7 validation C-MAP score pad 5 0.8102616305995011
epoch 7 validation C-MAP score pad 3 0.7558257208483178
epoch 7 validation AP score 0.7840074052936382
LB 0.77


## Experiment 1:  EfficientNetV2

epoch 4 validation loss 0.5133525133132935
epoch 4 validation C-MAP score pad 5 0.7769704190741368
epoch 4 validation C-MAP score pad 3 0.7150149243158047
epoch 4 validation AP score 0.7333005957188452
LB 0.75

Self-stopped, monitoring val loss


## Experiment 2:  EfficientNetV2s21k
min delta for stopping = 0.05

epoch 4 validation loss 0.0433591827750206
epoch 4 validation C-MAP score pad 5 0.8031072083713884
epoch 4 validation C-MAP score pad 3 0.7493177295153591
epoch 4 validation AP score 0.767883196340048


## Experiment 3:  EfficientNetV2s21k
min delta = 0.05
LR = 5e-5

epoch 6 validation loss 0.25485432147979736
epoch 6 validation C-MAP score pad 5 0.7422590514044632
epoch 6 validation C-MAP score pad 3 0.6742159424792
epoch 6 validation AP score 0.6880393243403731


## Experiment 4:  EfficientNetV2s21k
min_delta = 0.1  (I misunderstand this parameter - min value change to count as an IMPROVEMENT)
LR = 5e-4 (baseline)

No improvement

## Experiment 5:  EfficientNetV2s21k
min delta = 0 (baseline)
Patience = 32 (baseline = 8)

Definately too much patience.  Was still training and getting worse at 10 epochs.  Currently checking 2x per epoch.  That is probably about right.  Just increase the patience to 12.

epoch 4 validation loss 0.0433591827750206
epoch 4 validation C-MAP score pad 5 0.8031072083713884
epoch 4 validation C-MAP score pad 3 0.7493177295153591
epoch 4 validation AP score 0.767883196340048

epoch 10 validation loss 0.05074794963002205
epoch 10 validation C-MAP score pad 5 0.8354327913019514
epoch 10 validation C-MAP score pad 3 0.7893742878952542
epoch 10 validation AP score 0.8281749756666639

## Experiment 6:  EfficientNetV2s21k
min delta = 0 (baseline)
Patience = 12 (baseline = 8)

epoch 4 validation loss 0.0433591827750206
epoch 4 validation C-MAP score pad 5 0.8031072083713884
epoch 4 validation C-MAP score pad 3 0.7493177295153591
epoch 4 validation AP score 0.767883196340048

epoch 10 validation loss 0.05074794963002205
epoch 10 validation C-MAP score pad 5 0.8354327913019514
epoch 10 validation C-MAP score pad 3 0.7893742878952542
epoch 10 validation AP score 0.8281749756666639

Not sure why C_Map keep improving long after the loss term is optimal.  Still investigating.  Possibly due to the use of the wrong loss mechanism. Try replacing with crossentropyloss

## Experiment 7: CrossEntropyLoss() - LB 0.78

epoch 18 validation loss 1.054071068763733
epoch 18 validation C-MAP score pad 5 0.8412279049249943
epoch 18 validation C-MAP score pad 3 0.7969627738596239
epoch 18 validation AP score 0.833789723789657
LB 0.78 (But an improvement in position over expt 6.

Big improvement to the training, stability, and match between loss and CMAP.  Stick with this!  Reduce the patience back to 8.  That was probably reasonable, just the loss was too unstable.

## Experiment 10: New Baseline  - LB 0.77

Train version 41, Infer Version 19
MODEL: tf_efficientnetv2_s_in21k
MelSpec:   HOP_LENGTH = 512, N_FFT = 1024, N_MELS = 128
8 second chunkes from beginning and end, randomly crop 5 secs
Some wave augmentation
MIXUP_ALPHA = 0.2
PATIENCE = 4
FMIN = 20
FMAX = 16000
LR = 5e-4

epoch 8 validation loss 1.3673970699310303
epoch 8 validation C-MAP score pad 5 0.8445087966813041
epoch 8 validation C-MAP score pad 3 0.8030302135199845
epoch 8 validation AP score 0.7879463094268143

## Experiment 11: PCEN

Very unstable performance perhaps consider reducing the learning rate or adding momentum.  

epoch 4 validation loss 1.607954978942871
epoch 4 validation C-MAP score pad 5 0.8040465107851941
epoch 4 validation C-MAP score pad 3 0.7516582913775843
epoch 4 validation AP score 0.7285973206173826

epoch 6 validation loss 1.96861732006073
epoch 6 validation C-MAP score pad 5 0.7938789440154892
epoch 6 validation C-MAP score pad 3 0.7387371347920555
epoch 6 validation AP score 0.6722138759808616

## Experiment 12: PCEN + Parameter changes

Torch defaults for adam: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
[Hidashi Arai's PCEN params](https://www.kaggle.com/code/hidehisaarai1213/birdcall-resnestsed-effnet-b0-ema-all-th04): 
```python
pcen_parameters = {
    "gain": 0.98,
    "bias": 2,
    "power": 0.5,
    "time_constant": 0.4,
    "eps": 0.000001}
    
melspectrogram_parameters = {
    "n_mels": 128,
    "fmin": 20,
    "fmax": 16000}
```

Stopped its self after 4 epochs, still unstable

## Experiment 13: PCEN + Parameter changes - LB 0.75

Change fmax to 14,000, and LR to 1e-5

epoch 15 validation loss 1.5439732074737549
epoch 15 validation C-MAP score pad 5 0.8311729336765122
epoch 15 validation C-MAP score pad 3 0.7871052448426695
epoch 15 validation AP score 0.7550373083216941



## Experiment 14: Melspec + Background Noise

fmax to 14,000, and LR to 1e-4

epoch 27 validation loss 1.0146327018737793
epoch 27 validation C-MAP score pad 5 0.875263584620366
epoch 27 validation C-MAP score pad 3 0.841088661847015
epoch 27 validation AP score 0.8343872916232328

epoch 29 validation loss 1.0268561840057373
epoch 29 validation C-MAP score pad 5 0.8753144970705895
epoch 29 validation C-MAP score pad 3 0.8409932326938147
epoch 29 validation AP score 0.8304511935645512

Currently failing submission by timing out.

## Experiment 15:  Changed datatype scaling

Train with a float normalised to [0,255], quantized to uint8. 
That didn't work so I used a float on [0,255]


Still failing submission by timing out.

## Experiment 16: More changes to scaling

Trained & infer with float on [0,1], still timing out.
Repeated on Robby Neils Inference notebook
Trained & infer with uint8 on [0,255], get errors


## Experiment 17:  Investigate timeout

Forked Train version 41, infer version 19 (from experiment 10)
Setup infer directly in the train notebook, to find out what is breaking the code.
Could be:
Loading method
Saving method
Missmatch between preprocessing steps
Start from a working notebook, 
- Check the problem is replicated with the 'bad' weights, 
- Check the notebook is still making 'good' weights. 
- Then continue modifying the training until the problem occurs, or it works & I can use for infer.
- Once it works, fork the notebook, delete training and analysis and use for inference & submissions

Found it worked with 4 & 6 but fell over when I just let it run.  Something must be wrong with the validation, or the callbacks, so it's overtraining.  Note that I previously had a decent result with 9 epochs from the original notebook.

## Experiment 18: LB  - 0.79

Made changes to the validation part, on cutmix, so now it's using CrossEntropyLoss().  Noted that the cmap is using the logits, not probs, no softmax used, so not relevent.

LB 0.79 after 7 epochs
2 Hours, 39 minutes to train   = 22 min/epoch

8622.4s 79  epoch 7 validation loss 1.375878095626831
8622.4s 80 epoch 7 validation C-MAP score pad 5 0.4975795031713206
8622.4s 81 epoch 7 validation C-MAP score pad 3 0.5038834680847067
8622.4s 82 epoch 7 validation AP score 0.7755459399679198

## Experiment 19:  LB - 0.78

Set `torch.set_flush_denormal(True)`   In the training, and chopped up the image to be 256 x 157  to see if that helps.

This one trained on its own for 18 Epochs, with patience = 4, but it had already peaked by epoch 11.  So I'm going to need to work on the validation & the end point detection.


epoch 11 validation loss 1.0729780197143555
epoch 11 validation C-MAP score pad 5 0.9009987381986319
epoch 11 validation C-MAP score pad 3 0.8720500573190078
epoch 11 validation AP score 0.8314608659245869
LB Score: 0.78


epoch 15 validation loss 1.0435360670089722
epoch 15 validation C-MAP score pad 5 0.9105115179164951
epoch 15 validation C-MAP score pad 3 0.8849517760189537
epoch 15 validation AP score 0.8378778331203828
LB Score: 0.78


epoch 17 validation loss 1.0515474081039429
epoch 17 validation C-MAP score pad 5 0.9083941841235895
epoch 17 validation C-MAP score pad 3 0.8816117467235616
epoch 17 validation AP score 0.8387618336363746
Not submitted to LB

Total time 5 hours 29 minutes (19 minutes/epoch)

## Experiment 20:  LB-0.76

Set to 12 epochs, Changed the hop length to 312, plust cropping/padding, to ensure a 256x256 image size.  

## Experiment 21:

Try a stronger classifier head:
```python
import torch.nn as nn class ClassifierHead(nn.Module):
	def __init__(self, num_classes, num_features, dropout_rate=0.5): 
		super().__init__() 
			self.dense_layer = nn.Linear(num_features, num_features)     
			self.relu = nn.ReLU(inplace=True) 
			self.dropout = nn.Dropout(p=dropout_rate) 
			self.output_layer = nn.Linear(num_features, num_classes) 
			
	def forward(self, x): 
		x = self.dense_layer(x) 
		x = self.relu(x) 
		x = self.dropout(x) 
		x = self.output_layer(x) 
		return x
```

In the model `__ init__:
```python

self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=True) 
num_features = self.backbone.classifier.in_features # Replace the old head with the new classifier head
self.classifier = ClassifierHead(num_classes, num_features)
```

in the model `__forward__:
```python
def forward(self, x): 
	x = self.backbone(x) 
	x = self.classifier(x) 
	return x
```

This didn't get anywhere, but there isn' t much to conclude from it yet, as I made changes to the validation_epoch_end, that messed up the monitoring.  Now replaced with on_validation_epoch_end,  but remembering to reset the lists of step outputs afterwards.

## Experiment 22: 
Try BCEwithlogits loss.  This is not the 'pretraining' notebook, version 6

epoch 4 train loss 0.020428864285349846
epoch 4 validation loss 0.02092205174267292
epoch 4 validation C-MAP score pad 5 0.5345939166876332
epoch 4 validation C-MAP score pad 3 0.43983363088185906
epoch 4 validation AP score 0.26213535814683464

Stopped by its self as the val loss had stagnated.  

## Experiment 23:
Finished de-bugging the monitoring code.  Pre-train with the 21, 22, 23 dataset.  Fine-tune the backbone, then save the weights and load into a training notebook with an extra output layer to the head, train that with BCE loss.

Currently version 17 should be fine, trains well on 23 dataset & has nice plots.

## Experiment 24 & 25:
First iterations at running pretraining then training.

### Pretraining 
Found that 10-4 seems about right for Leaning rate for the fine tuning.  But with only a couple of training steps to the pre-training, my fine-tuning was converging on AP of about 0.11.

Pretraining:  Unfreeze after 6 epochs:  seems a bit late, it had already settled down after about 3
Loss: Cross Entropy
Batch Size:  16   (I did this because I was worried about memory allocation)
Weighted Sampling:  None

### Fine Tuning

Fine tuning was run with:
Weights from Epoch 2 of pre-training
BCEloss
Batch Size 32
Weighted Sampling
Extended head with only the new layers unfrozen (initialised to ones)
Unfreezing the whole classifier head after 4 epochs
10-4 Learning rate
Patience = 3

The result using weights sampled from 2 train epochs was very stratified by sample size.  The pre-training model was already at AP=0.069, and topped out at 0.71, so there was probably not much improvement in subsequent epochs.   

Fine tuned Model topped out with an AP = 0.11 afte 18 epochs

## Experiment 26:

Kept the same immature weights as previous example, but repeated fine-tuning but unfreeze after 3, LR = 10-5, weighted sampling 1/sqrt(N).

This tained even more slowly,  and after 6 epochs had an AP of 0.078, and not improving much.  

## Experiment 27:

### Pretraining
Weights from epoch 11, expt 25

### Finetuning


## Experiment 28:

  
### Pretraining:  
On G15 (G7 can't seem to run this larger dataset, or code)
Switch from CE to BCE after 3 epochs
Unfreeze after 5 epochs
Weighted sampling
Patience = 2
Didn't go particularly well.  settled around AP=0.075

## Experiment 29:

As with 28 but warmup LR from 10-5 to 10-4 over first 5 epochs
Reduced the dropout rate to 0.1
Change to BCE loss after 10 epochs

This helped a little, but I realise I've been adding two new layers, when I only intended 1.  Interrupted at epoch 2.  AP = 0.104 but not improving much.  Simplifying the head to only one new layer.

## Experiment 30

Simplified the classifier head, but left the rest the same.  At Epoch 4 AP = 0.096   So still training quite slow.

## Experiment 31 
Put learning rate back to 10-4 initially  Increasing to 10-3  
Didn't change much, still stalls on AP=10.5

## Experiment 32
BCE loss and LR set to 10-3, no warmup
Self-stopped after 3 epochs as the loss was increasing.   So this suggests much lower LR might be the trick.

## Experiment 33
BCE loss and LR set to 10-5, no warmup
AP settled on about .105 after 7 epochs

## Experiment 34
BCE loss and LR set to 10-6, no warmup
AP settled on about .105 after 7 epochs

### Experiment 36
BCE loss, LR 10-4, no pre-training with birds
Same result as 34

### Experiment 37 
Replace the whole head, but use the pre-trained weights for the backbone, 
Under sample and wieghted sampling
BCE loss, LR = 10-4
Behaved much like before
I stopped after 2 epochs and changed to CrossEntropyLoss()
Looks much the same, after the second epoch0 still ahve ap on .036.
What if the problem the whole time has been CrossEntropyLoss() converging on only the most common classes?   So pre-training this way has been unhelpful?

### Experiment 38
Repeat above with CE loss  but without pre-training.  So now the only change is the sampling
No improvement, self stopped after 3 epochs.   Really common That's pretty interesting!  Will repeat with BCE loss

### Experiment 39
As above but with BCE loss  But if the result is the same.  What it's saying is that if we weight the samples, it's not really training.  So before it was "training" just by guessing the most common classes & ignoring the rest.

Try using focal loss.  Or possibly just remove some samples and make sure the algo is working at all.  Maybe all that augmentation is too much.  Also, have a look at the end result of the pre-training, once experiment 25 finishes, that should be quite revealing.

### Experiment 40
Success!  I've realised that **it was the weighted sampling that was screwing things up**.  To be specific, it was wildly over-agressive, creating more imbalance than it removed!  Also that looking at the result from Experiment 25, it seems it's not really needed.  There is little correlation between frequency and accuracy.  AP = 0.24 after 4 or so epochs.  Now to adjust the LR a bit better for next time.

### Experiment 41
It still seems to be getting stuck a little early.  Things to optimise:
- Learning Rate - increase to 1e-3
- Unfreeze the rest of the head earlier

### Experiment 42 PCEN for pre-train
- Try PCEN again, see if it brings down the gap between training & validation

### Experiment 43 Fine Tune   LR e-3  WD e-5
My learning rate is already pretty high, but the model is still converging on quite low performance.   Better than with LR of e-4 though.

Epoch 2 AP = 31.5

### Experiment 44 Fine Tune LR 5 e-3  WD 1e-4
Tried LR 5 e-3  WD 1e-4 but the performance after the first val was already less than with e-3.  Stopped and changed to 5e-4 LR, 1e-5 WD.
This didn't help much either
Stopped and changed the loss function back to crossentropy.

Decided CE actually makes more sense for this comp since no thresholds, it's the one that more matches to a predicted probability of the various classes.  Which is more like what I want.  If there are two that sound similar, better to guess both at 50% than one at 1% one at 99%, or vice versa, or two at 99 etc.    But use the no-call to set the others to zero, assuming it has a high precision.

Epoch 1 AP = 29.9

But anyway, this result was similar to the previous ones.  Maybe it's the training head?
Try replacing the whole head instead.

### Experiment 45
Replaced the whole classifier head.  
Used pre-training from Expt 26 epoch 11
CrossEntropyLoss
LR 5e-4
It trained really well stopped before end.   
cmap5 = 87 after 12 epochs
submitted to comp, with the no-call column removed from final dataframe & softmax in use

### Experiment 46
Tried the same, but with the whole head, plus a relu and dropout=0.35, just in case this was the problem.    Behaved as before, settling around 0.27

### Experiment 47
As with 46, but releasing the whole classifier head after just one epoch instead of 4. 
Didn't help, same result

### Experiment 48,
One last try, unfreezing at epoch 0
No improvement, still gets stuck around AP=31 

### Experiment 49,
Repeat, but with LR = 10-3
Epoch 2 seems stagnated around 0.29

## Experient 50
LR 1e-3
Set running for max 24 epochs
Replace Head

## Experient 51
pre-training
melspec
LR 1e-3
BCE loss
batch size 32.

stopped after 1 epoh and changed back to CE loss
stopped after 1 epoch as it seemed to have stalled around .3.  put back to 1e-4, and changed decay to 1e-5 (from 1e-4)

### Experiment 52
fine tuning but with less augmentation
backbone tune after 8 epochs
Performed about the same as 50.  CMAP peaked around 84.5 after 8 epochs.

### Experiment 53
Not much change (much the same result, it was a mistake, I meant to turn off the bird-training)

Questions I need to answer:
Does PCEN help
Does a more complex head help
Does a high scoring Cmap with a no-call function do well on the LB?  Otherwise I need to consider BCE 
Why does pre-training favour LWRAP over CMAP
After all that, build a cleaner dataset, & have an awesome model.

### Experiment 54
Complex Head, no pre-training.   
Seems to stall around AP=0.3.  So there's probably the answer, the complex head is an issue.  Nothing to do with pre-training.   So I could conclude from this that pre-training might as well be done on just 21 & 22 data to avoid over-fitting.

### Experiment 55
Re-running the larger head experiment, after realising it's been missing the ReLu step all this time
Running on Windows
Stopped after 3 epochs as it was stagnating around AP=0.19
Re-trying with LR=10-3, Decay 10-4
Stopped at epoch 2, now stalled on AP=.29

Conclusion, the larger head just isn't working for this problem.

### Experiment 56
Same as 55 but with the small head.     Submitted best cmap weights to Kaggle, infer version 29
Scored 78 on LB    These weights are good enough for a detector to build a new dataset.

### Experiment 57
Same as 54 but PCEN 
Scored 76 on LB

### Experiment 58
Same but bce loss

### Experiment 59
Ran on BirdClef21 dataset
CMAP peaked at 72.6,  AP at 71.2

### Experiment 60
This is a repeat of 59 with less augmentation, added a factor of 2 and a parameter AUG, to control
Still using BCE loss.   CMAP reached 0.738, AP .718     Good enough for my planned detector

### Experiment 61
Same as above but for the 23 dataset.  Goal is to make a good classifier just for the purpose of setting up the clean dataset.  Haven't run this because NVIDIA drivers on my G7 died.

### Experiment62
Creating a new detection file using weights from experiment58

### Experiment63
Creating a new detection file using weights from experiment 61

Experiment64
Re-ran Experent 61, but put the loss reduction back to nothing from reduction='sum', which was possibly causing inf values to appear in the predictions.

Experiment 65
Re-ran the detection algorithm, to make a new detections csv.

Experiment 66
Trying to figure out why the dataset now has 14 rare birds instead of just 5
re-ran the dataset creation

Experiment 67
Re-running the dataset generation, but with generated from newer weights from  64, and changed the keep-all threshold to 12

Experiment 68
Re-training with the expt67 dataset, less patience, keeping the last 14 checkpoints

### Exp69  New Dataset, V3, no primary bird score

### Exp 70
Multiclass training with the new dataset, no bias towards birds of interest
LB77

### Exp 71 -   LB:78
New Dataset, including score for primary bird

### Exp 72 - LB:78
Use dataset from 71, plus increase background noise snr to 0 to 2db, unfreeze backbone after 4

Training looks better now, it's progressing more steadily over the epochs, rather than shooting to a suspiciously good score in the first 3.

### EXP 73
Kept changes from 72, but also increased spec-augment and colored noise
Spec Augment:  3 lines (from 2), max % 10 (from 5)
Colored Noise (brown, pink, gaussian)  was 2-5db, changed to 1-2

When I've got a decent augmentation recipe, re-run on the 21 dataset and try again to pre-train.
Also re-try using a larger classifier head (actually do that first in case it works, then keep first layer)

### Exp74
Sampled then sample with 2-8 extras, plus dived by 2, produced 27826 samples  Included primary bird for scoring

EXP75
Used dataset from exp74,  crashed at epoch 5, but AP was .88, compared with .91 for exp72 at the same stage, and .905 for exp73.  So it is definately training more slowly.

### Exp76
Trying different ratios
exp69     48305        0-6, no dividing, included primary fird  LB77
exp72:     26876        0-6 no dividing, include primary bird    LB 78
exp73      27089       2- 20 / 2 produced inlcude primary   LB 77
75            27256                4-20  // 2
				27303                5-20 // 2
				49205                5-20//1    Going to try training with this.   no primary bird
				29443                5-20//1.5
				29208                4-12//1.5   -  So it's the // factor   Note - This is  similar to using 2-8//1
91			39500                5-20//1.2
			
/kaggle/input/birdclef23-extras-8-sec-ogg/train_audio/XC363503_15.ogg

## Exp77:  LB77
Trying that last one   Min5, Max20 //1,   49k extra samples

### Exp78: LB77
Same as 77, but with mixup alpha = 0.5

### Exp79: LB78
Produce a new dataset with 4-12//1.5  Using primary bird.  Using same settings as before

### EXP80
On the Kaggle platform tried effnetV2 and also tf_efficientnet_b0_ns   with the larger head, got similar results to before, stalling at low values.

### Exp81: LB76
tf_efficientnet_b0_ns  with the normal head on Kaggle

## Exp82: LB78
Switch to effnetV2 instead of V2- 21k, trained on PC, accidentally stopped after 8 epochs, when it wasn't quite finished, but anyway scored 78.  I suspect this was a small win

### Exp84
Pre-training the best model on 21-22 data.  Timed-out after 4 epochs, so i re-loaded it and continued.

### Exp85: LB78
Implemented Label Smoothing and re-ran the classifier on 4-12//1.5, with V2s, no pre-training

### Exp86:  LB77
Label smoothing, with dataset from exp71, used pre-training, and kept fine-tuning after 3 epochs.
That it cas gone backwards compared with exp72 suggests the pre-training didn't help.  Could be the label smoothing, except that that seems OK according to exp85.  Also a remote chance that the problem comes from setting AutoLR find to true.

### Exp87: LB77
Same as exp86, but with the 4-12//1.5 dataset.  Something clearly wen't wrong.  Re-running without any backbone fine-tuning

### Exp 88: LB74, 77, 77
Repeat exp 86, but no backbone fine-tuning

### Exp 89: LB77
Repeated exp87 but no backbone fine-tuning, windows machine

### Exp 90: LB78
LB78 after 18 epochs
14 epochs scored LB78 also.
5-12//1.5   with no smoothing,  pre-train, no backbone tuning   Not 100% sure there was no 

### Exp 91
Linux machine, no label smoothing, pre-train with no backbone fine-tuning  
New dataset 5-12 // 1.2

**Hypothesis:** That the main issue is over-training.   
Tested by re-submitting exp88 with two earlier epochs, scores improved to LB77 in both cases 
Also re-submit exp 90 epoch 14 to test this theory, got same score, LB78, with 4 less epochs.

It also appears that label smoothing either isn't helping or hurts the score.  I should stop doing that. 

Final Plan:  Pre-train with 21-22, no label-smoothing, no backbone tuning, ensemble best epochs 30,832 samples from first and last 8 seconds, plus extras from:

- No extras  (see if I can get back to LB79, like exp18!), Kaggle Platform
- 1-2//1  (24,834 extras) With primary bird, windows   - scoring now, notebook V79
- 2-3//1 (25,767) exp94, windows
- 4-12//1.5 (29,200 extras) With Primary bird, current best: epoch 18 -  lb78 (exp90)
- 5-20//1.2 (39,500 extras) No primary bird, Linux
- 2-3//1 (21,000) completely random, also changed the sampling to [3,-2] to avoid first 9 secs

### Exp92: LB77
Windows, training 1-2//1    reached cmap5=92.5

### Exp93 
Running on Kaggle, no-extras, reached cmap5=89.5

Exp94
Exp94 2-3//1 Ran on windows,  reached cmap5=92


## Summary 
Basically none of this made any difference.  The fundamental problem wasn't my image classifier, that was > 10% better than the one I started with.  It was something to do with the way I was sampling from the training data, and application of the classifier on the soundscapes.