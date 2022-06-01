# EEG-ML

Currently, trying to recreate the following paper:

https://iopscience.iop.org/article/10.1088/1741-2552/ac4430

The code is pretty well written, but for my purpose the goal is validation 
and implementation of various other structures to test the maximal accuracy
currently achievable.

### Problems [[Github issue]](https://github.com/Kubasinska/MI-EEG-1D-CNN/issues/22)
In the above study the author reported accuracy above 97% which is
practically impossible. There could be problem with the data
preprocessing or the train/test split that leads to this number.
Additionally, there are other studies that seem to exhibit the
same issue:

https://iopscience.iop.org/article/10.1088/1741-2552/ac1ed0
https://iopscience.iop.org/article/10.1088/1741-2552/ab6f15

So this leads to the conclusion that the current code I'm
working with needs to be through audited and tested. 