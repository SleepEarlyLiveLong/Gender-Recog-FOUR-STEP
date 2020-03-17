# Gender-Recog-FOUR-STEP
Realization of  speech gender recognition within FOUR steps.

This project is developed to offer a simple way to implement gender recognition according to the speaker's voice. The file structure is shown as follows:

```
-src
   |-myfunctions
         |-myKMS
         |-myKNN
         |-myMLS
         |-myNaiveBayes
         |-mySphProcs
         |-myStatistics
         |-mySURF
         |-mySVM
         |-YOURcustomfts
   |-samples
         |-female
         |-male
   |-S1FeatureExtraction.m
   |-S2DatasetDivide.m
   |-S3GenderRecogTrainVal.m
   |-S4GenderRecogTest.m
-README.md
```

Files in folder "myfunctions" are factory functions called by FOUR major functions:
```
S1FeatureExtraction.m
S2DatasetDivide.m
S3GenderRecogTrainVal.m
S4GenderRecogTest.m
```
These functions realize four sub-processes of gender recognition separately, which are feature extrcation, data division, model training and model test.


All users need is to run these 4 XXX.m files by order, and then results will show up.
