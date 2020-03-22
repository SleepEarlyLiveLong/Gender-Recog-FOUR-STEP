# Gender-Recog-FOUR-STEP
Realization of  speech gender recognition within FOUR steps.

This project is developed to offer a simple way to implement gender recognition according to the speaker's voice. The file structure is shown as follows:

```
-src
   |-GUIversion0.1
         |-GenderRecogGUI.fig
         |-GenderRecogGUI.m
   |-GUIversion0.2
         |-GenderRecogGUI.fig
         |-GenderRecogGUI.m
   |-GUIversion0.3
         |-GenderRecogGUI.fig
         |-GenderRecogGUI.m
   |-libsvm322
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

You can run this project in two ways:


## 1: With GUI ##
There are THREE versions in three separate folders available for you. 

For version 0.1 in folder 'GUIversion0.1', only one classifier is implemented and the rest needs to be done by users themselves, which is a kind of after class exercise for students. Meanwhile, 4 unfinished features are set aside for students to realize without limitations.

For version 0.2 in folder 'GUIversion0.2',all 5 classifier are already implemented, however, 4 unfinished features are still set aside, waiting for your creativity.

For version 0.3 in folder 'GUIversion0.3',all 5 classifier along with 15 features are already implemented, which is the full version of this project.


## 2: Without GUI ##
All users need is to run these 4 XXX.m files by order, and then results will show up.
