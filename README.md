* Step1: Process labellers' csv and calculate certainty score for each image. 
* Step2: Collect those which has certainty score>0.5. 
* Step3: Re-train a pre-trained VGG model on our binary-class dataset.
* To deal with imbalance data, we apply (1) resampling (2) augmentation on train set only. Then, we save a model with the best macro-f1 score. 
