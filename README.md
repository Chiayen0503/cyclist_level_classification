How to setup </br>
* Download folder ````merging_labels/```` and unzip it to current directory </br>
  [link]:(https://drive.google.com/drive/folders/1y2RzB5JiS-YpDPF43e8l8Itxjc19ECiJ?usp=sharing)
* Create a folder ````gopro_videos/```` in current directory
* Download and unzip folder ````batch_1/```` and put it in ````gopro_videos/```` </br>
  [link]: (https://drive.google.com/file/d/1YWTkk_99bELAXiCn0F1p30rSuiTWLBe5/view?usp=sharing)

How to run notebooks </br>
* Step1: Visit ````certainty.ipynb````. Process labellers' csv and calculate certainty score for each image. Next, include those who has certainty score>0.5 to raw data.  
* Step2: Visit ````train_cnn.ipynb````. Re-train a pre-trained VGG model on our binary-class dataset. To deal with imbalance data, we apply (1) resampling (2) augmentation on train set only. Then, we save a model with the best macro-f1 score of test set. 
