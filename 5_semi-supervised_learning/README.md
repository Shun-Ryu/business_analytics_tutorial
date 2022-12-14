# ๐คMixMatch๋ฅผ ๊ตฌํํ  ๋ ์๊ฐํด๋ด์ผ ํ๋ ๊ฒ๋ค

## The things to consider for implementing MixMatch Algorithm

![image-20221226195436600](./attachments/image-20221226195436600.png)



๐ฅ์ด๋ฒ Tutorial์์๋ **Semi-Supervised Learning Method** ์ค, **Holisticํ ์ ๊ทผ ๋ฐฉ๋ฒ์ธ MixMatch**๋ฅผ ๊ตฌํํด๋ณด๋ฉด์, **๊ตฌํ ์์ ๊ณ ๋ คํด์ผ ํ  ์ฌํญ๋ค**์ ๋ํด์ ์๊ฐํด ๋ณด๋ ์๊ฐ์ ๊ฐ์ผ๋ ค๊ณ  ํ๋ค. Github์ ์๋ ์ฌ๋ฌ๊ฐ์ง ๊ตฌํ์ฒด๋ค์ ํ์ธํ์๋๋ฐ, ๋ผ๋ฌธ์๋ ๋๋ฌ๋์ง ์์ ์ฌํญ๋ค์ด ๋ณด์ด๊ธฐ์ ์ด Tutorial์์ ์ฌ๋ฌ๊ฐ์ง ํ์คํธ์ ํจ๊ป In-Depthํ๊ฒ ์์๋ณด๊ณ ์ ํ๋ค.

์ด๋ ์ฌ์ค MixMatch ๋ฟ๋ง ์๋๋ผ, Original MixMatch (from Google) ์๊ณ ๋ฆฌ์ฆ์์ ํ์๋์ด ๋์จ **FixMatch**์์๋ ๋์ผํ ๊ตฌํ์ ๊ณ ๋ คํด์ผ ํ๋ ์ฌํญ์ด๋ฏ๋ก, ์ด๋ฒ Tutorial์ ํตํด ์ ๋ฐฐ์๋ณด๋ ์๊ฐ์ ๊ฐ๋๋ก ํ์.

๋ผ๋ฌธ์์๋ ์ ๋ค๋ฃจ์ง ์๊ณ , ๊ตฌํ์ฒด์์๋ ์ ์ค๋ช์ด ์๋ ์๋์ 2๊ฐ์ง ์ฃผ์ ๋ก Tutorial์ ์ ๊ฐํ๋ ค ํ๋ค. ๋๋จธ์ง ๋ถ๋ถ์ ๋๋ถ๋ถ ๊น๋ค๋กญ์ง ์๋ค.

- MixMatch์์ **EMA(Exponential Moving Average)๋ก Teacher๋ชจ๋ธ**์ ๋ง๋๋ ๊ฒ์ด ์ค์ํ๊ฐ? 
- MixMatch์์ **Interleaving** ๊ตฌํ์ ์ค์ํ๊ฐ?



# Table of Contents

- [Background of MixMatch](#Background-of-MixMatch)

  - [1. Data Augmentation](#1-Data-Augmentation)
  - [2. Label Guessing and Label Sharpening](#2-Label-Guessing-and-Label-Sharpening)
  - [3. MixUp](#3-MixUp)

- [Tutorial. Deep Understanding of MixMatch Implementation](#Tutorial-Deep-Understanding-of-MixMatch-Implementation)

  - [1. Tutorial Notebook](#1-Tutorial-Notebook)
  - [2. Setting](#2-Setting)
  - [3. Implementation of MixMatch](#3-Implementation-of-MixMatch)
  - [4. Result (Accuracy)](#4-Result_Accuracy)

- [Final Insights](#Final-Insights)

- [Conclusion](#Conclusion)

- [References](#References)

  

-------

# Background of MixMatch

## 1. Basic Concept

MixMatch๋ ๊ธฐ์กด์ Consistency Regularization๊ณผ Pseudo-Labeling๊ณผ ๊ฐ์ ๊ธฐ๋ฒ์์ ๋ฒ์ด๋์, ๊ธฐ์กด์ ๋ฐฉ๋ฒ๋ก ๋ค์ ์ฌ๋ฌ๊ฐ๋ฅผ ๊ฒฐํฉํ์ฌ Holistic(์ ์ฒด๋ก ์ )์ธ Approach๋ก ์ ๊ทผํ ์ต์ด์ ๋ผ๋ฌธ์ด๋ค. (from Google Research). ์ง๊ธ์ ์ ๋ชจ๋ฅผ์๋ ์์ผ๋, ์ฒ์ ์ด ๋ฐฉ์์ด ๋์์ ๋, ๊ธฐ์กด์ ๋ฐฉ๋ฒ๋ก ๋ค์ ์๋ํ๋ Performance๋ก ๋ง์ ์ด๋ค์๊ฒ ์ ์ ํ ์ถฉ๊ฒฉ์ ์ ์ฌํ์๋ ๋ผ๋ฌธ์ด๋ค. 



**MixMatch์์ ์ฐจ์ฉํ๊ณ  ์๋ ์ ์ฒด์ ์ธ ๋ฐฉ๋ฒ๋ก ๋ค์ ์๋์ ๊ฐ๋ค.**

1. Data Augmentation 
2. Label Guessing & Label Sharpening
3. MixUp



์์ ๊ฒ๋ค์ ๋ชจ๋ ๊ณผ๊ฑฐ์ ๋ผ๋ฌธ๋ค์์ ๋ง์ด ๋ณด์์๋ ๊ธฐ๋ฒ๋ค์ธ๋ฐ, ์ด๋ฌํ **์ฌ๋ฌ๊ธฐ๋ฒ๋ค์ ์กฐํฉํ์ฌ ์ ์ฒด์ ์ธ Architecture ๊ตฌ์กฐ ํํ๋ก ์ข์ ์ฑ๋ฅ**์ ๊ฐ์ ธ๊ฐ๋ค๋ ๊ฒ์ด, MixMatch ๋ฅผ ํฌํจํ ํ์ Match์๋ฆฌ์ฆ ๊ธฐ๋ฒ๋ค์ ํน์ง์ด๋ผ๊ณ  ๋ณผ ์ ์๋ค. ํนํ Holisticํ๋ค๊ณ  ํ์ง๋ง, MixMatch ๊ฐ์ ๊ฒฝ์ฐ ํนํ๋ ๊ตฌํ๊ณผ ๊ตฌ์กฐ๊ฐ ๋งค์ฐ ๋จ์ํ๋ฉด์๋ ์ข์ ์ฑ๋ฅ์ ๊ฐ์ ธ์๊ธฐ ๋๋ฌธ์ ์ ๊ตฌ์์ ์ธ ๋ผ๋ฌธ ์ด๋ผ๊ณ  ๋ณผ ์ ์๊ฒ ๋ค.

์ด์ค ๊ฐ์ฅ ์ค์ํ ๊ฒ์ MixUp์ด๋ฉฐ ์ด๋ฅผ Shuffle๊ตฌ์กฐ๋ฅผ ๊ฐ์ ธ๊ฐ๋ฉฐ Data์ Label์ Matchingํด ์ฃผ์๊ธฐ ๋๋ฌธ์ MixMatch๋ผ๊ณ  ๋ถ๋ฅด์ง ์๋ ์ถ๋ค.







> ๐ฅ์ด์  ์๊ณ ๋ฆฌ์ฆ์ ์์(Sequence)๋๋ก ์์๋ณด์..!



## 1) Data Augmentation

๋จผ์  ๋ชจ๋  Data์ Data Augmentation์ ์ํํ๋ค. ์ผ๋ฐ์ ์ผ๋ก Computer Vision Deep Learning Model๋ค์ด ๊ทธ๋ ๋ฏ, **Regularization๋ก์จ Augmentation์ ์ฌ์ฉํ์ฌ ์ผ๋ฐํ ์ฑ๋ฅ์ ํฅ์**์ํจ๋ค. ํนํ ๋์ค์ ์์ MixUp์์ ๋ ๋ค์ํ ์กฐํฉ์ผ๋ก Labeled Data์ Unlabeled Data๋ฅผ ์์ด์ ํ์ตํ  ์ ์๋๋ก ๋ง๋ค์ด ์ค๋ค.

![image-20221226232414467](./attachments/image-20221226232414467.png)

์์ ๊ทธ๋ฆผ์ฒ๋ผ Labeled Data๋ Batch๋ณ๋ก 1ํ Random Augmentation์, Unlabeled Data๋ Batch๋ณ๋ก Kํ Random Augmentation์ ์งํํ๋ค. ์ด๋ ๋ผ๋ฌธ์์ ์งํํ Augmentation์ ์๋์ ๊ฐ์ด **Random horizontal flips์ Random Crops**์ด๋ค.



> ![image-20221226232556152](./attachments/image-20221226232556152.png)



๋ํ ๋ผ๋ฌธ์์๋ Unlabeled Data์ ๋ํด์๋ 2ํ์ Augmentation์ ์งํํ๋๋ก Hyper-Parameter๋ฅผ ์ธํํ์๋ค.

>  ![image-20221227010258137](./attachments/image-20221227010258137.png)





## 2) Label Guessing and Label Sharpening

์ด ๋ฐฉ์์ Pseudo Labeling๊ณผ ๋์ผํ ๋ฐฉ์์ด๋ฉฐ, **Only Unlabeled Data์ ๋ํด์๋ง Label Guessing**์ ์ํํ๋ค. ๋ํ ๋ง์ฐฌ๊ฐ์ง๋ก Guessing๋ **Unlabeled Data์ Label์ ๋ํด์๋ง Label Sharpening**์ ์งํํ๋ค. ์ ์ฒด์ ์ธ Flow๋ ์๋์ ๊ฐ๋ค. 

>  ๊ตฌํ ์์ Augmented๋ Label Data์ Guessing์ ํ๋ ๊ฒ์ด ์๋์ ๊ผญ ์ ์ ํด์ผ ํ๋ค.



![image-20221226200950638](./attachments/image-20221226200950638.png)

์์ ๊ฐ์ด Batch๋ณ๋ก ์๋์ฐจ๋ฅผ K๊ฐ์ Random Augmentaton ํํ, Guessing๋ Label๋ค์ Averageํ๊ณ , ๊ทธ ๊ฐ์ Sharpeningํ๋ค. Sharpening์ด๋ผ๋ ๊ฒ์ ํ๋ฅ ์ด ๋์ ๊ฒ์ ์ข ๋ ๊ฐ์กฐ(Temperature๋ผ๋ Hyper-Parameter T๋ฅผ ์ฌ์ฉํ์ฌ, ์ผ๋ง๋ ๊ฐ์กฐํ ์ง ์กฐ์ ํ๋ค.) ์ด Label Sharpenig์ ํตํด Unlabeled Data์ Pseudo-Label์ ๋ํ Entropy๊ฐ Minimization๋๋ค. (์ฆ, ํ๋์ Guessing Label์ ๋ ๊ฐ์กฐํ๋ค๋ ์ด์ผ๊ธฐ์. ์ ์ฒด Guessing Label์ด Unifromํํ๋ฅผ ๋๋ค๋ฉด, Entropy๊ฐ Maximization์ด ๋๋ค.) 

> ํนํ Entropy Minimization์ 2005๋ Semi-supervised learning by entropy minimization (Yves Grandvalet and Yoshua Bengio) ๋ผ๋ฌธ์ ๊ด์ฐฐ์ ํตํด Idea๋ฅผ ์ป์๋ค๊ณ  ์ ์๋ค์ ์ด์ผ๊ธฐ ํ๋ค. ์ด๋ ๊ทธ๋ฆฌ๊ณ  High-Density Region Assumption์ ๋ ๊ฐ์กฐํ๊ธฐ ์ํจ์ด๋ค.

![image-20221226234523055](./attachments/image-20221226234523055.png)

![image-20221226233648148](./attachments/image-20221226233648148.png)

์์ ๊ฐ์ด ์ผ๋ฐ์ ์ธ Softmax์ ํ๋ฅ ๊ฐ์ 1/T ์น์ ํ์ฌ ๊ฐ์ ๊ฒฐ์ ํ๊ฒ ๋๋ค. T๊ฐ์ด 2.0, 0.5, 0.1์ ๋ฐ๋ผ์, ๊ฐ์ด ๋ ์์ ์๋ก Winnner-Take-All์ ํ๊ฒ ๋๋ค. ๋ผ๋ฌธ์์๋ ์๋์ ๊ฐ์ด Hyper-Parameter T๊ฐ์ ์๋์ ๊ฐ์ด 0.5๋ก ์ธํ ํ์์ผ๋ฉฐ, ์ฐ๋ฆฌ์ ๊ตฌํ์์๋ ๋ง์ฐฌ๊ฐ์ง๋ก 0.5๋ก ์ธํ ํ  ์๊ฐ์ด๋ค.

> ![image-20221226233938750](./attachments/image-20221226233938750.png)



## 3) MixUp

> MixMatch์ ์ด๋ฆ์ ์ Mix๊ฐ ๋ค์ด๊ฐ๋์ง์ ๋ํ ์ด์ ์ด๋ค. ๊ทธ๋งํผ ์ค์ํ ๋ถ๋ถ์ด๋ค.

MixUp(2018, mixup: BEYOND EMPIRICAL RISK MINIMIZATION)์ ๋์จ ๊ธฐ๋ฒ์ผ๋ก์จ, ์๋๋ ์ข ๋ Decision Boundary๋ฅผ Smoothํ๊ฒ ํ์ฌ Emperical Risk Minimization์ Supervised Learning์์ ํ์ฉํ๊ธฐ ์ํด ์ ์๋ ๋จ์ํ ๊ธฐ๋ฒ์ด๋ค. ๊ทธ๋ฌ๋ 2019๋ Interpolation consistency training for semi-supervised learning(ICT)์์ ์ฒ์์ผ๋ก ์ด ๋ฐฉ๋ฒ์ Semi-Supervised Learning์ ์ ์ฉ์ ํ์๋ค. **MixUp์ ํตํด์ ์ข ๋ Smoothํ Boundary๋ฅผ ๊ฐ์ ํ๋ Regularization**์ ์ํํ  ์ ์๋ค. 



>  Unlabeld Data์ ๋ํด์๋ง MixUp์ ์งํํ ๊ธฐ์กด์ ICT ๋ฐฉ์๊ณผ ๋ค๋ฅด๊ฒ, MixMatch๋ Labeled Data์ Unlabeled Data๋ชจ๋์ MixUp์ ์ํํ๋ค. 



์ผ์ข์ Data Agumentation๊ณผ ์ ์ฌํ๋ฐ, ์์ 1)์์ Augmentationํด์ ์๊ฒจ๋ Data์ ํ๋ฒ๋ MixUp์ด๋ผ๋ Augmentation์ ์งํํ๊ณ , ๊ทธ X Data์ ๋ํ ์์ธก๊ณผ Augmented๋ Target Y๊ฐ ์ฌ์ด์ ์ค์ฐจ๋ฅผ ์ค์ด๋ ๋ฐฉ์์ผ๋ก ํ์ตํ๋ค. ์์ธํ๊ฒ๋ ์๋์ ๊ทธ๋ฆผ๊ณผ ๊ฐ๋ค.

![image-20221227010035007](./attachments/image-20221227010035007.png)





- **1) Augmented Set์ ์ค๋นํ๋ค.**
  - ์์์ ์งํํ Labeld Set์ Data์ 1ํ Augmentation
  - Unlabeld Set Data์ Kํ Augmentation
- **2) Shuffle Set์ ์ค๋นํ๋ค.**
  - Augmentation Set์ Copyํ์ฌ ๋ณต์ฌํ๊ณ , Labeled Set๊ณผ UnLabeled Set์ ๋ชจ๋ ํฉ์ณ์ ์๋๋ค. ์ด๋ Data X์ Target Y๋ ๊ฐ๊ฐ ์์ด ์ค๋ค.
- **3) MixUp์ ์งํํ๋ค.**
  - 1)๊ณผ 2)์์ ์ค๋นํ 2๊ฐ์ Set์ Beta ๋ถํฌ์์ ๋ฝ์ Lambda๊ฐ์ ํตํด์ Weighted Sum์ ํตํด Mixupํ๋ค. ์ด๋ Lambda๋ 0.5~1์ฌ์ด์ ๊ฐ์ผ๋ก ์ฃผ๋๋ก ํ์ฌ, Shuffle Set๋ณด๋ค Augmented Set์ ๋ Weight๋ฅผ ์ฃผ๋๋ก ํ๋ค. ๊ทธ ์ด์ ๋ ๊ฐ๊ฐ์ Target_L(Labeled)๊ณผ Target_U(Unlabeled)์ ๋ํ์ฌ, Matching๋๋๋ก ๊ฐ์ ์ฃผ๊ธฐ ์ํจ์ด๋ค. (**์๋ง ์ด ๋๋ฌธ์ MixMatch๋ผ๊ณ  ๋ถ๋ฅด๋๊ฒ ์๋๊ฐ ์ถ๋ค.**) ๊ทธ๋ ์ง ์๋๋ค๋ฉด Shuffle Set์ Bias๋๊ธฐ ๋๋ฌธ์ ์ Matching๋๋๋ก Lambda๋ฅผ ์กฐ์ ํ๋ค.
  - ๊ทธ๋ฆฌ๊ณ  MixUp๋ Data_L(Labeled)๊ณผ Data_U(Unlabeled)์ ๋ํด์๋ Model์ ๋ฃ๊ณ , ๊ฐ๊ฐ์ ์์ธก ๊ฐ์ธ Pred_L(Labeled)๊ณผ Pred_U(Unlabeled)๋ฅผ ๋ฝ๋๋ค.
  - ๊ทธ๋ฆฌ๊ณ  Pred_L๊ณผ Target_L์ฌ์ด์ Loss๋ฅผ Cross Entropy๋ฅผ ํตํด Supervised Loss๋ก ์ฌ์ฉํ๊ณ , Pred_U์ Target_U์ฌ์ด์ ๊ฐ์ Distance Metric(ex. MSE Loss)์ ์ฌ์ฉํด Unsupervised Loss๋ฅผ ๊ตฌํ๋ค.
  - ๊ทธ๋ฆฌ๊ณ  2๊ฐ์ Loss๋ฅผ ๋ฏธ๋ฆฌ์ ํด์ง Hyper-Parameter๊ฐ Weight๋ฅผ ํตํด ์กฐ์ ํ๋ค. (๊ทธ๋ฆฌ๊ณ  ์ด ๊ฐ์ Learning Rate์ฒ๋ผ, Ramp-Up์ ํตํด ์กฐ์ ํ๋๋ก ํ๋ค.)





----

# Tutorial. Deep Understanding of MixMatch Implementation

์ด๋ฒ ํํ ๋ฆฌ์ผ์์๋ ์ ์ฒด์ ์ธ MixMatch๋ฅผ Scratch๋ก ๊ตฌํํด ๋ณด๋ฉด์ ์๊ณ ๋ฆฌ์ฆ์ ์ดํดํด ๋ณด๋ คํ๋ค. ํนํ ์ ์ ์๋ค์ด ๋ผ๋ฌธ์์ ์ ๋๋ก ๊ฑด๋๋ฆฌ์ง ์์์ผ๋(ํน์ ๋ฐ๋๋ก ์ด์ผ๊ธฐํ์ผ๋!), ์์ ์๋ค์ ๊ตฌํ์ฒด ํน์ ๊ธฐํ ๋ค๋ฅธ ๊ตฌํ์ฒด๋ค์์ ์ด๋ฏธ ๊ตฌํํ๊ณ  ์์์ผ๋ ์ ๋๋ก ๋ผ์๊ฐ ์ด๋ฃจ์ด์ง์ง ์์ ์์ญ์ ๋ํ ์ดํด๋ฅผ ์คํ๊ณผ ํจ๊ป ๊ฐ์ ธ๊ฐ๋ ค ํ๋ค. ์ ๊ธฐํ๊ณ  ์ถ์ ์๋ฌธ์ 2๊ฐ์ง์ด๋ค.

- MixMatch์์ **EMA(Exponential Moving Average)**๋ก Teacher๋ชจ๋ธ์ ๋ง๋๋ ๊ฒ์ด ์ค์ํ๊ฐ? 
- MixMatch์์ **Interleaving** ๊ตฌํ์ ์ค์ํ๊ฐ?

>  ๋์์ ํด๋น ๋ผ๋ฌธ์ ๋ํด์ Reviewer๋ค์ด ์ด์ผ๊ธฐํ ๋ด์ฉ, ๊ทธ๋ฆฌ๊ณ  ๊ตฌํ์ฒด๋ค์ Issue์ ์ ๊ธฐํ ์๋ฌธ๋ค๋ ํจ๊ป ๋ณด๋ฉด์ ์ด์ผ๊ธฐ ํด ๋ณด๋ ค ํ๋ค.



## 1. Tutorial Notebook 

### ๐ฅ[Go to the tutorial notebook](https://github.com/Shun-Ryu/business_analytics_tutorial/blob/main/5_semi-supervised_learning/Tutorials/Tutorial_MixMatch.ipynb)



## 2. Setting

### Datasets

๋ฐ์ดํฐ๋ ์ ๋ชํ CIFAR-10์ ์ฌ์ฉํ๋๋ก ํ๋ค. 10๊ฐ์ Class๋ฅผ ๊ฐ๊ณ  ์๋ 32x32x3 Shape์ Imageset์ด๋ค.

![image-20221227100541139](./attachments/image-20221227100541139.png)

|      | Datasets                  | Description                               | Num Instances                                   | Image Size | Num Channels | Single Output Classes |
| ---- | ------------------------- | ----------------------------------------- | ----------------------------------------------- | ---------- | ------------ | --------------------- |
| 1    | CIFAR-10 (Classification) | 10๊ฐ์ Class๋ฅผ ๊ฐ์ง ์์ ์ด๋ฏธ์ง ๋ฐ์ดํฐ ์ | Training Set : 50,000<br />Testing Set : 10,000 | 32 x 32    | 3            | 10                    |

๋ฐ์ดํฐ์์ ์๋์ ๊ฐ์ ์ฝ๋๋ก ๋ถ๋ฌ์ค๊ฒ ๋๋ค. 

```python
# Load CIFAR-10 Labeled Images with Random Augmentation
class get_cifar10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, path_cifar10, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(get_cifar10_labeled, self).__init__(path_cifar10, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalize(self.data))

    def __getitem__(self, index):
        data_x, target_y = self.data[index], self.targets[index]

        if self.transform is not None:
            data_x = self.transform(data_x)

        if self.target_transform is not None:
            target_y = self.target_transform(target_y)

        return data_x, target_y
    

# Load CIFAR-10 Unlabeled Images with Random Augmentation (K=2)
class get_cifar10_unlabeled(get_cifar10_labeled):

    def __init__(self, path_cifar10, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(get_cifar10_unlabeled, self).__init__(path_cifar10, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])

```

์ด๋ Dataset์ Pytorch์ Dataloader๋ฅผ ํตํด Training Time์ ๋ถ๋ฌ์ง๊ฒ ๋๋ฉฐ, Dataset์ ๋์์ ์๋์ ์ฝ๋๋ก Augmentation์ด ์ด๋ฃจ์ด ์ง๋ค. Labeled Data๋ 1๋ฒ์ Random Augmentation์ด ์งํ๋๋ฉฐ, Unlabeled Data๋ K=2๋ก์จ 2๋ฒ์ Data Augmentation์ด ์ํ๋๋ค. ๋ค์์ ๊ธฐ๋ฅ์ด ์๋์ ์ฝ๋๋ก ์ด๋ฃจ์ด ์ง๋ค.

- Image Normalization (Standard Scaling)
- Image Transpose (from CIFAR-10 shape to Pytorch Shape)
- Augmentation Method #1 : Image Padding & Cropping
- Augmentation Method #2 : Image Horizontal Flipping

```python
# Dataset Normalization (Standard Scaling)
def normalize(x, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

# Transpose Image Shape for Pytorch shape from CIFAR-10 Original shape
def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

# Padding to image borders for cropping image
def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

# Augmentation Method 1 : Random Padding and Cropping 
class RandomPadandCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x

# Augmentation Method 2 : Random Horizontal Flipping 
class RandomFlip(object):
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

class ToTensor(object):
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x

def split_dataset(labels, num_train):
    labels = np.array(labels)
    index_train_labeled = []
    index_train_unlabeled = []
    index_valid = []

    num_labels = len(np.unique(labels))
    num_labeled_per_class = int(num_train / num_labels)

    for i in range(num_labels):
        index = np.where(labels == i)[0]
        np.random.shuffle(index)
        
        index_train_labeled.extend(index[:num_labeled_per_class])
        index_train_unlabeled.extend(index[num_labeled_per_class:-500])
        index_valid.extend(index[-500:])

    np.random.shuffle(index_train_labeled)
    np.random.shuffle(index_train_unlabeled)
    np.random.shuffle(index_valid)

    return index_train_labeled, index_train_unlabeled, index_valid

# Twice Augmentation for Unlabeld Image (eg. K=2)
class Multi_Augmentation:
    def __init__(self, transform_method):
        self.transform_method = transform_method
        # self.num_transform = num_transform

    def __call__(self, inp):
        aug_out_1 = self.transform_method(inp)
        aug_out_2 = self.transform_method(inp)
        return aug_out_1, aug_out_2
```



### Algorithms

์ฐ๋ฆฌ๋ MixMatch 1๊ฐ์ง๋ง ์ง์ค์ ์ผ๋ก ํ ๋ณด๊ณ ์ ํ๋ค. ํนํ ์์์ ์ ๊ธฐํ 2๊ฐ์ง, EMA ๋ฐฉ์์ Teacher Network์ Interleave์ ๋ํ ์ดํด๋ฅผ ํ๋ณด๊ฒ ๋ค.

| Algorithm | Target         | Description                                                  |
| --------- | -------------- | ------------------------------------------------------------ |
| MixMatch  | Classification | WideResNet์ Backbone์ผ๋ก ์ฌ์ฉํ Holistic Semi-Supervised Learning ์๊ณ ๋ฆฌ์ฆ์ ์ฌ์ฉ. |



## 3. Implementation of MixMatch

MixMatch๋ ์์์ ์ค๋ชํ์๋, Data Augmentation, Label Guessing & Sharpening ๊ทธ๋ฆฌ๊ณ  ๋ง์ง๋ง์ผ๋ก MixUp์ ๊ฒฐํฉํ ๋ฐฉ์์ผ๋ก ๊ตฌํ๋๋ค.  ์์ธํ๊ฒ ๊ทธ ๊ตฌํ์ฒด์ ๋ํด์ ์์๋ณด๋๋ก ํ์.



### 3-1. Loss, EMA, Interleaving Functions

MixMatch์ Learning๊ธฐ๋ฅ์ ๊ตฌ์ฑํ๊ธฐ ์ํ ํจ์์ ์งํฉ์ด๋ค. ํนํ๋ Training์ ์ํ ํจ์๋ค์ด๋ฉฐ, ์ฐจ๊ทผ์ฐจ๊ทผ ์์ธํ ์์๋ณด์



> Semi-Supervised Loss / Ramp-Up Function

์๋๋ Semi-Supervised Loss๋ฅผ ๊ตฌํ๊ธฐ ์ํ Each_Loss Class์ ๊ทธ 2๊ฐ์ supervised loss, unsupervised loss์ weighted sum์ ๊ตฌํ๊ธฐ ์ํ ramp_up ํจ์๋ฅผ ์ ์ํ๋ค. ramp_up์ ์ด epoch์์ ๋ฐ๋ผ ๊ตฌ์ฑ์ด ๋๋ฉฐ, ์ ํ์ ์ผ๋ก weight๊ฐ ์ฆ๊ฐ๋๋๋ก ๊ตฌํ์ด ๋์ด์๋ค. each_loss๋ labeled data์ target๊ฐ๊ณผ prediction๊ฐ์ ํตํด supervised loss(loss_l)๋ฅผ ๊ตฌํ๊ณ , unlabeled data์ target๊ฐ๊ณผ prediction๊ฐ์ ํตํด unsupervised loss(loss_u)๋ฅผ ๊ตฌํ๋ค. 

```python
# Ramp-up Function for balancing the weight between supervised loss and unsupervised loss
# - loss_total = loss_l + weight * loss_u
def ramp_up(current, rampup_length=epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
    
class Each_Loss(object):
    def __call__(self, pred_l, target_l, pred_u, target_u, epoch):
        probs_u = torch.softmax(pred_u, dim=1)

        loss_l = -torch.mean(torch.sum(F.log_softmax(pred_l, dim=1) * target_l, dim=1))
        loss_u = torch.mean((probs_u - target_u)**2)

        return loss_l, loss_u, lambda_u * ramp_up(epoch)

```



> ๐ฅ Exponential Moving Average for training Teacher Model Function

Fire Emoticon์ ๋ถ์๋ค. ๊ทธ๋งํผ ์ค์ํ๋ค๋ ๋ป. Exponential Moving Average(EMA)๋ฅผ ํตํ์ฌ Student Model์ Parameter๋ฅผ Teacher Model์ Parameter๋ก ์ ์ดํ๋ ํจ์์ด๋ค. ๋ฐ์์ ์คํ์ ํตํด ์์๋ณด๊ฒ ์ง๋ง, ์ด EMA๊ตฌํ์ด ๋์ด์ง์ง ์์ผ๋ฉด ๋ชจ๋ธ์ ์ ๋๋ก ํ์ต์ด ์ด๋ฃจ์ด ์ง์ง ์๋๋ค. (๋ผ๋ฌธ์์๋ ์คํ๋ ค ๋ฐ๋๋๋ ์ค๋ช์ ํ๊ณ ์๋ค.) ์ฑ๋ฅ ํฅ์์ ํ์์ ์ธ ํจ์์ด๋ฉฐ, ๊ทธ์ ๋ํ ์ค๋ช์ด ์์ผ๋ฏ๋ก ๊ผญ ๊ตฌํ์ ํด์ผ๋ง ํ๋ค. Student Model๋ง์ผ๋ก๋ Prediction์ฑ๋ฅ์ด ์ ๋์ค์ง ์๋๋ค.

```python
class exponential_moving_average(object):
    def __init__(self, student_model, teacher_model, alpha=0.999):
        self.model = student_model
        self.ema_model = teacher_model
        self.alpha = alpha
        self.student_params = list(student_model.state_dict().values())
        self.teacher_params = list(teacher_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.student_params, self.teacher_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for student_param, teacher_param in zip(self.student_params, self.teacher_params):
            if teacher_param.dtype==torch.float32:
                teacher_param.mul_(self.alpha)
                teacher_param.add_(student_param * one_minus_alpha)
                # customized weight decay
                student_param.mul_(1 - self.wd)
```



> ๐ฅ Interleaving Function

์ญ์๋ Fire Emoticon์ ๋ถ์๋ค. ์๋์ Interleaving์ ๊ตฌํ์ ๋ผ๋ฌธ ์ ์ ์์ ๊ตฌํ์ฒด์๋ ์กด์ฌํ๋ฉฐ, ๊ทธ ์ธ์ ๋๋ถ๋ถ์ ์ธ๊ธฐ์๋ ๊ตฌํ์ฒด์์๋ ์๋์ ๋ฐฉ์์ผ๋ก Interleaving์ ์ฌ์ฉํ๋ค. Interleaving์ Labeled Data์ Unlabeled Data์ ๊ฐ๋ค์ ์๋ก Mixingํด์ฃผ์ด, Model์ ๊ณ์ฐํ๊ธฐ ์ํด ์กด์ฌํ๋ค. ์๋ํ๋ฉด, ๋๋ถ๋ถ์ ๊ตฌํ์ฒด์ ์์ด Labeled Data๋ฅผ Model์ ์๋ ฅํ์ฌ Supervised Loss๋ฅผ ๊ตฌํ๊ณ , ๊ทธ ์ดํ์ Unlabeled Data๋ฅผ Model์ ์๋ ฅํ์ฌ Unsupervised Loss๋ฅผ ๊ตฌํ๊ธฐ ๋๋ฌธ์ ๋ฌธ์ ๊ฐ ๋ฐ์ํ๋ค. ์ด๋ ๊ฒ ๊ฐ๊ฐ์ Data๋ฅผ Model์ ๋ฐ๋ก, 2๋ฒ ํ์์ ๊ณ์ฐํ  ๊ฒฝ์ฐ, Batch Normalization์ ์งํ ํ  ๋์ ๋๊ฐ์ ๋ถํฌ๊ฐ ์ ์ฒด์ Batch์ ๋ถํฌ๋ฅผ ๋๋ณํ์ง ์๊ธฐ ๋๋ฌธ์, ํ์ต์์ Bias๊ฐ ์๊ธฐ๊ฒ ๋๋ค.

๋ฐ๋ผ์ ์ด๋ฒ Tutorial์์๋ Interleaving์ ์ฌ์ฉํ ๊ฒฝ์ฐ์, ์ฌ์ฉํ์ง ์๋ ๊ฒฝ์ฐ์ ํ์ต ๋ฐฉ๋ฒ์ ๋ํด์ ๋๊ฐ์ง ๋ชจ๋ ๊ตฌํ์ ์งํํ์๋ค.(์ฝ๋์ Training๋ถ๋ถ์ ํ์ธํ์ธ์..!) ๊ทธ๋ฆฌ๊ณ  ์๋์ Test์์ Interleaving์ ๊ณ ๋ คํ์ง ์๋ ํ์ต ๋ฐฉ๋ฒ์ ์์ด์, ํ์ต์ด ์ ๋์ง ์๋ ๋ค๋ ๊ฒ์ ์คํ์ ํตํด ๋ฐํ๋ด๋๋ก ํ๊ฒ ๋ค.

```python
def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]
```





### 3-2. Training Functions

Training Function์ MixMatch์ ์์์ ๋ฐ๋ผ ๊ตฌํ๋์ด ์๋ค. ์ฌ์ค ์ด๋ ค์ด ์ฝ๋๋ ์๋๋ฉฐ, ์ง๊ด์ ์ผ๋ก ์ดํด๊ฐ ๋๋ฏ๋ก ๊ฐ๋จํ๊ฒ ์์๋๋ก ํ๋ฒ ์ดํด๋ณด๋๋ก ํ์.



> 1. Data Augmentation

Data Augmentation์ ์งํํ๋ ์ฝ๋์ด๋ค. ์์์ ์ ์ํ๋ CIFAR-10์ ๋ถ๋ฌ์ค๋ฉด์ Augmentation์ ํ๋๋ก Pytorch์ DataLoader๋ฅผ ๋ง๋ค์๊ณ , ๊ทธ์ ๋ฐ๋ผ์ ๋ฐ์ดํฐ๋ฅผ ๋ถ๋ฌ์ค๊ฒ ๋๋ค. 

- Labeled Data : inputs_l / target_l
- Unlabeld Data : inputs_u, inputs_u2 (์ฆ, K=2)

์ฌ๊ธฐ์ try, except๊ฐ ์๋๋ฐ ์ด๋ ๋ณดํต supervised์์๋ training์์ batch์ ๋ง์ถฐ์ data๋ฅผ loadํ์ง๋ง, MixMatch๊ฐ์ ๊ฒฝ์ฐ Labeled Data์ Unlabeled Data๊ฐ ์๋ก ๊ฐ์๊ฐ ๋ค๋ฅด๊ธฐ ๋๋ฌธ์(๋น์จ์ ๋ฐ๋ผ ๋ค๋ฅด์ง๋ง Unlabeled๊ฐ ๋ ๋ง๊ฑฐ๋ ๊ฐ์), Batch์ ๊ด๊ณ์์ด training_iteractionํ์์ ๋ฐ๋ผ Data๋ฅผ loadingํ๊ธฐ ๋๋ฌธ์ ๋ชจ๋  Data Loader์์ sampling์ ๋ค ์ํํ์ ๊ฒฝ์ฐ, ๋ค์๊ธ Data Loader๋ฅผ ์ํ Iterator๋ฅผ ๋ง๋ค๊ธฐ ์ํด except ์ฝ๋๊ฐ ์กด์ฌํ๋ค. ํฌ๊ฒ ์ด๋ ค์ธ ๊ฒ ์๋ ์ฝ๋์ด๋ค.

```python
#########################################
# 1. Data Augmentation
#########################################
try:
    inputs_l, targets_l = next(iter_train_labeled)
except:
    iter_train_labeled = iter(labeled_trainloader)
    inputs_l, targets_l = next(iter_train_labeled)


try:
    (inputs_u, inputs_u2), _ = next(iter_train_unlabeled)
except:
    iter_train_unlabeled = iter(unlabeled_trainloader)
    (inputs_u, inputs_u2), _ = next(iter_train_unlabeled)

# Transform label to one-hot
batch_size = inputs_l.size(0)
targets_l = torch.zeros(batch_size, 10).scatter_(1, targets_l.view(-1,1).long(), 1)

if use_cuda:
    inputs_l, targets_l = inputs_l.cuda(), targets_l.cuda(non_blocking=True)
    inputs_u = inputs_u.cuda()
    inputs_u2 = inputs_u2.cuda()
```



> 2. Label Guessing and Label Sharpening

Pseudo Labeling์ ์ํ์ฌ Unlabeled Data๋ฅผ ์์ธกํ๋ ๋จ๊ณ์ด๋ค. ์์ธก์ ํ๊ณ  ๋์ Temperature Hyper-Parameter๋ฅผ ํตํด label๊ฐ์ ํ๋ฅ ์ sharpeningํ๋ค. ์ด๊ฒ๋ ์ด๋ ค์ธ ๊ฒ ์๋ ์ฝ๋์ด๋ค.

```python
#########################################
# 2. Label Guessing & Label Sharpening 
#########################################
with torch.no_grad():
    # compute guessed labels of unlabel samples
    outputs_u = model(inputs_u)
    outputs_u2 = model(inputs_u2)
    p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
    pt = p**(1/Temperature)
    targets_u = pt / pt.sum(dim=1, keepdim=True)
    targets_u = targets_u.detach()
```



> 3. MixUp

๋งค์ฐ ์ค์ํ ๋ถ๋ถ์ด๋ค. MixMatch์ ํต์ฌ์ด ๋๋ ์ฝ๋๋ผ๊ณ  ๋ณผ ์ ์๋ค. (๊ทธ๋ฌ๋ ๊ตฌํ์ด ์ด๋ ต์ง ์๋ค.). ์๋์ 2๊ฐ์ Data๋ฅผ MixUpํด ์ค๋ค.

- ์๋์ Augmented๋ Labeled Data์ Unlabeled Data
- ์ถ๊ฐ์ ์ผ๋ก Labeled Data+Unlabeld Data๋ฅผ ๋ถ์ฌ์ Shuffleํ Data

๊ทธ ๋์ ๋๊ฐ์ Data๋ฅผ Weighted Sum์ ํด ์ฃผ๋๋ฐ, ์๋์ Augmented Data์ ์ข ๋ ๊ฐ์ค ์น๋ฅผ ๋ ์ฃผ์ด, ๋ชฉํํ๋ Target๊ฐ๊ณผ Matching๋๋๋ก ๊ตฌ์ฑ์ ํ๋ค. ๊ทธ๋์ ์๊ณ ๋ฆฌ์ฆ ์ด๋ฆ์ด MixMatch์ด์ง ์์๊น ์ถ๋ค.

```python
#########################################
# 3. MixUp 
#########################################
all_inputs = torch.cat([inputs_l, inputs_u, inputs_u2], dim=0)
all_targets = torch.cat([targets_l, targets_u, targets_u], dim=0)

Lambda_ = np.random.beta(alpha, alpha)

Lambda_ = max(Lambda_, 1-Lambda_)

idx = torch.randperm(all_inputs.size(0))

input_a, input_b = all_inputs, all_inputs[idx]
target_a, target_b = all_targets, all_targets[idx]

mixed_input = Lambda_ * input_a + (1 - Lambda_) * input_b
mixed_target = Lambda_ * target_a + (1 - Lambda_) * target_b
```



> 4. Interleaving or No-Interleaving

์ด๊ฒ ๋ผ๋ฌธ์์๋ ๊ทธ๋ ๊ณ  ์ธํฐ๋ท์๋ ๊ทธ๋ ๊ณ  ์ ์ค๋ช์ด ๋์ด์์ง ์๋ ๋ถ๋ถ์ด๋ค. ๋๋ถ๋ถ์ github์ ๊ตฌํ์ฒด์์ interleaving๋ฐฉ์์ ์ฌ์ฉํ๋๋ฐ, ์ด๋ mixed๋ labeled data์ unlabeled data๋ฅผ ์๋ก interleaving(๋ฐ์ดํฐ์ sample์ ์ผ๋ถ๋ถ์ ์๋ก ๊ตํ)ํ๋๋ก ํ์ฌ, batch๋ณ๋ก batch-normalization์ ํ ๋ ๊ทธ ๋ถํฌ๊ฐ ๋ณํ๋์ง ์๋๋ก ํ๋ ๊ธฐ๋ฅ์ ๊ตฌํํด ๋์ ๊ฒ์ด๋ค. ์ batch-normalizationํ  ๋ ๋ถํฌ๊ฐ ์ํ์ต๋์ง ์๋ํ๋ฉด, ์๋์ ๊ตฌํ์์ ์ฒ๋ผ labeled data๋ฅผ model์ ๋ฐ๋ก ํ๋ฆฌ๊ณ , unlabeled data๋ฅผ model์ ๋ฐ๋ก ํ๋ฆฌ๊ธฐ ๋๋ฌธ์ด๋ค.

๐ฅ์ด ๋๋ฌธ์ ๊ฐ๊ฐ์ model์ด batch-normalization์ parameter๊ฐ ๊ฐ๋ณ์ ์ผ๋ก bias๋๋ฉฐ ํ์ต์ด ์ด๋ฃจ์ด์ง ๊ธฐ ๋๋ฌธ์ด๋ค.

๐ฅ๋ฐ๋ผ์ MixMatch๋ฟ๋ง ์๋๋ผ, FixMatch๊ฐ์ ๊ฒฝ์ฐ๋ ๋ง์ฐฌ๊ฐ์ง์ธ๋ฐ, ์ด๋ ๊ฒ Labeled Data์ Unlabeled Data๋ฅผ Model์ ๋ฐ๋ก ํ๋ฆด๊ฒฝ์ฐ..๊ทธ๋ฆฌ๊ณ  Backbone Model์์ batch-normalization์ ์ฌ์ฉํ  ๊ฒฝ์ฐ๋ ๊ผญ Interleaving function์ ๊ตฌํํ์ฌ ์ฌ์ฉํด์ผ ํ๋ค. ๊ทธ๋ ์ง ์์ ๊ฒฝ์ฐ ํ์ต ์์ฒด๊ฐ ์ ์ด๋ฃจ์ด์ง์ง ์๋๋ค!

๐ฅ๋ํ Google Reseasrch์ Fix Match์ ์๋ค์ด github issues์ ๋ต๋ณํ๊ธฐ๋ฅผ, Multiple-GPUs๋ฅผ ์ฌ์ฉํ  ๊ฒฝ์ฐ, Interleaving์ ํตํด ๋ฐ์ดํฐ๋ฅผ ์์ด์ค๋ค์ ๊ฐ GPU๋ก ํ๋ ธ์๋ ์ญ์๋ Batch-Norm์ด ์ ํ์ต๋๊ธฐ ๋๋ฌธ์ ์ด๋ ๊ฒ ๊ตฌํํ์๋ค๊ณ ํ๋ค. ๊ทธ๋ฌ๋ ์ด๋ Tensorflow์ ๊ฒฝ์ฐ์ด๊ณ  Pytorch๋ Multi-GPUs๋ฅผ ์ํ Batch-Norm ๊ตฌํ์ด ๋ฐ๋ก ์์ด๋ฏ๋ก, ์ด๋ ๊ฒ ํ  ํ์๋ ์๋ค๊ณ  ์๊ฐ๋๋ค.

๐ฅ๊ทธ๋ฌ๋ Labeled Data์ Unlabeled Data๋ฅผ ํฉ์ณ์, ํ๋ฒ์ Model์ ํ๋ฆด ๊ฒฝ์ฐ๋ ์ด์ผ๊ธฐ๊ฐ ๋ฌ๋ผ์ง๋ค. ์ด ๋์๋ Interleaving์ด ํ์๊ฐ ์์ผ๋ฉฐ, ๋น์ฐํ ํ๋ฒ์ Batch์ 2๊ฐ์ ๋ฐ์ดํฐ ํํ๊ฐ ๋์์ ๋ค์ด๊ฐ๋ฏ๋ก, ๋ถํฌ์ ๋ณํ๊ฐ ์ ์ ํ ์ ํ์ต์ด ๋๋ค. 

๐ฅ๊ทธ๋ฐ๋ฐ ์ฌ๊ธฐ์์ ์ฃผ์ํด์ผ ํ  ๊ตฌํ ์ฌํญ์ด ์๋๋ฐ, interleaving์ ํ  ๊ฒฝ์ฐ Batch-Size๊ฐ 1:2๋ก ๋๋๋ฉด์ ํ์ต์ด ๋์ง๋ง, No-interleaving์ผ ๊ฒฝ์ฐ Batch-Size๊ฐ 3๋ฐฐ๊ฐ ๋๊ธฐ ๋๋ฌธ์ ๊ธฐ์กด์ Learning Rate๋ก๋ ํ์ต์ด ๋๋ ค์ง๊ฒ ๋๋ค. ์ด๋ฅผ ์ํด์ ๋ง์ด๋ค ์ฌ์ฉํ๋ Batch Size๊ฐ k๋ฐฐ๊ฐ ๋๋ฉด Learning Rate๋ sqrt(k)๋ฐฐ๋ก ์ฆ๊ฐ์ํค๋ ๋ฐฉ์์ผ๋ก ํ์ต์ ์งํํ๋ฉด ํ์ต์ด ์ ๋๋ ๊ฒ์ ํ์ธํ  ์ ์์๋ค.

```python
########################################
# 4. Interleaving or No-Interleaving
########################################
if use_interleaving:
    # 1) interleave labeled and unlabed images between batches to get correct batchnorm calculation 
    mixed_input = list(torch.split(mixed_input, batch_size))
    mixed_input = interleave(mixed_input, batch_size)

    # 2) labeled prediction
    logits = [model(mixed_input[0])]
    
    # 3) unlabeled prediction
    for input in mixed_input[1:]:
        logits.append(model(input))

    # 4) de_interleave to calculate labeled supervised loss and unlabeld unsupervised loss properly 
    logits = interleave(logits, batch_size)

    logits_l = logits[0]
    logits_u = torch.cat(logits[1:], dim=0)
else:
    # No Interleaving and calculate both labeled and unlabeled sample.
    # The model is used only once to predict logits. So A calculation of batchnorm is proper.
    # But if you want to use this no-interleaving method then you should adjust the learning rate (with multiply sqrt(k))
    # k means k-times of increased batch size
    logits = model(mixed_input)
    split_logits = list(torch.split(logits, batch_size))
    logits_l = split_logits[0]
    logits_u = torch.cat(split_logits[1:], dim=0)
```



> 5. Semi-Supervised Loss

๊ฐ๋จํ ๊ตฌํํ๋ Semi-Supervised Loss์ด๋ค. ๊ฐ๋จํ๋ค. 2๊ฐ์ Loss๋ฅผ ๊ฐ๊ฐ ๊ตฌํด์ weighted sum์ ์ํํ๋ค.

```python
########################################
# 5. Semi-Supervised Loss 
########################################
loss_l, loss_u, weight = each_loss(logits_l, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_index/train_iteration)

loss_total = loss_l + weight * loss_u

# record loss
losses_total.update(loss_total.item(), inputs_l.size(0))
losses_l.update(loss_l.item(), inputs_l.size(0))
losses_u.update(loss_u.item(), inputs_l.size(0))
weights.update(weight, inputs_l.size(0))

```



> 6. Backpropagation

๋น์ฐํ backpropagation. Backpropagation์ Student Model์๋ง ํ์ต์ด ์งํ๋๋ค. Teacher๋ชจ๋ธ์ ์๋์ EMA Learning์ผ๋ก ์งํ๋๋ค.

```python
########################################
# 6. Backpropagation 
########################################
optimizer.zero_grad()
loss_total.backward()
optimizer.step()

```



> 7. EMA Learning for Teacher Model

EMA์ ํตํ Teacher Model์ ํ์ต์ด๋ค. ๊ฒฐ๊ตญ์๋ ์ฐ๋ฆฌ๊ฐ ์ฌ์ฉํ  ๋ชจ๋ธ์ Teacher๋ชจ๋ธ์ด๋ฉฐ(Student Model์ ํ์ต์ด ์ ์ด๋ฃจ์ด์ง์ง ์๊ณ , ์ฑ๋ฅ์ด ์ ๋์ค์ง ์๋๋ค.), ์ ์๋ค์ EMA๊ฐ ์คํ๋ ค ๋ชจ๋ธ ์ฑ๋ฅ์ ์์ฒ(hurt)๋ฅผ ์ค๋ค๊ณ  ํ์์ผ๋, EMA๋ฅผ ํตํ Teacher Model์ด ๊ฒฐ๋ก ์ ์ผ๋ก ํจ์ฌ ํ์ต๋ ์๋๊ณ  ์ฑ๋ฅ๋ ์ ๋์จ๋ค. ์ด๋ ์ฌ๋ฌ ๊ตฌํ์ฒด์์ EMA๋ฅผ ๊ธฐ๋ธ์ผ๋ก ๊ฐ์ ธ๊ฐ๊ณ  ์๊ณ , ์ธํฐ๋ท ์ปค๋ฎค๋ํฐ ์์์๋ ๋ง์ ์ด๋ค์ด EMA๋ฅผ ํตํด ์ฑ๋ฅ์ ๋ฝ์๋ผ ์ ์๋ค๊ณ  ๋งํ๊ณ  ์๋ค.

```python
########################################
# 7. EMA Learning for Teacher Model 
########################################
if is_ema is True:
    ema_optimizer.step()
```







## 4. Result_Accuracy

- ์ธก์  ๋จ์ : Accuracy
- Unsupervised Augmentation ํ์ K=2 (๋ผ๋ฌธ์์ ์ ์ํ ์์น)
- Accuracy๋ Testset์ ๋ํด์ ๊ณ์ฐ



### 4-1) Original ๋ชจ๋ธ๊ณผ์ ๋น๊ต

| #Labels   | 250          | 2000         |
| --------- | ------------ | ------------ |
| Paper     | 88.92 ยฑ 0.87 | 92.97 ยฑ 0.15 |
| This code | 86.76        | 91.57        |

์์ ํ๋ฅผ ๋ณด๋ฉด ์ ๋ฐ์ ์ผ๋ก ๊ตฌํ์ด ์ ๋์์์ ํ์ธํ  ์ ์๋ค. Seed๊ฐ์ด๋ Hyper-Parameter์ ๋ฐ๋ผ์ ๊ฒฐ๊ณผ ์ฑ๋ฅ์ด ์กฐ๊ธ์ฉ ๋ฌ๋ผ์ง ์ ์๋ค๊ณ  ์๊ฐํ๋ค. ๋ํ **์๊ฐ ๊ด๊ณ์ ํ์ต์ ๋งค์ฐ ์ค๋ ์๊ฐ ๋๋ฆฌ์ง ๋ชปํ์๊ณ , ์์ ํ ์๋ ดํ์ง ์์ ์ํ์์ ํ์ต์ ์กฐ๊ธฐ ์ข๋ฃ ํ๋ค๋ ๊ฒ**์ ๊ฐ์ํด์ผ ํ๊ฒ ๋ค.





### 4-2) Test Result


|      | #Labeles | Interleave             | EMA  | Accuracy                    |
| ---- | -------- | ---------------------- | ---- | --------------------------- |
| 1    | 250      | -                      | -    | 63.00 (์ดํ ์ฑ๋ฅ ๊ณ์ ํ๋ฝ) |
| 2    | 250      | -                      | O    | 61.87 (์ดํ ์ฑ๋ฅ ๊ณ์ ํ๋ฝ) |
| 3    | 250      | O                      | -    | 80.31                       |
| 4    | 250      | O                      | O    | **86.76**                   |
| 5    | 250      | O (No-Interleave Mode) | O    | 84.17                       |
| 6    | 2000     | O                      | -    | 88.76                       |
| 7    | 2000     | O                      | O    | **91.57**                   |

Test๊ฒฐ๊ณผ๋ ์์ ๊ฐ๋ค. Label๋ ๋ฐ์ดํฐ๋ฅผ ๋ช๊ฐ๋ ์ผ๋์ง์ ๋ฐ๋ผ์ ๋ถ๋ฅํ์์ผ๋ฉฐ, Interleave๋ฅผ ์ฌ์ฉํ์ ๋์ ํ์ง ์์์๋, ๊ทธ๋ฆฌ๊ณ  EMA๋ก Teacher Model์ ํ์ตํ์ฌ Accuracy๋ฅผ ๊ณ์ฐ ํ์ ๋์ ํ์ง ์์์ ๋๋ฅผ ๋น๊ตํ์๋ค. **Label์ด ๋ง์ ์๋ก ์ฑ๋ฅ์ ํฅ์ ํจ๊ณผ๊ฐ ์ปธ์ผ๋ฉฐ, Interleave๊ฐ ์๋ค๋ฉด ์์ ํ์ต์ด ์๋๋ค๊ณ  ๋ณผ ์ ์๋ค. ๋ํ EMA๋ฅผ ํตํ Teacher Model์ ์ฑ๋ฅ์ด EMA ์๋ Student Model๋ณด๋ค ๋ ์ข์ ์ฑ๋ฅ์ ๋ฐํํจ**์ ์ ์ ์์๋ค. 

๋ํ ์๋กญ๊ฒ ๊ตฌํํ **No-Interleave Mode**๋ ์ญ์ ํ์ต์ด ์ ๋๋ค๋ ์ ์ ์ ์ ์์๊ณ , Interleave๋ Batch-Norm์ ๊ณ์ฐ๋ง ์ ๊ณ ๋ คํ๋ค๋ฉด, ํด๋น ํจ์์ ๊ตฌํ ์์ด ๋ค๋ฅธ ๋ฐฉ์์ผ๋ก๋ ํ์ต์ ์ ์ด๋ฃจ์ด์ง๋๋ก ๋ง๋ค์ ์๋ค๋ ๊ฒ์ ์๊ฒ ๋์๋ค. (No-Interleave Mode๊ฐ ์ฑ๋ฅ์ด ์ข ๋ ๋ฎ์๋ณด์ด์ง๋ง, ๊ธฐ๋ณธ์ ์ผ๋ก **์๊ฐ๊ด๊ณ์ ์๋ฒฝํ ์๋ ดํ์ง ์์ ์ํ์์ ํ์ต์ ์กฐ๊ธฐ ์ข๋ฃ**ํ์ฌ ๊ทธ๋ ๋ค๊ณ  ๋ณด๋ฉด ๋๊ฒ ๋ค. ๋ํ Semi-Supervised Loss์ Weighted Sum์ ์ํ Lambda์ Learning Rate์ Hyper-Parameter๋ฅผ ์ต์ ํํ์ฌ ๋ ์ข์ ๊ฒฐ๊ณผ๋ฅผ ๋ผ ์ ์์ ๊ฑฐ๋ผ ์๊ฐํ๋ค.)





----


# Final Insights

์ฐ๋ฆฌ๋ ์ด๋ฒ Tutorial์ ํตํ์ฌ, MixMatch๋ฅผ ๊ตฌํํ  ๋์, ์๋ ๋ผ๋ฌธ์์ ์ธ๊ธํ์ง ์๊ฑฐ๋ ์ ๋๋ก ์ค๋ชํ์ง ์๋ ์์ญ์ธ EMA์ Interleave์ ํจ๊ณผ์ ๋ํด์ ํ์ธํด ๋ณด์๋ค. Test๋ฅผ ํตํด ์ ๋ฐ์ ์ผ๋ก ์ป์ Insight๋ ์๋์ ๊ฐ๋ค.

- **Labels๊ฐ์ ๋ณํ์ ๋ฐ๋ฅธ ํจ๊ณผ**
  - 250๊ฐ๋ณด๋ค ํ์คํ 2000๊ฐ๋ก Label๋ ๋ฐ์ดํฐ๋ฅผ ๋๋ ธ์ ๋ ์ฑ๋ฅ ํฅ์์ด ํฌ๊ฒ ์ผ์ด๋ฌ์ผ๋ฉฐ ์๋ ด๋ ๊ต์ฅํ ๋น ๋ฅด๊ฒ ์งํ๋์๋ค. Label์ ์ต๋ํ ๋ง์ ํ๋ณด๊ฐ Semi-Supervised์๋ ์ฑ๋ฅ์ ๋์ผ์ ์์์ ํ์ธํ  ์ ์์๋ค.
  - ํ์ต๋๋ Loss๋ฅผ ๊ด์ฐฐํ์์ ๋, Supervised Loss๊ฐ ์ฒ์์๋ ๋น ๋ฅด๊ฒ ๋จ์ด์ง๋ค๊ฐ ๋์ค์๋ Unsupervised Loss๋ฅผ ๋จ์ด๋จ๋ฆฌ๋ ์ชฝ์ผ๋ก ์ด๋ํ์๋ค. ์ด๋ Ramp-Up์ ํจ๊ณผ๋ผ๊ณ  ์๊ฐ๋๋ฉฐ, ์ด๋ฅผ ํตํด ์ ์ฒด์ ์ธ Training Loss๊ฐ ๋ง์ด ๋จ์ด์ง์ง๋ ์์๋, Unsupervised Loss๋๋ฌธ์ Test Accuracy๊ฐ ์ง์์ ์ผ๋ก ์ฆ๊ฐํ๊ฒ ๋์๋ค.
- **EMA์ ํจ๊ณผ**
  - EMA๊ฐ ์์ด์ผ ํ์คํ ์ํ๋ ์์ค๊น์ง ์ฑ๋ฅ์ด ๋์ฌ ์ ์์์ ํ์ธํ  ์ ์์๋ค.
  - EMA๊ฐ ์์ด ๊ธฐ๋ณธ Student Model๋ก๋ ์ด๋์ ๋ ์์ธก์ด ๊ฐ๋ฅํ์ง๋ง, EMAํตํ Teacher Model์ด ๋ ์ฑ๋ฅ์ด ๋์์ ์ ์ ์์๋ค.
  - ํนํ Loss์ ๋ณํ๋ฅผ ๋ณด์์๋, EMA๋ฅผ ์ฌ์ฉํ Teacher Model์ Smoothํ๊ฒ ์์ ์ ์ธ ํ์ต์ด ๋๊ณ , ๋ํ Test Set์ ๋ํ์ฌ Ensembleํจ๊ณผ๋ฅผ ํตํ์ฌ ๋ ์ข์ Generalization ์ฑ๋ฅ์ ๋ณด์์ ์ ์ ์์๋ค.
  - ์ 1์ ์๋ ํด๋น EMA๋ฅผ Weight Deacy๋ฅผ ๋์ฒดํ๊ธฐ ์ํด ์ฌ์ฉํ๊ณ  ์์๋ค. ์ผ์ข์ Regularization์ ์งํํ๋ค๊ณ  ๋ณด๋ ๊ฒ ๊ฐ๋ค. ๋๋ ๊ฐ์ธ์ ์ผ๋ก Ensemble์ ํจ๊ณผ๋ ์๋ค๊ณ  ์๊ฐ์ด ๋ ๋ค.
- **Interleave์ ํจ๊ณผ**
  - Interleave๋ฅผ ์์ ํ์ง ์๊ณ , ๊ฐ๋ณ Label๊ณผ Unlabel Data์ ์์ธก์ ํ๋ค๋ฉด, ์ฑ๋ฅ์ด ์ง์์ ์ผ๋ก ํ๋ฝํ๊ฒ ๋๋ค. ์์์ ์ด์ผ๊ธฐ ํ๋ฏ, Batch-Norm์ด ์ ๋๋ก ๊ณ์ฐ๋์ง ์๊ธฐ ๋๋ฌธ์ด๋ผ๊ณ  ์๊ฐ๋๋ค.
  - Interleave๋ฅผ ์ฌ์ฉํ  ๊ฒฝ์ฐ, ์ ๋๋ก ํ์ต์ด ๋จ์ ์ ์ ์์๋ค. ํนํ๋ ์ผ๋ฐ์ ์ธ Interleave ์ฌ์ฉํ์ ๋์, No-Interleave Mode๋ฅผ ์ฌ์ฉํ์ ๋ ๊ฑฐ์ ์ ์ฌํ ๊ฒฐ๊ณผ๋ฅผ ์ป์์ ์์์ ์ ์ ์์๋ค. ๋ฌผ๋ก  ์ ๋๋ก ํ์ต๋๋๋ก ํ๊ธฐ ์ํด Learing Rate์ ์กฐ์ ์ ํ์ํ๋ค. (Batch Size๊ฐ ๋ณํํ๋ฏ๋ก)



๊ทธ๋ฆฌ๊ณ  ์ถ๊ฐ์ ์ผ๋ก ์ธํฐ๋ท์ ํตํ์ฌ ๊ด๋ จ๋ ์ ๋ณด๋ค์ ์์งํ์ฌ EMA์ Interleave์ ๋ํ์ฌ ์๋์ ๊ฐ์ด ์ ๋ฆฌ ํด ๋ณด์๋ค.



> EMA Case

์ ๋ผ๋ฌธ์ ์ ์๋ค์ ์๋์ ๊ฐ์ด EMA Parameter์ ์ฌ์ฉ์ MixMatch์ ์์ข์ ์ํฅ์ ์ฃผ๋ ๊ฒ๊ฐ๋ค๊ณ  ํ์๋๋ฐ ์ค์ ์ ์ผ๋ก EMA๋ ์ฑ๋ฅ์ ํฅ์์ํค๋ ํจ๊ณผ๊ฐ ์์์ผ๋ฉฐ, ์ ๋ผ๋ฌธ ์ ์๋ค์ ๊ตฌํ์ฒด ๋ฟ๋ง ์๋๋ผ ๋ค๋ฅธ ๊ตฌํ์ฒด๋ค๋ EMA๊ฐ Optional์ด ์๋๋ผ ํ์์ ์ผ๋ก ๋ค ๋ฃ์ ๊ฒ์ ๋ณด๋ฉด EMA๋ ์ฑ๋ฅ ํฅ์์ ๋์์ด ๋๋ค๋ ๊ฒ์ ์ ์ ์๋ค.

![image-20221227130933075](./attachments/image-20221227130933075.png)

๋ํ MixMatch์ Neurips Review์์ ์๋์ ๊ฐ์ Comment๊ฐ ์๋ค. ([Reviews: MixMatch: A Holistic Approach to Semi-Supervised Learning (neurips.cc)](https://proceedings.neurips.cc/paper/2019/file/1cd138d0499a68f4bb72bee04bbec2d7-Reviews.html))

"From the reproduction by my group, we found the **EMA plays an essential role in achieving the results**. Without it, there would be a non-unneglectable gap to the showed results. Therefore, it is encouraged to include an ablation study of the EMA to show its impact on the proposed model. "

์ด๋ฒ Tutorial์์๋ ๋ง์ฐฌ๊ฐ์ง ๊ฒฐ๊ณผ์๋๋ฐ, ๋ผ๋ฌธ ์ ์๋ค์๊ฒ EMA๋ฅผ ๋ฃ์์ ๋์ ๋บ์ ๋๋ฅผ Ablation Study๋ฅผ ํ๋๊ฒ ์ข๋ค๊ณ  Review๋ฅผ ํ์์ง๋ง, ์ ์๋ค์ EMA๋ฅผ ๋ฃ์์ ๋์๋ง Test๋ฅผ ์งํํ์๋ค. ๋ผ๋ฌธ ์  1์ ์๋ EMA๋ฅผ Weight Decay๋ฅผ ๋์ฒดํ์ฌ ์ฐ๊ณ  ์๋ค๊ณ  ํ๋ค.(MixMatch์ FixMatch์์๋)

![image-20221227132326819](./attachments/image-20221227132326819.png)

๊ทธ๋ฆฌ๊ณ  ์ฌ๋ฏธ์๋ ๊ฒ์ ์๋์๊ฐ์ด, Mean Teacher ๋ชจ๋ธ(https://arxiv.org/abs/1703.01780)์์๋ ๊ทธ๋ ์ง๋ง, Student Model๋ณด๋ค Teacher๋ชจ๋ธ์ด ์ฑ๋ฅ์ด ๋ ์ข์๋ค. ํ์ง๋ง ํด๋น Mean Teacher๋ผ๋ฌธ๋ ๊ทธ๋ ๊ณ  MixMatch๋ผ๋ฌธ๋ ๊ทธ๋ ๊ณ , Prediction์์ Teacher๋ฅผ ์ฐ๋๊ฒ ์ข์์ง, Student๋ฅผ ์ฐ๋๊ฒ ์ข์์ง์ ๋ํ Guideline์ด ์๋ค๋ ๊ฒ์ด ์์ฝ๋ค.

![image-20221227133008179](./attachments/image-20221227133008179.png)

> Interleave Case

๋ํ Interleave์ ๋ํด์ MixMatch์ 1์ ์๋ Github Issues์ ์๋์ ๊ฐ์ด ๋ต๋ณํ๊ณ  ์๋ค. Interleave๋ฅผ ์ฐ๋ ๋ชฉ์ ์ Multi-GPUs๋ฅผ ์ฌ์ฉํ ๋ Batch-Norm์ ์ ๊ณ์ฐํ๊ธฐ ์ํด์๋ผ๊ณ . ํ์ง๋ง, PyTorch์ ๊ฒฝ์ฐ๋ ๋ฐ๋ก Parallel Batch-Norm์ ๊ตฌํํ๊ณ  ์๊ธฐ ๋๋ฌธ์ ์ด๋ ๊ฒ ๊ตฌํํ  ์ด์ ๋ ์์ ๊ฒ ๊ฐ๋ค. 

![image-20221227131126671](./attachments/image-20221227131126671.png)



๊ทธ๋ฆฌ๊ณ  ์ฌ๋ฐ๋ ์ฌ์ค์ ์ ์ ์๋ Multi-GPUs๋ฅผ ์ํด์๋ผ๊ณ  ๋ต๋ณํ์ผ๋, ์ค์ ์ ์ผ๋ก ๋ณด๋ฉด Single-GPU์์๋ Model์ 2ํ ๊ฐ๊ฐ Inferenceํ๊ธฐ ๋๋ฌธ์ Batch-Norm์ ๋ถํฌ๊ฐ ๊นจ์ด์ ธ์ ์ฑ๋ฅ์ด ์ ๋์ค์ง ์๋๋ค. (์ ์ ์์ ๊ตฌํ์์๋ Parallel์ ์ฌ์ง์ด ์ฐ์ง ์๊ธฐ๋ ํ๋ค;;) ์ด์ ๋ํ Github Issues์ ๋ํ ๊ธ์ด ์์ด์ ์๋์ ์งง๊ฒ ๊ณต์ ํ๋ค. (๋์ ๋์ผํ ์๊ฐ์ด๋ค.)

![image-20221227131421312](./attachments/image-20221227131421312.png)



# Conclusion

๊ฒฐ๋ก ์ ์ผ๋ก, 

- MixMatch์์ **EMA(Exponential Moving Average)๋ก Teacher๋ชจ๋ธ**์ ๋ง๋๋ ๊ฒ์ **์ค์ํ๋ค.**
  - ๋ชจ๋ธ์ ์ฑ๋ฅ์ Student Model๋ง์ผ๋ก ๋์ค์ง ์๋๋ค.
  - EMA๋ Regularization ์ญํ ์ ํตํด Generalization์ ๋ ์ํ๋๋ก ๋ง๋ ๋ค.
- MixMatch์์ **Interleaving** ๊ตฌํ์ **์ค์ํ๋ค**.
  - Semi-Supervised์์ Labeled Data์ Unlabeled Data๋ฅผ ๊ฐ๊ฐ ๋ฐ๋ก(์ฆ 2ํ), Model์ ๋๋ฆฌ๊ฒ ๋  ๊ฒฝ์ฐ Batch-Norm๊ณ์ฐ์ด ๋ถํฌ๊ฐ Bias๋์ด ํ์ต์ด ์ ์ด๋ฃจ์ด ์ง์ง ์๋๋ค. ์ด๋ฅผ ๋ง๊ธฐ ์ํด Labeled Data์ Unlabeled Data๊ฐ Data Sample๋ค์ ์์ด์ฃผ๋ Interleaving์ด ํจ๊ณผ๋ฅผ ๋ณด๊ฒ ๋๋ค.
  - ๊ทธ๋ฌ๋ Labeled์ Unlabeled Data๋ฅผ ํ๋ฒ์ ๋์์ Model์ ๋ฃ์ด ๊ณ์ฐํ๋ค๋ฉด, Interleaving์ ๊ตณ์ด ํ์์์ ์ ์๋ค.



-----

# References

-  ๊ณ ๋ ค๋ํ๊ต ์ฐ์๊ฒฝ์๊ณตํ๋ถ ๊ฐํ์ฑ ๊ต์๋ Business Analytics ๊ฐ์ ์๋ฃ
- https://hipolarbear.tistory.com/19
- https://proceedings.neurips.cc/paper/2019/file/1cd138d0499a68f4bb72bee04bbec2d7-Reviews.html
- https://www.reddit.com/r/MachineLearning/comments/jb2egk/d_consitency_training_how_do_uda_or_fixmatch/
- https://github.com/kekmodel/FixMatch-pytorch/issues/19
- https://github.com/google-research/fixmatch/issues/20
- https://github.com/kekmodel/FixMatch-pytorch/issues/36
- https://github.com/google-research/fixmatch/issues/37

