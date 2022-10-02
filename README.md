# Chapter 1. Dimensionality Reduction (Tutorial)




# Table of contents

- Overview
- 



# Overview

## Problem

1) Tabular Data에도 물론 High-Dimensional Data가 있으나, 최근에 많이 사용되는 비정형 Data인 Image, Natural Language Data들은 대부분 아주 높은 데이터 차원을 가지고 있다. 차원이 높아지게 되면 차원의 저주(Curse of dimensionality)에 빠지게 되는데, 차원이 높아질수록 Machine Learning 알고리즘의 학습이 어려워지고(Noise Data가 증가 등), 연산량도 매우 높아지는 경향이 있다. 또한 고차원 데이터에서는 모델의 Generalization 성능을 높이기 위해서 더 많은 수의 Data가 일반적으로 더 필요하게 된다.

2) 또한 Visualize는 3차원 까지로만 할 수 있기 때문에, 데이터의 Understanding과 Expression을 위하여 고차원의 데이터를 1~3차원 까지로 표현해야 하는 경우가 존재한다.

   

## Solution

이 두가지 문제(ML 알고리즘 성능의 문제 및 Visualization의 문제)를 해결하기 위해, 많은 연구원, 엔지니어들이 다양한 차원 감소 기법(Dimensionality Reduction)을 개발 해 왔다.

- 목적 : 모델에 Fitting에 가장 좋은 최고의 변수들의 subset을 찾는 것

- 고전적인 Dimensionality Reduction의 방법론 정리 (출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/gehjerbhgwawhzs/01_1_Dimensionality%20Reduction_Overview%20and%20Variable%20Selection.pdf?dl=0))![image-20221002173123682](assets/image-20221002173123682.png)



- 최근에는 딥러닝 기반 Representation Learning, Embedding 등을 활용한 기법으로 Dimensionality Reduction을 많이 수행한다.



본 Tutorial에서는 고전적인 방법에서의 Dimensionality Reduction기법을 살펴보려 한다. 그리고 Feature Selection과 Extraction 관점에서 아래의 방법들을 Tutorial로 진행해 보도록 하겠다.



1. **Supervised** - Feature Selection
   - Genetic Algorithm
2. **Unsupervised** - Feature Extraction
   1. MDS (Multidimensional scaling)
   2. LLE
   3. ISOMAP
   4. t-SNE



# Supervised - Feature Selection

## Genetic Algorithm



# Unsupervised - Feature Extraction

## MDS (Multidimensional scaling)

## LLE

## ISOMAP

## t-SNE

