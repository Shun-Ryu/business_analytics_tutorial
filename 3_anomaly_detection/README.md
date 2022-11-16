# Anomaly Detection의 한계를 알아보자.



## The limitation of anomaly detection algorithm



🔥이번 Tutorial에서는 Anomaly Detection이 실제 Practical한 상황에서 어떠한 제약점이 있는지 파악해 보고, 어떠할 때 쓰면 안되는지 확인 해 보고자 한다. 확인해보고자 하는 문제는 2가지 이다.



### 1. Anomaly Detection for "Regression To Anomaly Detection"

- Manufacturing 공정 등의 Data는 기본적으로 Target값이 Continuous한 경우가 많이 있다. 이런 경우, 특정 Threshold 이상의 값은 '이상치(Anomaly)'로, Threshold 이하의 값은 '정상치(Normal)'로 Target Data를 Binary Categorization을 하여, Anomaly Detection이나 Classification으로 문제를 풀려는 시도를 일반적으로 많이 생각한다.

- 따라서 이렇게 근본적으로 Regression Task인 것들을 Thresholding하였을 때, 과연 Anomaly Detection과 같은 알고리즘이 잘 동작하는지 테스트를 진행 해 보았다. (특히 현업의 Manufacturing 공정에서는 대부분이 '정상'데이터이며 '이상'데이터는 매우 적은 Imbalanced한 Data가 대부분의 Case이다. 그러나 이번 실험에서는 Regression을 통한 Imbalanced한 부분을 없애기 위하여 Normal Class와 Abnormal Class를 5:5 수준으로 나누었다.)

- 이를 통해 Anomaly Detection을 근본적 Regression Task에 써도 될지, 그 한계를 알아보려고 한다.

### 2. Anomaly Detection for "Classification To Anomaly Detection"

- 일반적으로 Unsupervised기반의 Anomaly Detection보다 Supervised Classification이 성능이 더 높은 경우가 많이 있다.
- 실제로 간단한 Task에서 Anomaly Detection이 Supervised Classification보다 어느정도의 성능의 차이가 있는지 알아보려 한다.
- 이를 통해 Anomaly Detection이 Supervised Classification Task에 써도 될지, 그 한계를 알아보려 한다.





# Table of Contents

- [Background of SVM](#Background-of-SVM)
  
  - [1. Basic Concept](#1-Basic-Concept)
  - [2. About SVM](#2-About-SVM)
  - [3. Linear SVM](#3-Linear-SVM)
  - [4. Kernel SVM](#4-Kernel-SVM)
  
- [Tutorial - Competion for tabular datasets](#Tutorial_Competion-for-tabular-datasets)
  
  - [1. Tutorial Notebook](#1-Tutorial-Notebook)
  - [2. Setting](#2-Setting)
  - [3. Result (Accuracy)](#3-Result_Accuracy)
  - [4. Result (Training Time)](#4-Result_Training-Time)
  - [5. Result (Inference Time)](#5-Result_Inference-Time)
  
- [Final Insights](#Final-Insights)
  
  - [1. Training Time 관점](#1-Training-Time-관점)
  - [2. Inference Time 관점](#2-Inference-Time-관점)
  - [3. Accuracy 관점](#3-Accuracy-관점)
  - [4. 그 외의 생각들](#4-그-외의-생각들)
  - [5. 결론](#5-결론)
  
  







# Background of SVM

## 1. Basic Concept

- - -




## 2. About SVM

- 

## 3. Linear SVM

- - 

## 4. Kernel SVM

- 


- 

# Tutorial_Regression_2_AnomalyDetection

위에서 우리는 SVM에 대해서 상세히 알아보았으니, 과연 SVM이 현재에도 Tabular Data에서 적절한 선택인지 비교를 해보자. 아래의 Tutorial Link를 통해 Notebook으로 각 Dataset에 따른 Algorithm의 속도와 성능을 비교할 수 있다.



## 1. Tutorial Notebook 

### 🔥[Go to the tutorial notebook](https://github.com/Shun-Ryu/business_analytics_tutorial/blob/main/2_kernel_based_learning/Tutorials/tutorial_svm_comparison.ipynb)



## 2. Setting

### Datasets

데이터셋은 아래와 같이 2개의 유명한 Tabular 형태의 Regression Dataset을 사용합니다. 

|      | Datasets                        | Description                                                  | Num Instances | Num Inputs (Xs) | Num Outputs (Ys) |
| ---- | ------------------------------- | ------------------------------------------------------------ | ------------- | --------------- | ---------------- |
| 1    | Diabetes (Regression)           | 당뇨병 환자 데이터 (1년 후 당뇨의 진행정도를 Target값으로 함) | 442           | 10              | 1                |
| 2    | Boston House Price (Regression) | Boston의 집값에 대한 Data                                    | 1797          | 64              | 10               |

데이터셋은 아래와 같은 코드로 불러오게 됩니다.

```python
if dataset_name == 'diabetes_r':
    x, y= datasets.load_diabetes(return_x_y=true)
elif dataset_name == 'boston_r':
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=none)
    x = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    y = raw_df.values[1::2, 2]
else:
    pass
```

각 Dataset은 Regression Target이므로, 각 Dataset을 Anomaly에 사용하기 위하여 사용되는 Threshold값은 아래와 같다. 각 값은 전체 데이터의 Median 값이다. Regression Task에 Imbalanced에 의한 영향을 줄이기 위해 중앙값을 사용하여 양불 Data의 Balance를 맞추었다.

- Diabetes : 140 
- Boston House Price : 21





### Algorithms

알고리즘은 아래와 Regression 알고리즘과 Anomaly Detection을 서로 비교합니다.

- Regerssion 
  - SVR을 사용하여 Regression Task에서 Regression Algorithm을 사용하고 예측한 값을 특정 Threshold로 Classification하여 양불을 판정하는데 사용합니다.
- Anomaly Detection
  - 4가지의 알고리즘(One-Class SVM, Isolation Forest, Autoencoder Anomaly Detection, Mixture Of Gaussian)을 사용하여, 데이터를 양불로 Binary Classification문제로 전처리 후, 양품 데이터만을 학습하여 Anomaly를 탐지한다.

|      | Algorithm           | Target            | Description                               |
| ---- | ------------------- | ----------------- | ----------------------------------------- |
| 1    | Linear SVR          | Regression        | 선형 SVR                                  |
| 2    | Kernel SVR          | Regression        | 선형 SVR + Kernel Trick(using rbf kernel) |
| 3    | One-Class SVM       | Anomaly Detection |                                           |
| 4    | Isolation Forest    | Anomaly Detection |                                           |
| 5    | Autoencoder AD      | Anomaly Detection |                                           |
| 6    | Mixture of Gaussian | Anomaly Detection |                                           |



## 3. Usage Code

### SVR

성능이 어느정도 검증된 기법인 SVR을 사용하여, Regression Task를 예측한다. 그리고 예측된 결과를 Threshold로 나누어, 양불을 판정한다. 아래와 같은 코드로 학습과 추론하여 Regression을 예측한다. Linear SVR과 RBF SVR을 사용하였으며, param_grid에 있는 Hyper-parameter를 Grid Search하여 모델 최적화를 진행하였다.

```python
param_grid = [
    {'kernel': ['linear'], 'C': [1.0, 2.0, 3.0, 10., 30., 100.]},
    {'kernel': ['rbf'], 'C': [1.0, 2.0, 3.0, 5.0, 10., 30., 100.],
    'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
]

elapsed_time_kernel_svr = []

svr_regressor = SVR(kernel='rbf')
# svm_classifier = svm_classifier.fit(x_train, y_train)

start_time = datetime.now()
grid_search = GridSearchCV(svr_regressor, param_grid, cv=7, scoring="neg_mean_squared_error", verbose=2)
best_svr_regressor = grid_search.fit(x_train, y_train)
elapsed_time_kernel_svr.append((datetime.now()-start_time).total_seconds())

start_time = datetime.now()
y_pred = best_svr_regressor.predict(x_test)
elapsed_time_kernel_svr.append((datetime.now()-start_time).total_seconds())

```



아래와 같이 예측한 값을 위에서 설정한 threshold값으로 양불(양품 +1, 불량 -1) Labeling을 해 준다. 이를 통해서 Answer Y값의 Classification된 값 과의 비교를 통해 Accuracy를 계산한다.

```python
y_pred_c = y_pred.copy()
y_pred_c[y_pred > threshold_anomaly] = -1
y_pred_c[y_pred <= threshold_anomaly] = 1

acc_svr_kernel = accuracy_score(y_test_c, y_pred_c)

print('Confusion Matrix\n', confusion_matrix(y_test_c, y_pred_c))
print('Best Prameters ', grid_search.best_params_)
print('Accuracy ', acc_svr_kernel)
print('Elapsed Time(train, test) ', elapsed_time_kernel_svr)
```



그 결과 다음과 같은 결과를 얻을 수 있다.

|                  | Diabetes               | Boston |
| ---------------- | ---------------------- | ------ |
| Confusion Matrix | [[34 11]<br/> [11 33]] |        |



```
Confusion Matrix
 [[34 11]
 [11 33]]
Best Prameters  {'C': 100.0, 'gamma': 0.1, 'kernel': 'rbf'}
Accuracy  0.7528089887640449
```





## 4. Result_Accuracy

- 측정 단위 : 정확도 %
- Dataset은 Testset 20%, Training 72%, Validation 8%를 기준으로 진행하였다.
- Accuracy는 Testset에 대해서만 계산하였다. (당연히!)
- 모델은 Validation 기준으로 Loss가 가장 적은 Best Model로 Testing을 진행함

|      | Algorithm                    | Diabetes     | Digits       | Iris         | Breast Cancer |
| ---- | ---------------------------- | ------------ | ------------ | ------------ | ------------- |
| 1    | Linear SVM                   | <u>81.16</u> | 97.77        | 96.66        | 98.24         |
| 2    | Kernel SVM (rbf)             | **83.11**    | **99.16**    | 96.66        | 98.24         |
| 3    | Basic ANN                    | 78.57        | <u>97.78</u> | **100**      | <u>98.25</u>  |
| 4    | TabNet (Deep Learning Model) | 79.22        | 96.94        | 76.66        | 92.98         |
| 5    | XGBoost                      | **83.11**    | 96.11        | 96.66        | 95.61         |
| 6    | LightGBM                     | 77.92        | 96.66        | 96.66        | 93.85         |
| 7    | CatBoost                     | 80.51        | 97.77        | 96.66        | 96.49         |
| 8    | Random Forest                | 75.97        | 96.66        | 96.66        | 97.36         |
| 9    | Linear RVM                   | **83.11**    | 95.55        | <u>96.99</u> | **99.12**     |
| 10   | Kernel RVM                   | **83.11**    | 95.00        | 96.66        | 97.36         |





# Tutorial_Classification_2_AnomalyDetection

위에서 우리는 SVM에 대해서 상세히 알아보았으니, 과연 SVM이 현재에도 Tabular Data에서 적절한 선택인지 비교를 해보자. 아래의 Tutorial Link를 통해 Notebook으로 각 Dataset에 따른 Algorithm의 속도와 성능을 비교할 수 있다.







# Final Insights

## 1. Training Time 관점

- **SVM은 타 방식 대비 전반적으로 장점을 갖고 있음** ✅

- SVM은 전반적으로 Training Time이 매우 우수하며, 같은 시간 대비 Boosting, Bagging, NN, RVM 계열들 대비 다양한 Hyper Parameter를 탐색할 수 있음. SVM보다 빠른 방식은 크게 차이는 나지 않으나 Diabetes Dataset에서의 LightGBM정도 밖에 존재하지 않았음 (그러나 Parallel Threading을 쓴다면 Boosting 계열이 더 빠를 수 있음)
- 물론 Training Time 등의 속도는 어떠한 언어로 된 구현체(ex. C, Rust 등)인지에 따라 다르고, Coding Trick을 썼느냐에 따라 다를 수 있음. 그리고 SVM은 매우 Optimization된 Code로 Scikit-Learn에 구현되어 있기 때문에 이러한 결과를 보였을 것 같음. 그러나 XGBoost나 LightGBM같은 경우도 속도를 높이기 위한 다양한 방식(ex. Cache Hit Optimization 등)을 사용하기 때문에, 완전 다른 알고리즘이 불리하다고 보기는 힘듬.
- 그러나 XGBoost 같은 경우 알고리즘 구현 방식 자체가 Thread Processing을 가정하기 때문에, single thread기반에서는 속도적 이득을 얻기가 힘드리라 봄
- 따라서 ARM Cortex A가 아닌, 그 외의 Real-Time Embedded System에서의 Training에서는 SVM이 타 알고리즘을 압도하는 속도적인 이익을 얻을 것이라 생각됨
- 그리고 Tabular Data에서 Hyper Parmeter Searching을 가미한 Baseline 모델을 찾아내는데 SVM이 매우 적합하리라 생각됨



## 2. Inference Time 관점

- **SVM은 타 방식 대비 Big Dataset에서는 단점을 갖고 있음** ❌
- **그러나 SVM은 타 방식 대비 Small Dataset에서는 나쁘지 않는 속도를 갖고 있음(Single Thread Embedded Real-Time System에 적합)** ✅

- Diabetes는 700개 정도의 Dataset인데, 이정도 크기 이후부터 Inference에서는 SVM이 다른 방식 대비 빠른 속도를 보이지는 않음. 물론 GPU를 사용하지 않는 NN계열보다는 빠를 수 있겠으나, Boosting, Random Forest, RVM 등에 모두 속도가 밀림

- 그러나 Small Dataset을 한번에 Inference할 때에는 장점을 갖추고 있음. 이를 보았을때 Single Thread의 Embedded System에서 Real-Time Inference나 FPGA, ASIC으로 구현했을때의 속도는 SVM이 가장 빠르리라 생각됨 (특히 Inference Time에 Single Instance를 처리하는 속도가 가장 빠르리라 예상)

- 알고리즘 특성상 Boosting계열에서 정렬 등이 필요하여 이때 Boot-up되는 속도가 SVM대비 오래 걸리는 것으로 추정됨

  



## 3. Accuracy 관점 

- **SVM이 타 방식 대비 Accuracy가 유사하거나 더 좋은 경우도 있음** ✅
- Accuracy도 SVM이 전체적으로 모든 Dataset에서 다른 알고리즘 대비 가장 좋거나(Diabetes, Digits), 2~3위 수준(Irs, Breast Cancer)의 Accuracy를 나타냄. 
- 특히나 Linear SVM (Soft Margin)이 다른 모델들 대비 크게 떨어지지 않는 Accuracy를 보여줌으로써, Kernel이 필요한 특이한 Case의 임의로 생성된 Dataset이 아닌한, 매우 빠르고 정확하게 Classification을 하는 능력을 갖추었다고 보여짐. 특히나 Dimension이 커지면서 Linear Model로 분류되는 Hyper Plane을 찾기가 더 쉽지 않을까 생각됨.
- 물론 Dataset이 크지 않고 단순하며, 전반적으로 Accuracy가 유사하게 높으므로(물론 Use-Case에 따라서 0.1% Accuracy도 매우 크다고 볼 수 있긴 하지만) 이 결과로만 가지고 SVM이 다른 알고리즘보다 확실히 뛰어나다고 볼수는 없음.
- 그러나 확실한건, **Training Time과 Inference Time대비, SVM이 다른 최신의 알고리즘보다 떨어진다고 보기 어려우며, 오히려 더 좋은 경우 있다고 말할 수 있음**.  따라서 Silver Bullet은 없으므로, SVM알고리즘을 실무에서 꼭 검증 해 볼 필요는 있음



## 4. 그 외의 생각들

- 그 외 Tabnet같은 경우 좋은 성능을 내지 못하는 경우가 있는데, Dataset에 적은 경우 특히나 그러하며(Iris, Breast Cancer), Tabnet은 category encoding에 유리한 측면이 있다고 보여, 좀 더 복잡한 column을 가진 dataset, 좀 더 큰 dataset에서 활용되면 좋을 것이라 생각됨
- 그리고 Basic ANN은 Dropout과 SELU등의 활용으로 Tabular에서 다른 알고리즘 못지않은 성능을 보일 수 있다는 것을 확인함
- 또한 RVM은 타 알고리즘 대비 Training Time이 상대적으로 매우 길었지만, Inference Time은 타 알고리즘대비 매우 빠른 편임. 또한 성능도 DIgits Dataset 외에는 타 알고리즘 대비 상대적으로  높은 성능을 보이고 있음. 
- 그러나 RVM은 Digits 데이터가 1700개 가량 밖에 안되는 경우에도, 35분이나 걸리는 아주 긴 Training시간을 갖음. 따라서 RVM은 Small Dataset에서만 꼭 고려해야할 알고리즘으로 생각됨(Hyper Parameter Tunning도 자동으로 이루어질 수 있으며, Output에 대한 Uncertainty를 구할 수 있음)
- RVM은 Gaussian Process와 유사하기 때문에 데이터가 커짐에 따라 매우 느려지고 O(N^3), Feature의 개수가 늘어날수록 Accuracy가 떨어지는 경향을 보이지 않나 생각이 듬
- RVM Training의 빠른 구현 Tipping, M., & Faul, A. (2003). Fast marginal likelihood maximization for sparse Bayesian models도 존재하므로, 해당 구현을 사용한다면 이번 Tutorial의 구현보다는 더 긍정적인 느낌을 받았으리라 생각됨
- Random Forest는 전반적으로 Training Time이나 Inference Time에서 큰 장점은 없었으며(빠른 편이 아니었음), Accuracy 성능 역시 다른 알고리즘 보다 딱히 뛰어나다고 보이지는 않음(Breaset Cancer제외 Boosting보다 전반적으로 떨어짐). 향후에는 Random Forest보다는, SVM, Boosting, RVM을 좀 더 고려하지 않을까 생각이 듬 (물론 Hyper Parameter를 좀 더 테스트 하면 다를 수는 있을 듯)
- Boosting 계열에서 비교하자면, 속도 측면에서는 LightGBM이 빠른 편이지만 Accuracy 측면에서는 XGBoost나 CatBoost가 더 좋은 성능을 보이며 속도도 크게 느리지는 않음. LightGBM보다 XGBoost나 CatBoot를 좀 더 고려하는게 좋지 않을까 생각됨



## 5. 결론

- 결론적으로는 SVM알고리즘은 지금도 쓸만한 알고리즘이라고 말할 수 있음.

