# Chapter 1. Dimensionality Reduction (Tutorial)




# Table of contents

- [Overview](#Overview)

- [Supervised Feature Selection](#Supervised-Feature-Selection)

  - [Genetic Algorithm](#Genetic-Algorithm)

- [Unsupervised Feature Extraction](#Unsupervised-Feature-Extraction)

  - [MDS](#MDS)
  - [ISOMAP](#ISOMAP)
  - [LLE](#LLE)
  - [t-SNE](#t-SNE)

  



# Overview

## 1) Problem

1) Tabular Data에도 물론 High-Dimensional Data가 있으나, 최근에 많이 사용되는 비정형 Data인 Image, Natural Language Data들은 대부분 아주 높은 데이터 차원을 가지고 있다. 차원이 높아지게 되면 차원의 저주(Curse of dimensionality)에 빠지게 되는데, 차원이 높아질수록 Machine Learning 알고리즘의 학습이 어려워지고(Noise Data가 증가 등), 연산량도 매우 높아지는 경향이 있다. 또한 고차원 데이터에서는 모델의 Generalization 성능을 높이기 위해서 더 많은 수의 Data가 일반적으로 더 필요하게 된다.

2) 또한 Visualize는 3차원 까지로만 할 수 있기 때문에, 데이터의 Understanding과 Expression을 위하여 고차원의 데이터를 1~3차원 까지로 표현해야 하는 경우가 존재한다.

   

## 2) Solution

이 두가지 문제(ML 알고리즘 성능의 문제 및 Visualization의 문제)를 해결하기 위해, 많은 연구원, 엔지니어들이 다양한 차원 감소 기법(Dimensionality Reduction)을 개발 해 왔다.

- 목적 : 모델에 Fitting에 가장 좋은 최고의 변수들의 subset을 찾는 것

- 고전적인 Dimensionality Reduction의 방법론 정리 (출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/gehjerbhgwawhzs/01_1_Dimensionality%20Reduction_Overview%20and%20Variable%20Selection.pdf?dl=0))![image-20221002173123682](assets/image-20221002173123682.png)



- 최근에는 딥러닝 기반 Representation Learning, Embedding 등을 활용한 기법으로 Dimensionality Reduction을 많이 수행한다.



본 Tutorial에서는 고전적인 방법에서의 Dimensionality Reduction기법을 살펴보려 한다. 그리고 Feature Selection과 Extraction 관점에서 아래의 방법들을 Tutorial로 진행해 보도록 하겠다.



1. **Supervised** - Feature Selection
   - ❌Forward, Backward, Stepwise Selection
   - ✅Genetic Algorithm
2. **Unsupervised** - Feature Extraction
   1. ❌PCA
   2. ✅MDS (Multidimensional scaling)
   3. ✅LLE
   4. ✅ISOMAP
   5. ✅t-SNE



# Supervised (Feature Selection)



## Genetic Algorithm

### Notebook Tutorial

- [Go to the tutorial]() 
- [Reference Code](https://github.com/prakhargurawa/Feature-Selection-Using-Genetic-Algorithm) 



### About Feature Selection

- Dimensionality Reduction에는 간단하게 입력의 Feature들을 줄이는 기법들을 사용할 수 있다.
- 기본적으로 변수(Feature) Selection을 위해서는, 전역 탐색(like Grid Search)와 같은 방법이 가장 좋은 결과를 가져올 수 있으나 현실적으로 너무 느린 방법론이므로, 다양한 대체하는 방법론들이 개발되어져 왔다.

- 가장 간단히는 변수들을 저진선택하거나, 후진소거하거나, 단계적 선택을 하는 등의 탐색으로 속도를 빠르게 할 수 있으나, 알고리즘의 특성상 많은 조합의 최적 탐색값은 찾을 수 없다.

- 따라서 기존의 간단한 변수 Selection 방법을 넘어서서, Random성을 사용한 자연을 모방한 Meta-Heuristic방법인 Genetic Algorithm을 사용해 효율적이며 최적의 변수를 Selection하는 기법에 대한 Tutorial을 진행 해 보고자 한다.



### Genetic Algorithm Summary

- 생명체의 생식과정을 모사한 진화 알고리즘의 한 종류
- 기본적인 유전 알고리즘은 아래의 3가지 순서에 따라 구현된다.
  - Selection : 현재 가능 해집합에서 우수한 해들을 선택, 다음 세대를 생성하기 위한 부모 세대로 지정
  - Crossover : Selection에서 선택된 부모 세대들이 서로 유전자를 교환하여 새로운 세대 생성
  - Mutation : 낮은 확률로 유전자에 변이를 발생시켜, Random성을 통해 Local Optimum에서 빠져나올 수 있는 Chance를 제공
- Genetic Algorithm 도식도  (출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/hnpfo9kmdovs3kp/01_2_Dimensionality%20Reduction_Genetic%20Algorithm.pdf))

![image-20221002192054687](./assets/image-20221002192054687.png)





### Step by Step



#### Step 0. Dataset 준비하기

이번에는 Genetic Algorithm의 각각의 Step을 하나씩 설명하며, Code로 구현을 해보도록 하겠다. 일단 Dataset은 Boston Dataset을 가져와서 사용하도록 하겠다. (Scikit-Learn에서는 이제 1.2버전부터 Boston Dataset을 가져오는 함수를 Deprecated할 예정이다. 윤리적 이슈라고 한다. Study목적으로만 사용해야 한다.)

```python
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=4096)
```



#### Step 1. Initialization

전체 Feature에 대해서, Feature들은 Row열로 염색체(Chromosome)라는 Meta Information 정보드를 가질 수 있다. 이 Chromosome 전체는 Population이라고도 부르며, 그것의 개수(Size)은 Hyper-parameter이므로 적절히 잘 선택을 해줘야 한다.

그리고 각각의 염색체는 Binary Encoding(0과 1값)을 하여, 각 염색체별로 어떠한 Feature정보를 가져올지 선택 해 준다.

- 1 : Feature를 가져오기
- 0 : Feature를 가져오지 않기



*8개의 Chromosome으로 이루어진 Population 예시(출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/hnpfo9kmdovs3kp/01_2_Dimensionality%20Reduction_Genetic%20Algorithm.pdf))*

![image-20221002213438844](./assets/image-20221002213438844.png)

해당을 Python 코드로 구현하면 아래와같다.

```python
# Step 1. Initialization
num_features = x_train.shape[1] 
num_population = 8 # hyper-parms for the number of population

def create_init_chromosome(num_features):
    chromosome_instance = [random.randrange(2) for _ in range(num_features)]
    return chromosome_instance


population = [create_init_chromosome(num_features) for _ in range(num_population)]
```



아래는 생성된 population에 대한 Output 예제이다. 8개의 Chromosome Row가 생성되었고, 해당 Feature들은 Random으로 1로 선택된 것을 볼 수 있다.

![image-20221002215315272](./assets/image-20221002215315272.png)



#### Step 2. Fitting Models

이제 선택된 Chromosome별로 예측 모델들을 학습한다. (예제의 Chromosome은 총 8개이므로, 8개의 모델을 생성하여 각 Chromosome별로 성능을 Test해야 한다.) 아래와 같이 Random함수를 통해 각 Feature별 0과 1로 선택할지 말지 여부가결정된 상태에서, Linear모델 등의 모델을 학습한다.

![image-20221012012758734](./assets/image-20221012012758734.png)

아래와 같은 Python Code로 간단히 Linear Model을 Fitting할 수 있다.

```python
chromosome_selected = [bool(chromo) for chromo in chromosome] 
x_train_selected = x_train[:, chromosome_selected]
X_test_selected = x_test[:, chromosome_selected]

linear_model.fit(x_train_selected, y_train)
```







#### Step 3. Model Evaluation

가장 좋은 성능(ex.  `R^2`)인 염색체를 찾기 위해 모델 적합도를 평가한다. 평가 Metric은 R-Square를 사용해도 되지만, MSE, MAE같은 Loss Function을 사용해도 된다. 아래와 같이 Fitting된 모델 각각의 Score를 구해서 모델을 평가하고, 평가된 Score를 저장해 둔다.

```python
# calculate chromosome scores
score = -99999

def calc_score(population):
    scores_chromosome = []
    for chromosome in population:
        if sum(chromosome) == 0: # Chromosome에 1이 하나도 없을 때
            score = -999999 
        else:
            chromosome_selected = [bool(chromo) for chromo in chromosome] 
            x_train_selected = x_train[:, chromosome_selected]
            X_test_selected = x_test[:, chromosome_selected]

            linear_model.fit(x_train_selected, y_train)
            score = linear_model.score(X_test_selected, y_test) 

        scores_chromosome.append(score)
    return scores_chromosome

scores_chromosome = calc_score(population)
```



#### Step 4. Selection

적합도가 높은 염색체들을 부모(Parents) 세대로 선택한다. (다음 세대를 위함). 해당 부모만 교배(Crossover) 및 돌연변이를 만들어 낼 수 있다. 

Deterministic하게 상위 X%만큼만 선택하거나, Probabilistic하게 Score의 %에 따라서 가중치에 따라 Random하게 선택도 가능하다.

![image-20221012013507918](./assets/image-20221012013507918.png)



하기 코드는 Deterministic하게 상위 50%만 선택하게 된다.

```python
# Step 4. Select Parent Chromosome
num_deterministic_selection = int(np.round(num_population * 0.5))

parent_chromosome_index = np.array(scores_chromosome).argsort()[-num_deterministic_selection:]
```



#### Step 5. Crossover & Mutation

유전자들을 교배 및 돌연변이 생성하여, 새로운 Population 생성한다.



##### 5-1) Crossover

선택된 부모 Chromosome들을 2개를 선택하여, 각각의 유전자 정보를 교환한다. (출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/hnpfo9kmdovs3kp/01_2_Dimensionality%20Reduction_Genetic%20Algorithm.pdf))

![image-20221012013651064](./assets/image-20221012013651064.png)



Python Code로는 아래와 같이 구현을 한다. 50%만을 랜덤하게 Crossover하도록 구현하였다.

```python
# 현재의 선택된 Best Parenet 중에서 random으로 좋은 애들 2개를 선택함
select_2 = random.sample(list(range(len(candidate_parent_score))), 2)
winner_p1 = candidate_parent_chromosome[min(select_2, key=lambda idx: candidate_parent_score[idx])]

select_2 = random.sample(list(range(len(candidate_parent_score))), 2)
winner_p2 = candidate_parent_chromosome[min(select_2, key=lambda idx: candidate_parent_score[idx])]
        
# cross-over
c1 = []
c2 = []
	for i in range(len(winner_p1)):
		if random.random() < 0.5: # Random Crossover (50%)
			c1.append(winner_p1[i])
			c2.append(winner_p2[i])
		else:
			c1.append(winner_p2[i])
			c2.append(winner_p1[i])
```



##### 5-2) Mutation

Crossover로 생성된 새로운 Child Chromosome에 대하여 Random하게 값을 Flip한다.



아래의 예시는, 2개의 Child Chromosome에 대하여 Random하게 값을 Flip(0 ->1, 1 -> 0)함을 보여준다. (출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/hnpfo9kmdovs3kp/01_2_Dimensionality%20Reduction_Genetic%20Algorithm.pdf))

Hyper-Parameter Threshold값에 의해서 Mutation을 일으키며, 아래는 0.01이 Threshold일 때의 예시이다.



![image-20221012013841080](./assets/image-20221012013841080.png)



코드로 구현하면 아래와 같으며, 20%의 확률로 Mutation을 일으키도록 하였다. (Threshold가 높을수록 Mutation을 많이하게되며, Global Optimum을 찾기위하여 더욱 많은 변형을 주게 된다.)

```python
# mutation
if random.random() <= 0.2:
	idx_mutation = random.randrange(0,len(c1))
	c1[idx_mutation] = 1 - c1[idx_mutation]
if random.random() <= 0.2:
	idx_mutation = random.randrange(0,len(c2))
	c2[idx_mutation] = 1 - c2[idx_mutation]    
```





#### Step 6. Final Generation

최적 변수 집합(최적 유전자) 선정을 수행한다. GA는 사용자가 정해준 Generation의 횟수에 따라 반복 계산된다. 

Step 2~5의 반복을 통하여 지속적으로 좋은 Chromosome조합을 찾아 Score가 가장 좋은 Feature들을 Selection 함





Best Chromosome은 최종적으로 아래와 같이 계산이 되고, Feature가 1로 선택이 된다.

![image-20221012014104583](./assets/image-20221012014104583.png)



아래 그림은 세대에 따라 Best Score값을 보여주며, 우측은 Population의 Score 표준편차가 얼마나 흔들리는지 보여줌(20%의 돌연변이 발생으로 표준편차가 많이 흔들리게 됨)

![image-20221012014129376](./assets/image-20221012014129376.png)

Best Chromosome은 Python Code로 아래와 같이 구현할 수 있다.

```python
best_index = np.argmax(candidate_parent_score)
best_score = candidate_parent_score[best_index]
best_parent_chromosome = candidate_parent_chromosome[best_index]
```





# Unsupervised Feature Extraction

Unsupervised Feature Extraction은 Selection과 달리 Label이 없는 상태에서, X값을 변환하여 차원을 축소하는 기법이다.



## MDS

### Notebook Tutorial

- [Go to the tutorial]() 
- [Reference Code](https://gist.github.com/Bollegala/24c5f6d9a5c9770c86f24316e8b170fd) 



MDS는 Multidimensional Scaling의 약자로써, PCA처럼 선형 차원축소 기법이다. 하지만 Distance Matrix만 있으면 진행할 수 있으므로, PCA보다 활용의 자유도가 높은 편이며, 향후 진행될 ISOMAP의 기본 로직이 되는 알고리즘이다.



#### Step 1. Distance Matrix 구하기

기본적으로 Raw Data (X Matrix)에서 Distance Matrix를 구한다. Distance Matrix는 다양한 수식을 사용할 수 있으며, Euclidean, Manhattan 거리라던지, Correlation등의 Similarity도 사용 가능하다.



아래 그림과 같이 d x n Matrix는 n x n Matrix로 변환하여 구한다. (출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/sgg7d9s6mxxtu41/01_3_Dimensionality%20Reduction_PCA%20and%20MDS.pdf?dl=0))

![image-20221012020722937](./assets/image-20221012020722937.png)



Python코드로 Distance Matrix를 구하면 아래와 같다. (Data는 3개 사용한다.)

```python
# 1. 3개의 Data를 정의합니다. 각각들의 Dimension은 알려져있지 않고, 단지 Similarity(or Distnace) Metric으로 거리가 계산된, Distance Matrix를 갖고 있습니다.
n = 3  
Y = numpy.array([[20, 18], [2, 13], [7, 24]], dtype=float)

D = numpy.zeros((n, n), dtype=float)

for i in range(0, n):
    for j in range(0, n):
        D[i, j] = numpy.linalg.norm(Y[i,:] - Y[j,:]) # L2-Nrom으로 정규화 합니다.

print("Distance Matrix D")
print(D)
```





#### Step 2. Distance Matrix를 사용해 B Matrix를 구한다.

Distance Matrix를 사용하여, 바로 Dimensionality Reduction된 좌표계로 변환은 힘들다. 따라서 B Matrix라는 중간 다리를 만들어서 최종 줄어든 좌표계로 변환을 진행한다.

B Matrix는 Distance Matrix D를 통해 아래의 수식으로 계산할 수 있다. (출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/sgg7d9s6mxxtu41/01_3_Dimensionality%20Reduction_PCA%20and%20MDS.pdf?dl=0))

![image-20221012021022899](./assets/image-20221012021022899.png)



Python Code로 구현하면 아래와 같다.

```python
def bval(D, r, s):
    n = D.shape[0]
    total_r = numpy.sum(D[:,s] ** 2)
    total_s = numpy.sum(D[r,:] ** 2)
    total = numpy.sum(D ** 2)
    val = (D[r,s] ** 2) - (float(total_r) / float(n)) - (float(total_s) / float(n)) + (float(total) / float(n * n))
    return -0.5 * val
```





#### Step 3. B Matrix에서 Eigen Value와 Eigen Vector를 구하고, 이를 통해 좌표를 변환한다.

좌표변환의 수식은 다음과 같다.  (출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/sgg7d9s6mxxtu41/01_3_Dimensionality%20Reduction_PCA%20and%20MDS.pdf?dl=0))



아래의 수식과 같이, B Matrix는 X와 X^T의 Inner Product로 구할 수 있고, 

![image-20221012021259367](./assets/image-20221012021259367.png)

이는 B는 symmetric, positive, semi-definite and of rank p이므로 다음과 같이 eigen value와 eigen vector로 표현 가능하다. (eigen-decomposition)

![image-20221012021307041](./assets/image-20221012021307041.png)

이를 통해 그 Half인 X를 구할 수 있으며, 최종적으로 변환된 X의 좌표는 아래와 같다.

![image-20221012021313079](./assets/image-20221012021313079.png)

이를 코드로 구현하면 아래와 같다.

```python
# 3. B에서 Eigen Vector와 Eigen Value를 구합니다.
a, V = numpy.linalg.eig(B)
idx = a.argsort()[::-1]
a = a[idx]
V = V[:,idx]
print("Eigen Values =", a)
print("Eigen vectors=", V)

A = numpy.diag(numpy.sqrt(a))
X = numpy.dot(V, A)

print("\nMatrix A")
print(A)

print("\nMatrix X")
print(X)
```





## ISOMAP



### Notebook Tutorial

- [Go to the tutorial]() 
- [Reference Code](https://github.com/lwileczek/isomap) 





ISOMAP은 Isometric Feature Mapping의 약자이며, 비선형 차원축소 기법이다. 기본적으로 MDS를 Base로 하는데, Distance를 Nearest neighbor를 통하여 구하고, 이를 잇는 Shortest Path를 통해 Distance Matrix를 구한다. 그 구해진 Distance Matrix를 통해 MDS로 차원을 축소하게된다.

Step 1, 2를 통해 Distance(D) Matrix를 만들고 Step 3에서 MDS를 사용해 Dimensionality를 Reduction한다.



ISOMAP의 Concept을 도식화하면 아래와 같다. (출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/4v4odamp86brwnv/01_4_Dimensionality%20Reduction_ISOMAP_LLE_tSNE.pdf?dl=0))

![image-20221012022222341](./assets/image-20221012022222341.png)



#### Step 1 & 2. Neighborhood Graph 구조를 만들고, 동시에 Shortest Path로 Distance Matrix D를 구한다.

epsilon-Isomap을 수행하여, eps보다 작은 dist만을 adjacency matrix를 구해서 neighbor를 구한다.

그 외에 k-Isomap을 사용하면 k-nearest neighbor 함수를 사용할 수 있다.

Shortest Path 알고리즘은 Dijkstra부터 Floyd-Warshall, Bellman-Ford 등 다양한 기법을 사용하면 된다.



코드로 epsilon-Isomap을 구현하면 아래와 같다.

```python
# 1. Step 1&2 - epsilon-Neighborhood를 찾고 shortest-path 알고리즘으로 Distance Matrix를 만듬 
def make_adjacency(data, dist_func="euclidean", eps=1):
   """
   ISOMAP을 위한 epsilon-Neighborhood를 찾고 shortest-path 알고리즘으로 Distance Matrix를 만듬 
   Weighted Adjacency Matirx를 각 Point별로 찾는다. eps(epsiolon)안에 들어온 것들만 Neighbor로 취급한다. 

   Neighbor끼리의 거리 계산은 기본은 euclidean이지만, 아래와 같은 distance metric을 cdist가 지원한다.
    'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
    'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
    'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
    'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
    'sokalsneath', 'sqeuclidean', 'yule'.

    INPUT
      data - (ndarray) the dataset which should be a numpy array
      dist_func - (str) the distance metric to use. See SciPy cdist for list of
                  options
      eps - (int/float) epsilon value to define the local region. I.e. two points
                        are connected if they are within epsilon of each other.

    OUTPUT
      short - (ndarray) Distance matrix, the shortest path from every point to
          every other point in the set, INF if not reachable. 
   """
   n, m = data.shape
   dist = cdist(data.T, data.T, metric=dist_func)
   adj =  np.zeros((m, m)) + np.inf
   bln = dist < eps
   adj[bln] = dist[bln]
   short = shortest_path(adj)

   return short
```



#### Step 3. MDS를 수행하여 Embedding을 수행한다.

해당 내용은 MDS 알고리즘과 동일하다. MDS를 Python으로 심플하게 구현하면 아래와 같다. 이를통해 z변수로 축소된 차원을 얻을 수 있다. 

```python
# Step 3. MDS 알고리즘을 사용하여, 원본 dimension을 변환한다.
def MDS(d, dim, m):
    h = np.eye(m) - (1/m)*np.ones((m, m))
    d = d**2
    c = -1/(2*m) * h.dot(d).dot(h)
    evals, evecs = linalg.eig(c)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    evals = evals[:dim] 
    evecs = evecs[:, :dim]
    z = evecs.dot(np.diag(evals**(-1/2)))
    return z
```





## LLE



### Notebook Tutorial

- [Go to the tutorial]() 
- [Reference Code](https://github.com/JAVI897/LLE-and-its-variants)



LLE는 Locally Linear Embedding의 약자로써, Non-Linear Dimensionality Reduction 기법 중 하나이다. 이는 향후 t-SNE에 지대한 영향을 주었으므로 역사적으로 의미있는 기법이라고 볼 수 있겠다. 기본적으로 하나의 점이 주변의 이웃들의 점으로 표현되도록 Weight Matrix를 구하고, 그 Weight Matrix를 고정한 상태에서 차원축소된 좌표를 Weight를 유지한채로 좌표계를 새롭게 저차원에서 최적화하여 찾아낸다.

기본적으로 구현이 쉽고, local minima에 수렴하지 않으며, Non-Linear한 Manifold를 잘 표현해 낸다.

LLE의 Concept 이미지는 아래와 같다. (출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/4v4odamp86brwnv/01_4_Dimensionality%20Reduction_ISOMAP_LLE_tSNE.pdf?dl=0))

![image-20221012023543331](./assets/image-20221012023543331.png)



#### Step 1 & 2. 모든 점에 대해서 Neighbor를 계산하고, 각 point에 대해 선택된 Neighbor에 의해 자기 자신 point가 잘 표현될 수 있도록 선형 변환을 위한 Weight(W)를 계산한다.

Raw Data인 Matrix X의 각각의 point와 k-nearest neighbor point사이의 distance matrix를 사용하여, 최적화를 통해 W를 구한다. 이 Weight Matrix W를 통하여, 각각의 점들은 자신의 주변의 점들로 표현이 가능하다.



그리고 아래의 E(W)가 최소가 되도록, W를 찾는다.   (출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/4v4odamp86brwnv/01_4_Dimensionality%20Reduction_ISOMAP_LLE_tSNE.pdf?dl=0))

![image-20221012024000181](./assets/image-20221012024000181.png)

위에서 찾아진 Matrix W는 고차원 공간에서의 점들간의 관계를 표현하게 되며, 이를 그대로 고정하고 Step 3에서의 저차원에서의 y좌표계를 찾는데 사용한다.



이를 Python Code로 구현하면 아래와 같다.

```python
    def __compute_weights(self):
        """
        Compute weights

        X Matrix의 point와 k-neareset neighbor point사이의 distance matrix를 사용하여, 최적화를 통해 W를 구한다. 
        """
        
        dist_matrix = pairwise_distances(self.X)
        # k_n nearest neighbor indices
        knn_matrix = np.argsort(dist_matrix, axis = 1)[:, 1 : self.k_n + 1]
        
        W = [] # Initialize nxn weight matrix
        for i in range(self.n):
            x_i = self.X[i]
            G = [] # Local covariance matrix
            for j in range(self.k_n):
                x_j = self.X[knn_matrix[i][j]]
                G_aux = []
                for k in range(self.k_n):
                    x_k = self.X[knn_matrix[i][k]]
                    gjk = np.dot((x_i - x_j), (x_i - x_k))
                    G_aux.append(gjk)
                G.append(G_aux)
            G = np.array(G)
            G = G + self.reg*np.eye(*G.shape) # Regularization for G
            w = np.linalg.solve(G, np.ones((self.k_n))) # Calculate weights for x_i
            w = w / w.sum() # Normalize weights; sum(w)=1
            
            if self.verbose and i % 30 == 0:
                print('[INFO] Weights calculated for {} observations'.format(i + 1))
                
            # Create an 1xn array that will contain a 0 if x_j is not a 
            # neighbour of x_i, otherwise it will cointain the weight of x_j
            w_all = np.zeros((1, self.n))
            np.put(w_all, knn_matrix[i], w)
            W.append(list(w_all[0]))
            
        self.W_ = np.array(W)
```





#### Step 3. Weight를 사용해 변환된 공간에서의 각 point와 주변 point와의 거리의 차가 최소가 되도록 최적화 하여, y 좌표계를 계산한다.

고차원에서 구해진 W를 바탕으로 저차원의 y를 구하게 된다.   (출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/4v4odamp86brwnv/01_4_Dimensionality%20Reduction_ISOMAP_LLE_tSNE.pdf?dl=0))

![image-20221012024029129](./assets/image-20221012024029129.png)

결국 위의 수식에 대한 minimize를 하는 y값을 찾기 위해 수식을 전개하면 아래와 같으며, 우리는 M이라는 Matrix를 구할 수있다. M = (I-W)^T(I-W) 이다.

![image-20221012024231637](./assets/image-20221012024231637.png)

이 M은 Rayleitz-Ritz Theorem에 의해 Eigenvector들로 표현이 될 수 있으며, 가장 작은 D+1의 Eigenvector를 Matrix M에서 찾는다. 그리고 찾아진 Eigenvector들에서 맨 마지막 eigenvector [1, 1, 1, ...]을 제외한 D개의 eigenvector가 바로 차원이 축소된 y좌표계를 의미하게 된다.



코드로 구현하면 아래와 같다. (self.Y가 바로 변환된 좌표계 y이다.)

```python
# Compute matrix M : M = (I-W)^T(I-W) 
M = (np.eye(*self.W_.shape) - self.W_).T @ (np.eye(*self.W_.shape) - self.W_) 
eigval, eigvec = np.linalg.eigh(M) # Decompose matrix M
self.Y = eigvec[:, 1:self.dim +1]
```







## t-SNE



### Notebook Tutorial

- [Go to the tutorial]() 
- [Reference Code](https://lvdmaaten.github.io/tsne/) 



드디어 Visualization을 위한 차원축소의 끝판왕(?)인 t-SNE에 도달했다. t-SNE는 t-distributed Stochastic Neighbor Embedding의 약자다.

사실 t-SNE를 구현하기 위해서는 LLE를 거쳐, LLE를 Stochastic하게 나타내는 SNE를 사용하고, SNE를 Symmetric한 가정으로 수식을 바꾼 이후에, t-distribution을 저차원 공간에 적용하는 것으로 마무리 된다.



t-SNE와 MDS를 Visualization한 예시(MNIST)는 아래와 같다. (출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/4v4odamp86brwnv/01_4_Dimensionality%20Reduction_ISOMAP_LLE_tSNE.pdf?dl=0))

![image-20221012025644474](./assets/image-20221012025644474.png)



t-SNE는 아래와 같은 가정으로 구현이 되어있다.

- 가까운 이웃 객체들과의 거리 정보를 잘 보존하는 것이 멀리 떨어진 객체들과의 거리 정보를 보존하는 것 보다 중요함
- SNE는 local pairwise distance를 확정적(deterministic like LLE)이 아닌 확률적(probabilistic)으로 정의함
- 원래 차원과 임베딩 된 이후의 저차원에서 두 객체간의 이웃 관계는 잘 보존이 되어야 함
- 하기와 같이 고차원과 저차원에서의 확률 분포를 구한 후, 두 분포의 유사도를 구해 근사한다. 유사도는 KL-Divergence를 사용한다.
  - p_j|i = 고차원에서 객체 i가 객체 j를 이웃으로 택할 확률
  - q_j|i = 저차원에서 객체 i가 객체 j를 이웃으로 택할 확률
- 기본적으로 고차원에서는 Gaussian 분포를, 저차원에서는 t분포를 사용한다.
  - Gaussian 분포를 사용할 시에, Radius of gaussian을 구해야 하는데, 이때 원하는 수준의 entropy(perplexity)를 hyper-parameter로 사용하여, 적합한 Radius를 결정한다.



결국 p_j|i distribution를 구하고(이때 Perplexity를 사용해 Radius of Gaussian까지 최적화 하게 된다.), 이 p를 고정한 이후에, q_j|i distribution이 p와 동일한 확률에 최대한 가깝게 최적화 하는 문제이다. 물론 이때 p와 q모두 distribution이므로 KL-Divergence를 통해 Cost를 계산하게 된다.



p와 q 분포의 수식은 아래와 같다.  (출처 : [고려대학교 산업경영공학부 강필성 교수님 Business Analytics 수업교재](https://www.dropbox.com/s/4v4odamp86brwnv/01_4_Dimensionality%20Reduction_ISOMAP_LLE_tSNE.pdf?dl=0))

- p_j|i = 고차원에서 객체 i가 객체 j를 이웃으로 택할 확률 (Gaussian에 Normalization한 값)

![image-20221012025247470](./assets/image-20221012025247470.png)

- q_j|i = 저차원에서 객체 i가 객체 j를 이웃으로 택할 확률 (역시 Gaussian에 Normalization한 값)

  ![image-20221012025356233](./assets/image-20221012025356233.png)

- 해당 2개의 고차원과 저차원에서의 분포를 KL-Divergence로 Cost를 구한다.

  ![image-20221012025455410](./assets/image-20221012025455410.png)

- 구해진 Cost Function을 우리가 구하고자 하는 y에 대해 미분하면 아래와 같다.

  - ![image-20221012025525533](./assets/image-20221012025525533.png)

- 해당 Cost에 대한 y의 미분 값을 통하여, y의 값을 지속적으로 Gradient Descent를 사용해 업데이트 하면, 결국 우리가 찾고자하는 저차원의 y좌표계를 구할 수 있다.

  ![image-20221012025609588](./assets/image-20221012025609588.png)

  





#### Step 1.  p Matrix를 구한다. (고차원의 Gaussian Fitting된 분포)

위에서 전개한 p distribution은 아래와 같으며, 이에대한 p matrix를 사전에 fitting하여 구한다. 이때 perplexity를 입력으로 받아, 해당 radius of gaussian도 동시에 최적화를 진행하게 된다. 해당 p값이 구해지면 이 값은 fixed된 값으로 step 2에서 사용되게 된다.

![image-20221012025247470](./assets/image-20221012025247470.png)

파이썬으로 구현하면 아래와 같다. (Raw Data인 Matrix X에서 Matrix p를 구한다.)

```python
def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P
```



이때 x2p함수를 통해 실제 p는 아래와 같이 구해지는데, 이때 P에 P*4를 곱해준다. 이는 논문에서 Early Exaggeration이라고 불리우며, p분포의 확률에 대한 거리를 더 멀리 해 줌으로써, q분포가 더 cluster들이 멀리 떨어지는데 도움을 준다. 이를통해 Optimization이 더 잘 이루어 질 수 있도록 하는 Trick을 사용한다. (물론 나중에 P를 다시 4로 나눈다.)

```python
P = x2p(X, 1e-5, perplexity)
P = P + np.transpose(P)
P = P / np.sum(P)
P = P * 4.									# early exaggeration
P = np.maximum(P, 1e-12)
```



####         Step 2 & 3 & 4. KL Divergence Cost를 최소화하며 q Matrix를 업데이트 하며 저차원의 좌표계 y를 계산한다.

이제 구해진 p Matrix를 고정하고, q Matrix를 구한다. 이때 y는 random으로 초기화 한 상태에서 진행되며, 지속적으로 q distribution을 구하는데, gradient descent를 사용하여 KL-Divergence의 Cost를 최소화하는 방향으로 y값을 계속 업데이트 하면서 q값을 업데이트 한다.

밑의 2~4까지의 step을 iteration을 돌며 반복한다.

- Step 2. q Matrix를 구한다. (저차원에서 t-Distribution으로 Fitting해야 할 분포)

- Step 3.  Gradient를 계산한다.

- Step 4. Gradient(dC/dy)를 통하여 Y값(좌표값)을 업데이트 해 준다.



파이썬 코드로 구현하면 아래와 같다. 여기에서 재밌는건 코드의 가장 아랫쪽에 stop lying about P-values라는 구문이 있는데, 앞서 Step 1에서 사용한 Trick인 Early Exaggeration을 종료하고 p값을 원래값으로 돌린다고 보면 되겠다.

```python
    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        # 2. Q Matrix를 구한다. (저차원에서 t-Distribution으로 Fitting해야 할 분포)
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        # 3. Gradient를 계산한다.
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain

        # 4. Gradient(dC/dy)를 통하여 Y값(좌표값)을 업데이트 해 준다.
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y
```



