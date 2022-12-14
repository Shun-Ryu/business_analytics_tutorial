# 🤔MixMatch를 구현할 때 생각해봐야 하는 것들

## The things to consider for implementing MixMatch Algorithm

![image-20221226195436600](./attachments/image-20221226195436600.png)



🔥이번 Tutorial에서는 **Semi-Supervised Learning Method** 중, **Holistic한 접근 방법인 MixMatch**를 구현해보면서, **구현 상에 고려해야 할 사항들**에 대해서 생각해 보는 시간을 갖으려고 한다. Github에 있는 여러가지 구현체들을 확인하였는데, 논문에는 드러나지 않은 사항들이 보이기에 이 Tutorial에서 여러가지 테스트와 함께 In-Depth하게 알아보고자 한다.

이는 사실 MixMatch 뿐만 아니라, Original MixMatch (from Google) 알고리즘에서 파생되어 나온 **FixMatch**에서도 동일한 구현상 고려해야 하는 사항이므로, 이번 Tutorial을 통해 잘 배워보는 시간을 갖도록 하자.

논문에서도 잘 다루지 않고, 구현체에서도 잘 설명이 없는 아래의 2가지 주제로 Tutorial을 전개하려 한다. 나머지 부분은 대부분 까다롭지 않다.

- MixMatch에서 **EMA(Exponential Moving Average)로 Teacher모델**을 만드는 것이 중요한가? 
- MixMatch에서 **Interleaving** 구현은 중요한가?



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

MixMatch는 기존의 Consistency Regularization과 Pseudo-Labeling과 같은 기법에서 벗어나서, 기존의 방법론들을 여러개를 결합하여 Holistic(전체론적)인 Approach로 접근한 최초의 논문이다. (from Google Research). 지금은 잘 모를수도 있으나, 처음 이 방식이 나왔을 때, 기존의 방법론들을 압도하는 Performance로 많은 이들에게 신선한 충격을 선사하였던 논문이다. 



**MixMatch에서 차용하고 있는 전체적인 방법론들은 아래와 같다.**

1. Data Augmentation 
2. Label Guessing & Label Sharpening
3. MixUp



위의 것들은 모두 과거의 논문들에서 많이 보아왔던 기법들인데, 이러한 **여러기법들을 조합하여 전체적인 Architecture 구조 형태로 좋은 성능**을 가져갔다는 것이, MixMatch 를 포함한 후속 Match시리즈 기법들의 특징이라고 볼 수 있다. 특히 Holistic하다고 하지만, MixMatch 같은 경우 특히나 구현과 구조가 매우 단순하면서도 좋은 성능을 가져왔기 때문에 선구자적인 논문 이라고 볼 수 있겠다.

이중 가장 중요한 것은 MixUp이며 이를 Shuffle구조를 가져가며 Data와 Label을 Matching해 주었기 때문에 MixMatch라고 부르지 않나 싶다.







> 🔥이제 알고리즘을 순서(Sequence)대로 알아보자..!



## 1) Data Augmentation

먼저 모든 Data의 Data Augmentation을 수행한다. 일반적으로 Computer Vision Deep Learning Model들이 그렇듯, **Regularization로써 Augmentation을 사용하여 일반화 성능을 향상**시킨다. 특히 나중에 있을 MixUp에서 더 다양한 조합으로 Labeled Data와 Unlabeled Data를 섞어서 학습할 수 있도록 만들어 준다.

![image-20221226232414467](./attachments/image-20221226232414467.png)

위의 그림처럼 Labeled Data는 Batch별로 1회 Random Augmentation을, Unlabeled Data는 Batch별로 K회 Random Augmentation을 진행한다. 이때 논문에서 진행한 Augmentation은 아래와 같이 **Random horizontal flips와 Random Crops**이다.



> ![image-20221226232556152](./attachments/image-20221226232556152.png)



또한 논문에서는 Unlabeled Data에 대해서는 2회의 Augmentation을 진행하도록 Hyper-Parameter를 세팅하였다.

>  ![image-20221227010258137](./attachments/image-20221227010258137.png)





## 2) Label Guessing and Label Sharpening

이 방식은 Pseudo Labeling과 동일한 방식이며, **Only Unlabeled Data에 대해서만 Label Guessing**을 수행한다. 또한 마찬가지로 Guessing된 **Unlabeled Data의 Label에 대해서만 Label Sharpening**을 진행한다. 전체적인 Flow는 아래와 같다. 

>  구현 시에 Augmented된 Label Data에 Guessing을 하는 것이 아님을 꼭 유의 해야 한다.



![image-20221226200950638](./attachments/image-20221226200950638.png)

위와 같이 Batch별로 자동차를 K개의 Random Augmentaton 한후, Guessing된 Label들을 Average하고, 그 값을 Sharpening한다. Sharpening이라는 것은 확률이 높은 것을 좀 더 강조(Temperature라는 Hyper-Parameter T를 사용하여, 얼마나 강조할지 조정한다.) 이 Label Sharpenig을 통해 Unlabeled Data의 Pseudo-Label에 대한 Entropy가 Minimization된다. (즉, 하나의 Guessing Label을 더 강조한다는 이야기임. 전체 Guessing Label이 Unifrom형태를 띈다면, Entropy가 Maximization이 된다.) 

> 특히 Entropy Minimization은 2005년 Semi-supervised learning by entropy minimization (Yves Grandvalet and Yoshua Bengio) 논문의 관찰을 통해 Idea를 얻었다고 저자들은 이야기 한다. 이는 그리고 High-Density Region Assumption을 더 강조하기 위함이다.

![image-20221226234523055](./attachments/image-20221226234523055.png)

![image-20221226233648148](./attachments/image-20221226233648148.png)

위와 같이 일반적인 Softmax의 확률값에 1/T 승을 하여 값을 결정하게 된다. T값이 2.0, 0.5, 0.1에 따라서, 값이 더 작을 수록 Winnner-Take-All을 하게 된다. 논문에서는 아래와 같이 Hyper-Parameter T값을 아래와 같이 0.5로 세팅 하였으며, 우리의 구현에서도 마찬가지로 0.5로 세팅 할 생각이다.

> ![image-20221226233938750](./attachments/image-20221226233938750.png)



## 3) MixUp

> MixMatch의 이름에 왜 Mix가 들어갔는지에 대한 이유이다. 그만큼 중요한 부분이다.

MixUp(2018, mixup: BEYOND EMPIRICAL RISK MINIMIZATION)에 나온 기법으로써, 원래는 좀 더 Decision Boundary를 Smooth하게 하여 Emperical Risk Minimization을 Supervised Learning에서 활용하기 위해 제안된 단순한 기법이다. 그러나 2019년 Interpolation consistency training for semi-supervised learning(ICT)에서 처음으로 이 방법을 Semi-Supervised Learning에 적용을 하였다. **MixUp을 통해서 좀 더 Smooth한 Boundary를 강제하는 Regularization**을 수행할 수 있다. 



>  Unlabeld Data에 대해서만 MixUp을 진행한 기존의 ICT 방식과 다르게, MixMatch는 Labeled Data와 Unlabeled Data모두에 MixUp을 수행한다. 



일종의 Data Agumentation과 유사한데, 위의 1)에서 Augmentation해서 생겨난 Data에 한번더 MixUp이라는 Augmentation을 진행하고, 그 X Data에 대한 예측과 Augmented된 Target Y값 사이의 오차를 줄이는 방식으로 학습한다. 상세하게는 아래의 그림과 같다.

![image-20221227010035007](./attachments/image-20221227010035007.png)





- **1) Augmented Set을 준비한다.**
  - 위에서 진행한 Labeld Set의 Data에 1회 Augmentation
  - Unlabeld Set Data에 K회 Augmentation
- **2) Shuffle Set을 준비한다.**
  - Augmentation Set을 Copy하여 복사하고, Labeled Set과 UnLabeled Set을 모두 합쳐서 섞는다. 이때 Data X와 Target Y도 각각 섞어 준다.
- **3) MixUp을 진행한다.**
  - 1)과 2)에서 준비한 2개의 Set을 Beta 분포에서 뽑은 Lambda값을 통해서 Weighted Sum을 통해 Mixup한다. 이때 Lambda는 0.5~1사이의 값으로 주도록 하여, Shuffle Set보다 Augmented Set에 더 Weight를 주도록 한다. 그 이유는 각각의 Target_L(Labeled)과 Target_U(Unlabeled)에 대하여, Matching되도록 값을 주기 위함이다. (**아마 이 때문에 MixMatch라고 부르는게 아닌가 싶다.**) 그렇지 않는다면 Shuffle Set에 Bias되기 때문에 잘 Matching되도록 Lambda를 조절한다.
  - 그리고 MixUp된 Data_L(Labeled)과 Data_U(Unlabeled)에 대해서는 Model에 넣고, 각각의 예측 값인 Pred_L(Labeled)과 Pred_U(Unlabeled)를 뽑는다.
  - 그리고 Pred_L과 Target_L사이의 Loss를 Cross Entropy를 통해 Supervised Loss로 사용하고, Pred_U와 Target_U사이의 값은 Distance Metric(ex. MSE Loss)을 사용해 Unsupervised Loss를 구한다.
  - 그리고 2개의 Loss를 미리정해진 Hyper-Parameter값 Weight를 통해 조절한다. (그리고 이 값은 Learning Rate처럼, Ramp-Up을 통해 조절하도록 한다.)





----

# Tutorial. Deep Understanding of MixMatch Implementation

이번 튜토리얼에서는 전체적인 MixMatch를 Scratch로 구현해 보면서 알고리즘을 이해해 보려한다. 특히 원 저자들이 논문에서 제대로 건드리지 않았으나(혹은 반대로 이야기했으나!), 원저자들의 구현체 혹은 기타 다른 구현체들에서 이미 구현하고 있었으나 제대로 논의가 이루어지지 않은 영역에 대한 이해를 실험과 함께 가져가려 한다. 제기하고 싶은 의문은 2가지이다.

- MixMatch에서 **EMA(Exponential Moving Average)**로 Teacher모델을 만드는 것이 중요한가? 
- MixMatch에서 **Interleaving** 구현은 중요한가?

>  동시에 해당 논문에 대해서 Reviewer들이 이야기한 내용, 그리고 구현체들의 Issue에 제기한 의문들도 함께 보면서 이야기 해 보려 한다.



## 1. Tutorial Notebook 

### 🔥[Go to the tutorial notebook](https://github.com/Shun-Ryu/business_analytics_tutorial/blob/main/5_semi-supervised_learning/Tutorials/Tutorial_MixMatch.ipynb)



## 2. Setting

### Datasets

데이터는 유명한 CIFAR-10을 사용하도록 한다. 10개의 Class를 갖고 있는 32x32x3 Shape의 Imageset이다.

![image-20221227100541139](./attachments/image-20221227100541139.png)

|      | Datasets                  | Description                               | Num Instances                                   | Image Size | Num Channels | Single Output Classes |
| ---- | ------------------------- | ----------------------------------------- | ----------------------------------------------- | ---------- | ------------ | --------------------- |
| 1    | CIFAR-10 (Classification) | 10개의 Class를 가진 작은 이미지 데이터 셋 | Training Set : 50,000<br />Testing Set : 10,000 | 32 x 32    | 3            | 10                    |

데이터셋은 아래와 같은 코드로 불러오게 된다. 

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

이때 Dataset은 Pytorch의 Dataloader를 통해 Training Time에 불러지게 되며, Dataset은 동시에 아래의 코드로 Augmentation이 이루어 진다. Labeled Data는 1번의 Random Augmentation이 진행되며, Unlabeled Data는 K=2로써 2번의 Data Augmentation이 수행된다. 다음의 기능이 아래의 코드로 이루어 진다.

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

우리는 MixMatch 1가지만 집중적으로 파 보고자 한다. 특히 위에서 제기한 2가지, EMA 방식의 Teacher Network와 Interleave에 대한 이해를 파보겠다.

| Algorithm | Target         | Description                                                  |
| --------- | -------------- | ------------------------------------------------------------ |
| MixMatch  | Classification | WideResNet을 Backbone으로 사용한 Holistic Semi-Supervised Learning 알고리즘을 사용. |



## 3. Implementation of MixMatch

MixMatch는 위에서 설명하였던, Data Augmentation, Label Guessing & Sharpening 그리고 마지막으로 MixUp을 결합한 방식으로 구현된다.  상세하게 그 구현체에 대해서 알아보도록 하자.



### 3-1. Loss, EMA, Interleaving Functions

MixMatch의 Learning기능을 구성하기 위한 함수의 집합이다. 특히나 Training을 위한 함수들이며, 차근차근 상세히 알아보자



> Semi-Supervised Loss / Ramp-Up Function

아래는 Semi-Supervised Loss를 구하기 위한 Each_Loss Class와 그 2개의 supervised loss, unsupervised loss의 weighted sum을 구하기 위한 ramp_up 함수를 정의한다. ramp_up은 총 epoch수에 따라 구성이 되며, 선형적으로 weight가 증가되도록 구현이 되어있다. each_loss는 labeled data의 target값과 prediction값을 통해 supervised loss(loss_l)를 구하고, unlabeled data의 target값과 prediction값을 통해 unsupervised loss(loss_u)를 구한다. 

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



> 🔥 Exponential Moving Average for training Teacher Model Function

Fire Emoticon을 붙였다. 그만큼 중요하다는 뜻. Exponential Moving Average(EMA)를 통하여 Student Model의 Parameter를 Teacher Model의 Parameter로 전이하는 함수이다. 밑에서 실험을 통해 알아보겠지만, 이 EMA구현이 되어지지 않으면 모델은 제대로 학습이 이루어 지지 않는다. (논문에서는 오히려 반대되는 설명을 하고있다.) 성능 향상에 필수적인 함수이며, 그에 대한 설명이 없으므로 꼭 구현을 해야만 한다. Student Model만으로는 Prediction성능이 잘 나오지 않는다.

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



> 🔥 Interleaving Function

역시나 Fire Emoticon을 붙였다. 아래의 Interleaving의 구현은 논문 원 저자의 구현체에도 존재하며, 그 외의 대부분의 인기있는 구현체에서도 아래의 방식으로 Interleaving을 사용한다. Interleaving은 Labeled Data와 Unlabeled Data의 값들을 서로 Mixing해주어, Model을 계산하기 위해 존재한다. 왜냐하면, 대부분의 구현체에 있어 Labeled Data를 Model에 입력하여 Supervised Loss를 구하고, 그 이후에 Unlabeled Data를 Model에 입력하여 Unsupervised Loss를 구하기 때문에 문제가 발생한다. 이렇게 각각의 Data를 Model에 따로, 2번 태워서 계산할 경우, Batch Normalization을 진행 할 때에 두개의 분포가 전체의 Batch의 분포를 대변하지 않기 때문에, 학습시에 Bias가 생기게 된다.

따라서 이번 Tutorial에서는 Interleaving을 사용한 경우와, 사용하지 않는 경우의 학습 방법에 대해서 두가지 모두 구현을 진행하였다.(코드의 Training부분을 확인하세요..!) 그리고 아래의 Test에서 Interleaving을 고려하지 않는 학습 방법에 있어서, 학습이 잘 되지 않는 다는 것을 실험을 통해 밝혀내도록 하겠다.

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

Training Function은 MixMatch의 순서에 따라 구현되어 있다. 사실 어려운 코드는 아니며, 직관적으로 이해가 되므로 간단하게 순서대로 한번 살펴보도록 하자.



> 1. Data Augmentation

Data Augmentation을 진행하는 코드이다. 위에서 정의했던 CIFAR-10을 불러오면서 Augmentation을 하도록 Pytorch의 DataLoader를 만들었고, 그에 따라서 데이터를 불러오게 된다. 

- Labeled Data : inputs_l / target_l
- Unlabeld Data : inputs_u, inputs_u2 (즉, K=2)

여기에 try, except가 있는데 이는 보통 supervised에서는 training시에 batch에 맞춰서 data를 load하지만, MixMatch같은 경우 Labeled Data와 Unlabeled Data가 서로 개수가 다르기 때문에(비율에 따라 다르지만 Unlabeled가 더 많거나 같음), Batch와 관계없이 training_iteraction횟수에 따라 Data를 loading하기 때문에 모든 Data Loader에서 sampling을 다 수행했을 경우, 다시금 Data Loader를 위한 Iterator를 만들기 위해 except 코드가 존재한다. 크게 어려울 것 없는 코드이다.

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

Pseudo Labeling을 위하여 Unlabeled Data를 예측하는 단계이다. 예측을 하고 나서 Temperature Hyper-Parameter를 통해 label간의 확률을 sharpening한다. 이것도 어려울 게 없는 코드이다.

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

매우 중요한 부분이다. MixMatch의 핵심이 되는 코드라고 볼 수 있다. (그러나 구현이 어렵지 않다.). 아래의 2개의 Data를 MixUp해 준다.

- 원래의 Augmented된 Labeled Data와 Unlabeled Data
- 추가적으로 Labeled Data+Unlabeld Data를 붙여서 Shuffle한 Data

그 때에 두개의 Data를 Weighted Sum을 해 주는데, 원래의 Augmented Data에 좀 더 가중 치를 더 주어, 목표하는 Target값과 Matching되도록 구성을 한다. 그래서 알고리즘 이름이 MixMatch이지 않을까 싶다.

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

이게 논문에서도 그렇고 인터넷에도 그렇고 잘 설명이 되어있지 않는 부분이다. 대부분의 github의 구현체에서 interleaving방식을 사용하는데, 이는 mixed된 labeled data와 unlabeled data를 서로 interleaving(데이터의 sample의 일부분을 서로 교환)하도록 하여, batch별로 batch-normalization을 할때 그 분포가 변화되지 않도록 하는 기능을 구현해 놓은 것이다. 왜 batch-normalization할 때 분포가 잘학습되지 않냐하면, 아래의 구현에서 처럼 labeled data를 model에 따로 흘리고, unlabeled data를 model에 따로 흘리기 때문이다.

🔥이 때문에 각각의 model이 batch-normalization의 parameter가 개별적으로 bias되며 학습이 이루어지 기 때문이다.

🔥따라서 MixMatch뿐만 아니라, FixMatch같은 경우도 마찬가지인데, 이렇게 Labeled Data와 Unlabeled Data를 Model에 따로 흘릴경우..그리고 Backbone Model에서 batch-normalization을 사용할 경우는 꼭 Interleaving function을 구현하여 사용해야 한다. 그렇지 않을 경우 학습 자체가 잘 이루어지지 않는다!

🔥또한 Google Reseasrch의 Fix Match저자들이 github issues에 답변하기를, Multiple-GPUs를 사용할 경우, Interleaving을 통해 데이터를 섞어준다음 각 GPU로 흘렸을때 역시나 Batch-Norm이 잘 학습되기 때문에 이렇게 구현하였다고한다. 그러나 이는 Tensorflow의 경우이고 Pytorch는 Multi-GPUs를 위한 Batch-Norm 구현이 따로 있이므로, 이렇게 할 필요는 없다고 생각된다.

🔥그러나 Labeled Data와 Unlabeled Data를 합쳐서, 한번에 Model에 흘릴 경우는 이야기가 달라진다. 이 때에는 Interleaving이 필요가 없으며, 당연히 한번의 Batch에 2개의 데이터 형태가 동시에 들어가므로, 분포의 변화가 적절히 잘 학습이 된다. 

🔥그런데 여기에서 주의해야 할 구현 사항이 있는데, interleaving을 할 경우 Batch-Size가 1:2로 나뉘면서 학습이 되지만, No-interleaving일 경우 Batch-Size가 3배가 되기 때문에 기존의 Learning Rate로는 학습이 느려지게 된다. 이를 위해서 많이들 사용하는 Batch Size가 k배가 되면 Learning Rate는 sqrt(k)배로 증가시키는 방식으로 학습을 진행하면 학습이 잘 되는 것을 확인할 수 있었다.

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

간단히 구현하는 Semi-Supervised Loss이다. 간단하다. 2개의 Loss를 각각 구해서 weighted sum을 수행한다.

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

당연한 backpropagation. Backpropagation은 Student Model에만 학습이 진행된다. Teacher모델은 아래의 EMA Learning으로 진행된다.

```python
########################################
# 6. Backpropagation 
########################################
optimizer.zero_grad()
loss_total.backward()
optimizer.step()

```



> 7. EMA Learning for Teacher Model

EMA을 통한 Teacher Model의 학습이다. 결국에는 우리가 사용할 모델은 Teacher모델이며(Student Model은 학습이 잘 이루어지지 않고, 성능이 잘 나오지 않는다.), 저자들은 EMA가 오히려 모델 성능에 상처(hurt)를 준다고 하였으나, EMA를 통한 Teacher Model이 결론적으로 훨씬 학습도 잘되고 성능도 잘 나온다. 이는 여러 구현체에서 EMA를 기븐으로 가져가고 있고, 인터넷 커뮤니티 상에서도 많은 이들이 EMA를 통해 성능을 뽑아낼 수 있다고 말하고 있다.

```python
########################################
# 7. EMA Learning for Teacher Model 
########################################
if is_ema is True:
    ema_optimizer.step()
```







## 4. Result_Accuracy

- 측정 단위 : Accuracy
- Unsupervised Augmentation 횟수 K=2 (논문에서 제시한 수치)
- Accuracy는 Testset에 대해서 계산



### 4-1) Original 모델과의 비교

| #Labels   | 250          | 2000         |
| --------- | ------------ | ------------ |
| Paper     | 88.92 ± 0.87 | 92.97 ± 0.15 |
| This code | 86.76        | 91.57        |

위의 표를 보면 전반적으로 구현이 잘 되었음을 확인할 수 있다. Seed값이나 Hyper-Parameter에 따라서 결과 성능이 조금씩 달라질 수 있다고 생각한다. 또한 **시간 관계상 학습을 매우 오랜 시간 돌리지 못하였고, 완전히 수렴하지 않은 상태에서 학습을 조기 종료 했다는 것**을 감안해야 하겠다.





### 4-2) Test Result


|      | #Labeles | Interleave             | EMA  | Accuracy                    |
| ---- | -------- | ---------------------- | ---- | --------------------------- |
| 1    | 250      | -                      | -    | 63.00 (이후 성능 계속 하락) |
| 2    | 250      | -                      | O    | 61.87 (이후 성능 계속 하락) |
| 3    | 250      | O                      | -    | 80.31                       |
| 4    | 250      | O                      | O    | **86.76**                   |
| 5    | 250      | O (No-Interleave Mode) | O    | 84.17                       |
| 6    | 2000     | O                      | -    | 88.76                       |
| 7    | 2000     | O                      | O    | **91.57**                   |

Test결과는 위와 같다. Label된 데이터를 몇개나 썼는지에 따라서 분류하였으며, Interleave를 사용했을 때와 하지 않았을때, 그리고 EMA로 Teacher Model을 학습하여 Accuracy를 계산 했을 때와 하지 않았을 때를 비교하였다. **Label이 많을 수록 성능의 향상 효과가 컸으며, Interleave가 없다면 아얘 학습이 안된다고 볼 수 있다. 또한 EMA를 통한 Teacher Model의 성능이 EMA 없는 Student Model보다 더 좋은 성능을 발휘함**을 알 수 있었다. 

또한 새롭게 구현한 **No-Interleave Mode**도 역시 학습이 잘 된다는 점을 알 수 있었고, Interleave는 Batch-Norm의 계산만 잘 고려한다면, 해당 함수의 구현 없이 다른 방식으로도 학습을 잘 이루어지도록 만들수 있다는 것을 알게 되었다. (No-Interleave Mode가 성능이 좀 더 낮아보이지만, 기본적으로 **시간관계상 완벽히 수렴하지 않은 상태에서 학습을 조기 종료**하여 그렇다고 보면 되겠다. 또한 Semi-Supervised Loss의 Weighted Sum을 위한 Lambda와 Learning Rate의 Hyper-Parameter를 최적화하여 더 좋은 결과를 낼 수 있을 거라 생각한다.)





----


# Final Insights

우리는 이번 Tutorial을 통하여, MixMatch를 구현할 때에, 원래 논문에서 언급하지 않거나 제대로 설명하지 않는 영역인 EMA와 Interleave의 효과에 대해서 확인해 보았다. Test를 통해 전반적으로 얻은 Insight는 아래와 같다.

- **Labels개수 변화에 따른 효과**
  - 250개보다 확실히 2000개로 Label된 데이터를 늘렸을 때 성능 향상이 크게 일어났으며 수렴도 굉장히 빠르게 진행되었다. Label의 최대한 많은 확보가 Semi-Supervised에도 성능을 높일수 있음을 확인할 수 있었다.
  - 학습되는 Loss를 관찰하였을 때, Supervised Loss가 처음에는 빠르게 떨어지다가 나중에는 Unsupervised Loss를 떨어뜨리는 쪽으로 이동하였다. 이는 Ramp-Up의 효과라고 생각되며, 이를 통해 전체적인 Training Loss가 많이 떨어지지는 않아도, Unsupervised Loss때문에 Test Accuracy가 지속적으로 증가하게 되었다.
- **EMA의 효과**
  - EMA가 있어야 확실히 원하는 수준까지 성능이 나올 수 있음을 확인할 수 있었다.
  - EMA가 없이 기본 Student Model로도 어느정도 예측이 가능하지만, EMA통한 Teacher Model이 더 성능이 높음을 알 수 있었다.
  - 특히 Loss의 변화를 보았을때, EMA를 사용한 Teacher Model은 Smooth하게 안정적인 학습이 되고, 또한 Test Set에 대하여 Ensemble효과를 통하여 더 좋은 Generalization 성능을 보임을 알 수 있었다.
  - 제1저자는 해당 EMA를 Weight Deacy를 대체하기 위해 사용하고 있었다. 일종의 Regularization을 진행한다고 보는 것 같다. 나는 개인적으로 Ensemble의 효과도 있다고 생각이 든다.
- **Interleave의 효과**
  - Interleave를 아얘 하지 않고, 개별 Label과 Unlabel Data의 예측을 한다면, 성능이 지속적으로 하락하게 된다. 위에서 이야기 했듯, Batch-Norm이 제대로 계산되지 않기 때문이라고 생각된다.
  - Interleave를 사용할 경우, 제대로 학습이 됨을 알 수 있었다. 특히나 일반적인 Interleave 사용했을 때와, No-Interleave Mode를 사용했을 때 거의 유사한 결과를 얻을수 있음을 알 수 있었다. 물론 제대로 학습되도록 하기 위해 Learing Rate의 조절은 필요하다. (Batch Size가 변화하므로)



그리고 추가적으로 인터넷을 통하여 관련된 정보들을 수집하여 EMA와 Interleave에 대하여 아래와 같이 정리 해 보았다.



> EMA Case

원 논문의 저자들은 아래와 같이 EMA Parameter의 사용은 MixMatch에 안좋은 영향을 주는 것같다고 하였는데 실제적으로 EMA는 성능을 향상시키는 효과가 있었으며, 원 논문 저자들의 구현체 뿐만 아니라 다른 구현체들도 EMA가 Optional이 아니라 필수적으로 다 넣은 것을 보면 EMA는 성능 향상에 도움이 된다는 것을 알 수 있다.

![image-20221227130933075](./attachments/image-20221227130933075.png)

또한 MixMatch의 Neurips Review에서 아래와 같은 Comment가 있다. ([Reviews: MixMatch: A Holistic Approach to Semi-Supervised Learning (neurips.cc)](https://proceedings.neurips.cc/paper/2019/file/1cd138d0499a68f4bb72bee04bbec2d7-Reviews.html))

"From the reproduction by my group, we found the **EMA plays an essential role in achieving the results**. Without it, there would be a non-unneglectable gap to the showed results. Therefore, it is encouraged to include an ablation study of the EMA to show its impact on the proposed model. "

이번 Tutorial에서도 마찬가지 결과였는데, 논문 저자들에게 EMA를 넣었을 때와 뺐을 때를 Ablation Study를 하는게 좋다고 Review를 하였지만, 저자들은 EMA를 넣었을 때에만 Test를 진행하였다. 논문 제 1저자는 EMA를 Weight Decay를 대체하여 쓰고 있다고 한다.(MixMatch와 FixMatch에서도)

![image-20221227132326819](./attachments/image-20221227132326819.png)

그리고 재미있는 것은 아래와같이, Mean Teacher 모델(https://arxiv.org/abs/1703.01780)에서도 그렇지만, Student Model보다 Teacher모델이 성능이 더 좋았다. 하지만 해당 Mean Teacher논문도 그렇고 MixMatch논문도 그렇고, Prediction시에 Teacher를 쓰는게 좋은지, Student를 쓰는게 좋은지에 대한 Guideline이 없다는 것이 아쉽다.

![image-20221227133008179](./attachments/image-20221227133008179.png)

> Interleave Case

또한 Interleave에 대해서 MixMatch의 1저자는 Github Issues에 아래와 같이 답변하고 있다. Interleave를 쓰는 목적은 Multi-GPUs를 사용할때 Batch-Norm을 잘 계산하기 위해서라고. 하지만, PyTorch의 경우는 따로 Parallel Batch-Norm을 구현하고 있기 때문에 이렇게 구현할 이유는 없을 것 같다. 

![image-20221227131126671](./attachments/image-20221227131126671.png)



그리고 재밌는 사실은 원 저자는 Multi-GPUs를 위해서라고 답변했으나, 실제적으로 보면 Single-GPU에서도 Model을 2회 각각 Inference하기 때문에 Batch-Norm의 분포가 깨어져서 성능이 잘 나오지 않는다. (원 저자의 구현에서도 Parallel을 심지어 쓰지 않기도 한다;;) 이에 대한 Github Issues에 대한 글이 있어서 아래에 짧게 공유한다. (나와 동일한 생각이다.)

![image-20221227131421312](./attachments/image-20221227131421312.png)



# Conclusion

결론적으로, 

- MixMatch에서 **EMA(Exponential Moving Average)로 Teacher모델**을 만드는 것은 **중요하다.**
  - 모델의 성능의 Student Model만으로 나오지 않는다.
  - EMA는 Regularization 역할을 통해 Generalization을 더 잘하도록 만든다.
- MixMatch에서 **Interleaving** 구현은 **중요하다**.
  - Semi-Supervised에서 Labeled Data와 Unlabeled Data를 각각 따로(즉 2회), Model에 돌리게 될 경우 Batch-Norm계산이 분포가 Bias되어 학습이 잘 이루어 지지 않는다. 이를 막기 위해 Labeled Data와 Unlabeled Data간 Data Sample들을 섞어주는 Interleaving이 효과를 보게 된다.
  - 그러나 Labeled와 Unlabeled Data를 한번에 동시에 Model에 넣어 계산한다면, Interleaving은 굳이 필요없을 수 있다.



-----

# References

-  고려대학교 산업경영공학부 강필성 교수님 Business Analytics 강의 자료
- https://hipolarbear.tistory.com/19
- https://proceedings.neurips.cc/paper/2019/file/1cd138d0499a68f4bb72bee04bbec2d7-Reviews.html
- https://www.reddit.com/r/MachineLearning/comments/jb2egk/d_consitency_training_how_do_uda_or_fixmatch/
- https://github.com/kekmodel/FixMatch-pytorch/issues/19
- https://github.com/google-research/fixmatch/issues/20
- https://github.com/kekmodel/FixMatch-pytorch/issues/36
- https://github.com/google-research/fixmatch/issues/37

