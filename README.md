## Deep Active Learning

### Things I've done so far
Added Random Network distillation:
- Fixed target network with random initial weights.
- Fixed prediction network. On first iteration, trains for same number of 
epochs as target. On subsequent iterations it is fine-tuned 
(not trained from scratch) for a few epochs as specified in args.
- The predictor is trained to minimize the MSE loss between its output and 
the target’s output on the already labeled data. After training it I compute 
the loss on the unlabeled data,  data-points on which the mse loss is largest 
are selected for labeling and added to the training set for the next round. 
And I am only fine-tuning the predictor at each round (i.e. not retraining it 
from scratch).

- Meeting notes with Tom: Split unlabeled data into train/"test".
Train predictor on labeled data then sample train data one at a time.
Train using the sampled point (one or two SGD updates), then eval on the 
"test" data (holdout set). The points that show the best improvements
on the test data are those to be queried. 
The Target Net should be the classifier (so that it trains 
on the labeled data everytime.) And the predictor should also have
the same architecture. 

TODO:
- Right now things only work for CIFAR10 need to adapt for other datasets
- Try adding data augmentation?
- Is MSE loss between target and predictor the best way?

Python implementations of the following active learning algorithms:

- Random Sampling
- Least Confidence [1]
- Margin Sampling [1]
- Entropy Sampling [1]
- Uncertainty Sampling with Dropout Estimation [2]
- Bayesian Active Learning Disagreement [2]
- K-Means Sampling [3]
- K-Centers Greedy [3]
- Core-Set [3]
- Adversarial - Basic Iterative Method
- Adversarial - DeepFool [4]

### Prerequisites 
- numpy            1.14.3
- scipy            1.1.0
- pytorch          0.4.0
- torchvision      0.2.1
- scikit-learn     0.19.1
- ipdb             0.11

### Usage 

    $ python run.py

### Reference

[1] A New Active Labeling Method for Deep Learning, IJCNN, 2014

[2] Deep Bayesian Active Learning with Image Data, ICML, 2017

[3] Active Learning for Convolutional Neural Networks: A Core-Set Approach, ICLR, 2018

[4] Adversarial Active Learning for Deep Networks: a Margin Based Approach, arXiv, 2018
