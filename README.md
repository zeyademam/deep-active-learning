## Deep Active Learning

### Current code does: 
Here’s a summary of what we discussed:
1. Initialize two networks: C and S, both resnet18. We will refer to C_feat(x) as the output of the last layer (50 dimensional) before of the fully connected layer. 
2. Partition Training data T into two subsets chosen uniformly at random: LT of size 5k and UT of size 45k. LT will refer to the current pool of labeled training data, UT the pool of unlabeled training data. 
3. Train C on LT using random weight initialization (i.e. do not finetune) and using the regular CIFAR labels. 
4. Freeze the weights of C (torch.no_grad). Distill S on LT using C_feat as the teacher. 
5. Partition UT into two subsets chosen at random: UTT of size 80%, and UTE for the rest. UTT is the data we will “score” and decide whether or not to label next. UTE is the data we will use for evaluation. 
6. For each x in UTT: (i) Load the weights of S from end of step 4. (ii) Run 2 SGD updates of S’s weights using x only (using l2 loss between S_feat(x) and C_feat(x)). (iii) Compute the l2 distance between S and C_feat on all the data in UTE. This will be the “score” of x. Lower score is better because it means training on x helped minimize the loss on UTE. 
7. Pick the 100 points x in UTT with the lowest scores and add them to LT. 
8. Repeat steps 3 through 7 multiple times. 
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

- Latest Meeting Notes with Tom (below did not work): Try only finetuning the 
last layer
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
- Try adding data augmentation? (Tom said to hold off on this for now, 
just normalization is fine)
- Is MSE loss between target and predictor the best way?
- Make sure the predictor trains to saturation

Discussion with Ilya:
- Does SGD on one example work? 
- Try much smaller dataset. (i.e. truncate all of 50k CIFAR images to 5k say)
- Does feature matchine (i.e. L2 loss) work for distillation? 
What about mean feature matching -
i.e. l2norm(avg(teacher(batch)) - avg(student(batch)))
- What works without sota hacks (augmentations and stuff used in mix and match 
paper) might not work when including all the hacks. 

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
