# PyTorch-Medical-Image-Classification-Project

Problem: This project tackles a binary medical image classification task: detecting pneumonia from pediatric chest X-ray images. The goal is not perfect accuracy, but an explainable ML pipeline suitable for screening use case where false negatives (missed pneumonia) are more costly than false positives.

Dataset
- PneumoniaMNIST from MedMNIST
- 28×28 grayscale chest X-ray images
- Imbalanced dataset (~74% pneumonia)
- Predefined train / validation / test splits used to avoid leakage

Dataset splits
Train: 4,708 images| Validation: 524 images | Test: 624 images

Model overview
A simple convolutional neural network (CNN) is used as a baseline model.

Why a CNN?
Images have spatial structure. CNNs are designed to: detect local patterns (edges, textures) share parameters across space scale better than fully connected models for images

Architecture (conceptually)
- Convolution → ReLU → Max Pool
- Convolution → ReLU → Max Pool
- Fully connected layers for classification
The final layer outputs a single number called a logit.

Logits, sigmoid, and probabilities
The model does not directly output probabilities.

Instead: it outputs a logit (a raw score) a sigmoid function converts the logit into a probability. This is the same idea as logistic regression:
- logistic regression → linear score + sigmoid
- CNN → learned features + sigmoid

Loss function and training
Binary cross-entropy with logits
Training uses binary cross-entropy applied to logits.

Conceptually: penalizes confident wrong predictions, aligns with probabilistic binary classification, is the deep-learning equivalent of logistic regression loss

Because the dataset is imbalanced: 
- mistakes on the minority class (normal) are penalized more
- mistakes on the majority class (pneumonia) are penalized less

This prevents the model from simply predicting the majority class to get high accuracy.

Optimization
Optimizer: Adam
Learning rate: 0.001
Epochs: 5
Training updates model weights using gradient descent via backpropagation.


Evaluation metrics
Accuracy alone is not sufficient due to class imbalance.
The following metrics are used:
Accuracy
Precision
Recall
F1 score
Confusion matrix (TP, FP, FN, TN)

These metrics allow direct analysis of: false positives (normal flagged as pneumonia) and false negatives (missed pneumonia cases)

Threshold tuning (decision rule): 
The model outputs probabilities, but probabilities must be converted into decisions.

Default threshold
Probability ≥ 0.5 → pneumonia
Probability < 0.5 → normal


For screening: recall is prioritized, threshold is lowered to catch more pneumonia cases

Validation data is used to test multiple thresholds.
The selected threshold is 0.10, which yields very high recall.

Final test performance (screening setup)
Using threshold = 0.10 on the test set:
Recall ≈ 99%
Precision ≈ 76%
Very few missed pneumonia cases
More false positives (acceptable for screening)


Limitations
- Low image resolution (28×28)
- Single dataset
- No external validation
- Simple baseline architecture

Possible next steps: 
- Transfer learning with a pretrained CNN (e.g. ResNet)
- AUROC analysis
- Model calibration
- Grad-CAM for interpretability
- Higher-resolution medical datasets
