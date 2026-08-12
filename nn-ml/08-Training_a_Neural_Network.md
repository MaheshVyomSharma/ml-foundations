# 08. Training a Neural Network

## 1.1 What Does Training Mean?

Training a neural network means finding values for its trainable parameters that make its predictions increasingly accurate.

The trainable parameters are primarily:

```math
W^{[1]}, W^{[2]}, \ldots, W^{[L]}
```

and:

```math
b^{[1]}, b^{[2]}, \ldots, b^{[L]}
```

Training repeatedly performs:

```text
Data
 ↓
Forward Propagation
 ↓
Prediction
 ↓
Loss
 ↓
Backpropagation
 ↓
Gradients
 ↓
Optimizer
 ↓
Updated Parameters
```

Each repetition should, in general, move the model toward parameters that produce lower loss.

---

## 1.2 Before Training Begins

Before the first training step, several things must already be defined.

### Data

The model needs input features and corresponding targets:

```math
(\mathbf{x},y)
```

### Architecture

For example:

```text
Input
 ↓
Dense(64, ReLU)
 ↓
Dense(32, ReLU)
 ↓
Dense(1, Sigmoid)
```

### Loss Function

For binary classification:

```text
Binary Cross-Entropy
```

### Optimizer

For example:

```text
SGD
Adam
```

### Hyperparameters

Such as:

- learning rate
- batch size
- number of epochs

Only then can training begin.

---

## 1.3 Weight Initialization

The network needs initial values for its weights before it can make its first prediction.

Conceptually:

```math
W \leftarrow \text{initial values}
```

These values are generally initialized to small random values according to an appropriate initialization strategy.

Weights should generally **not all be initialized to the same value**.

Suppose all neurons in a layer begin with identical weights.

They receive the same inputs, produce the same outputs, receive the same gradients, and undergo the same updates.

They therefore continue learning the same thing.

This is called a **symmetry problem**.

Random initialization breaks this symmetry.

Biases, however, can often safely begin at zero because different weights already distinguish the neurons.

Specialized initialization methods such as Xavier/Glorot and He initialization will be considered separately.

---

## 1.4 The Training Dataset

Suppose the training dataset contains:

```math
N
```

examples.

Training does not normally mean showing the dataset to the network only once.

Instead, the network processes the training data repeatedly while progressively adjusting its parameters.

One complete pass through all `N` training examples is called an **epoch**.

---

## 1.5 Epoch

An **epoch** is one complete pass through the training dataset.

For example:

```text
Epoch 1
→ network sees all training examples

Epoch 2
→ network sees them again

Epoch 3
→ again
```

After each pass, the network's parameters should generally be better than they were earlier in training.

However:

```text
More epochs ≠ automatically better model
```

Too few epochs may cause underfitting.

Too many can contribute to overfitting.

---

## 1.6 Batch

A **batch** is a subset of the training dataset processed together.

Suppose:

```math
N=10{,}000
```

and the batch size is:

```math
B=100
```

The dataset is divided into approximately:

```math
\frac{10{,}000}{100}=100
```

batches.

Each batch produces a forward pass, loss calculation, backward pass, and parameter update.

---

## 1.7 Iteration

An **iteration** is one parameter-update step.

Therefore:

```text
One Batch
   ↓
Forward Pass
   ↓
Loss
   ↓
Backward Pass
   ↓
Optimizer Step
   ↓
One Iteration
```

If:

```math
N=10{,}000
```

and:

```math
B=100
```

then:

```math
\text{iterations per epoch}=100
```

If training runs for 20 epochs:

```math
20\times100=2000
```

parameter updates occur.

---

## 1.8 Epoch, Batch and Iteration

These three terms are easy to confuse.

```text
Epoch
= one complete pass through the training set

Batch
= subset processed together

Iteration
= one parameter update
```

Approximately:

```math
\text{Iterations per Epoch}
=
\left\lceil
\frac{N}{B}
\right\rceil
```

where:

```math
N=\text{number of training examples}
```

and:

```math
B=\text{batch size}
```

The ceiling accounts for a final batch that may contain fewer than `B` examples.

---

## 1.9 One Training Iteration

For a mini-batch:

```math
(X_{\text{batch}},Y_{\text{batch}})
```

the network performs four fundamental steps.

### Step 1 — Forward Propagation

```math
\hat{Y}
=
F(X_{\text{batch}};\theta)
```

### Step 2 — Calculate Loss

```math
J
=
L(Y_{\text{batch}},\hat{Y})
```

### Step 3 — Backpropagation

```math
\nabla_\theta J
```

is calculated.

### Step 4 — Parameter Update

For basic gradient descent:

```math
\theta
\leftarrow
\theta
-
\eta\nabla_\theta J
```

That completes one iteration.

---

## 1.10 One Epoch

An epoch repeatedly performs the training iteration for every batch:

```text
Epoch Begins

Batch 1
→ Forward
→ Loss
→ Backward
→ Update

Batch 2
→ Forward
→ Loss
→ Backward
→ Update

Batch 3
→ Forward
→ Loss
→ Backward
→ Update

...

Final Batch
→ Forward
→ Loss
→ Backward
→ Update

Epoch Ends
```

Then another epoch begins using the newly learned parameter values.

---

## 1.11 Why Shuffle the Training Data?

Before each epoch, training examples are commonly **shuffled**.

Suppose the dataset is ordered:

```text
Class A
Class A
Class A
Class A
...
Class B
Class B
Class B
...
```

Without shuffling, consecutive batches may contain highly biased subsets of the data.

Shuffling produces batches that better represent the overall training distribution.

Conceptually:

```text
Ordered Dataset
      ↓
Shuffle
      ↓
Create Mini-Batches
      ↓
Train
```

The data is typically shuffled between epochs.

---

## 1.12 Training, Validation and Test Sets

Neural-network data is commonly separated into:

```text
Training Set
Validation Set
Test Set
```

Each has a different role.

### Training Set

Used to calculate gradients and update parameters.

### Validation Set

Used during model development to evaluate generalization and guide choices such as:

- architecture
- hyperparameters
- number of epochs
- regularization

### Test Set

Used for final evaluation after model development is complete.

The test set should not repeatedly influence model-design decisions.

---

## 1.13 Training Loss

The **training loss** measures performance on data being used to optimize the network.

During successful training, it will generally decrease:

```text
Loss
│\
│ \
│  \
│   \__
│      \___
│
└──────────── Epochs
```

A falling training loss tells us that the network is becoming better at fitting its training data.

It does **not**, by itself, prove that the network generalizes well.

---

## 1.14 Validation Loss

After or during an epoch, the network can be evaluated on validation data.

Importantly:

```text
Validation Data
      ↓
Forward Propagation
      ↓
Validation Loss / Metrics
```

There is no parameter update from the validation data.

Validation performance estimates how well the current model behaves on unseen examples.

---

## 1.15 Detecting Underfitting

A model may be **underfitting** if both training and validation performance remain poor.

Conceptually:

```text
Training Loss   → high
Validation Loss → high
```

Possible causes include:

- insufficient model capacity
- too few training epochs
- poor features
- inappropriate architecture
- poor optimization settings

The model has not adequately learned the underlying pattern.

---

## 1.16 Detecting Overfitting

During overfitting, training performance continues improving while validation performance begins worsening.

Conceptually:

```text
Loss
│\
│ \             Validation
│  \          __/
│   \       _/
│    \_____/
│
│ Training
│   \____________
│
└──────────────── Epochs
```

The network has become increasingly specialized to the training data.

A typical signal is:

```text
Training Loss   → continues decreasing
Validation Loss → begins increasing
```

---

## 1.17 Generalization

The real objective of training is not merely:

```text
Minimize training loss
```

It is:

```text
Learn patterns from training data
        ↓
Perform well on unseen data
```

This ability is called **generalization**.

A model that memorizes its training set but performs poorly elsewhere is not a successful model.

---

## 1.18 Early Stopping

**Early stopping** stops training when validation performance stops improving.

Suppose validation loss behaves like:

```text
Epoch 1  → 0.80
Epoch 2  → 0.62
Epoch 3  → 0.51
Epoch 4  → 0.46
Epoch 5  → 0.45
Epoch 6  → 0.47
Epoch 7  → 0.50
```

The model around epoch 5 may generalize better than the later versions.

Instead of automatically using the final epoch, we can retain the model from the best validation point.

---

## 1.19 Patience

Validation loss naturally fluctuates.

Therefore, training is usually not stopped after one slightly worse epoch.

A **patience** value specifies how many epochs to wait for improvement.

For example:

```text
patience = 3
```

means roughly:

> Stop if the monitored validation quantity fails to improve for three consecutive epochs.

This prevents training from stopping because of a single noisy measurement.

---

## 1.20 Checkpoints

A **checkpoint** stores the model's state during training.

For example:

```text
Epoch 1
Epoch 2
Epoch 3 ← checkpoint
Epoch 4
Epoch 5 ← best validation result
Epoch 6
```

A checkpoint may contain:

- weights
- biases
- optimizer state
- training progress

Checkpoints allow training to be resumed or the best-performing model to be restored.

---

## 1.21 Training Mode vs Inference Mode

Some neural-network components behave differently during training and inference.

During training:

```text
Forward Pass
+ Loss
+ Backpropagation
+ Parameter Updates
```

During inference:

```text
Forward Pass Only
```

Components such as **dropout** and **batch normalization** also have specific training/inference behaviour.

Therefore, modern deep-learning frameworks explicitly distinguish between training and evaluation modes.

---

## 1.22 Hyperparameters

Unlike weights and biases, **hyperparameters** are not normally learned directly through backpropagation.

Examples include:

- learning rate
- batch size
- number of epochs
- number of layers
- neurons per layer
- activation functions
- optimizer
- regularization strength
- dropout rate

These choices strongly affect training behaviour.

---

## 1.23 Parameters vs Hyperparameters

The distinction is:

```text
Parameters
→ learned during training
→ weights and biases

Hyperparameters
→ configure training/model
→ chosen externally
```

For example:

```math
w=0.742
```

may be learned by gradient descent.

But:

```math
\eta=0.001
```

is usually selected as a training hyperparameter.

---

## 1.24 Monitoring Training

Useful quantities to monitor include:

```text
Training Loss
Validation Loss
Training Metric
Validation Metric
Learning Rate
```

For classification, metrics might include:

- accuracy
- precision
- recall
- F1 score

For regression:

- MAE
- RMSE
- R²

The loss drives optimization, while metrics help interpret practical model performance.

---

## 1.25 A Healthy Training Pattern

A desirable pattern is:

```text
Training Loss
↓ steadily

Validation Loss
↓ generally

Training Metric
↑

Validation Metric
↑
```

Some gap between training and validation performance is normal.

A rapidly growing gap may indicate overfitting.

---

## 1.26 Learning Curves

Plots of training and validation performance over epochs are called **learning curves**.

They help diagnose training behaviour.

### Both Losses High

```text
Possible underfitting
```

### Training Low, Validation High

```text
Possible overfitting
```

### Both Low and Similar

```text
Good fit / generalization
```

Learning curves are therefore one of the most useful diagnostic tools during neural-network development.

---

## 1.27 Data Leakage

A serious training mistake is **data leakage**.

This occurs when information from validation or test data improperly influences training.

For example, fitting a scaler using the complete dataset before splitting can leak information.

Correct workflow:

```text
Training Data
      ↓
Fit Preprocessing
      ↓
Transform Training Data

Validation/Test Data
      ↓
Use Already-Fitted Preprocessing
      ↓
Transform
```

The same principle encountered in classical machine learning applies to neural networks.

---

## 1.28 Training Pipeline

A practical neural-network workflow is:

```text
Prepare Data
     ↓
Split Data
     ↓
Fit Preprocessing on Training Data
     ↓
Build Network
     ↓
Initialize Parameters
     ↓
Choose Loss
     ↓
Choose Optimizer
     ↓
Train on Mini-Batches
     ↓
Monitor Validation Performance
     ↓
Stop Training
     ↓
Restore Best Model
     ↓
Final Test Evaluation
```

---

## 1.29 Training Is an Optimization Process

It is useful to strip away the terminology and see what is actually happening.

The network represents:

```math
\hat{y}=F(\mathbf{x};\theta)
```

The loss measures:

```math
J(\theta)
```

Backpropagation calculates:

```math
\nabla_\theta J
```

The optimizer modifies:

```math
\theta
```

The entire training problem is therefore:

```math
\theta^*
=
\operatorname*{arg\,min}_{\theta}
J(\theta)
```

Everything else helps this optimization happen efficiently and generalize well.

---

## 1.30 The Complete Picture So Far

The concepts covered so far now connect into one system:

```text
Artificial Neuron
      ↓
Weights + Bias
      ↓
Activation Function
      ↓
Neural Network Architecture
      ↓
Forward Propagation
      ↓
Prediction
      ↓
Loss Function
      ↓
Backpropagation
      ↓
Gradients
      ↓
Optimizer / Gradient Descent
      ↓
Updated Weights
      ↓
Repeat Across Batches and Epochs
      ↓
Trained Neural Network
```

This is the fundamental mechanism underlying neural-network learning.

---

## 1.31 Key Takeaways

- Training means learning useful values for the network's weights and biases.
- Weights require suitable initialization before training begins.
- Random initialization breaks symmetry between neurons.
- An **epoch** is one complete pass through the training dataset.
- A **batch** is a subset of training examples processed together.
- An **iteration** is one parameter-update step.
- Mini-batches are commonly shuffled between epochs.
- Training data updates parameters.
- Validation data evaluates the model during development without updating parameters.
- Test data is reserved for final evaluation.
- Training loss alone does not measure generalization.
- Underfitting occurs when the model fails to adequately learn the training pattern.
- Overfitting occurs when training performance improves while unseen-data performance deteriorates.
- Early stopping can stop training when validation performance ceases to improve.
- Patience prevents early stopping from reacting to small temporary fluctuations.
- Checkpoints allow useful model states to be saved and restored.
- Parameters are learned; hyperparameters configure the model and training process.
- Learning curves help diagnose underfitting, overfitting, and healthy training.
- Data leakage must be avoided just as in classical machine learning.
- The ultimate objective is not merely low training loss, but good generalization to unseen data.

### Memory Hook

```text
One Batch
→ One Training Iteration
→ One Parameter Update

All Batches
→ One Epoch

Many Epochs
→ Training

Training Data
→ Learn

Validation Data
→ Tune and Monitor

Test Data
→ Final Evaluation

Healthy Training
= Training Improves
+ Validation Improves

Training Loop:
Forward
→ Loss
→ Backward
→ Update
→ Repeat
```