# 19. Deep Learning Workflow and Model Development

## 1.1 From Concepts to a Complete Deep Learning System

So far, the individual pieces of deep learning have been studied separately:

```text
Neurons
Activations
Layers
Forward Propagation
Loss Functions
Backpropagation
Optimizers
Initialization
Regularization
Normalization
CNNs
RNNs
LSTMs / GRUs
```

In practice, these components form one complete engineering workflow.

A deep-learning project is not simply:

```text
Choose Neural Network
        ↓
Train
        ↓
Done
```

Instead, model development is an iterative process:

```text
Define Problem
      ↓
Prepare Data
      ↓
Choose Architecture
      ↓
Build Baseline
      ↓
Train
      ↓
Evaluate
      ↓
Diagnose
      ↓
Improve
      ↓
Evaluate Again
```

The purpose of this chapter is to connect everything into that complete process.

---

## 1.2 Step 1 — Define the Problem

Before selecting an architecture, determine exactly what the model must predict.

Examples:

```text
Image → Cat or Dog
→ Binary Classification

Image → One of 100 Objects
→ Multiclass Classification

Sensor History → Future Temperature
→ Regression / Forecasting

Sentence → Positive or Negative
→ Sequence Classification

Sequence → Label at Every Position
→ Sequence Labelling
```

The problem definition influences:

- architecture
- output layer
- loss function
- evaluation metric
- data preparation

Architecture should follow the problem, not the other way around.

---

## 1.3 Step 2 — Understand the Data Structure

Different neural architectures exploit different forms of structure.

```text
Ordinary Numerical Features
→ Feed-Forward Network

Images / Spatial Data
→ CNN

Sequential / Temporal Data
→ RNN / LSTM / GRU
```

The important question is:

> What relationships exist inside the input that the architecture should exploit?

For images:

```text
Spatial Locality
```

For sequences:

```text
Order and Temporal Dependency
```

For ordinary tabular data:

```text
Feature Relationships
```

Choosing architecture is therefore partly about matching the model's inductive structure to the data.

---

## 1.4 Neural Networks Are Not Automatically the Best Model

Deep learning is powerful, but it is not automatically the correct solution.

For many tabular datasets:

```text
Linear Models
Decision Trees
Random Forest
Gradient Boosting
```

may perform extremely well with:

- less data
- less computation
- easier interpretation
- faster experimentation

A practical ML engineer asks:

```text
What model suits the problem?
```

not:

```text
How can I force a neural network onto this problem?
```

Deep learning is a tool, not a default answer.

---

## 1.5 Step 3 — Split the Data

A typical workflow separates data into:

```text
Training Set
Validation Set
Test Set
```

Their roles are different.

### Training Set

Used to learn:

```text
Weights
Biases
Other Trainable Parameters
```

### Validation Set

Used to choose:

```text
Architecture
Hyperparameters
Regularization
Training Duration
```

### Test Set

Used for final evaluation after development decisions have been made.

Thus:

```text
Training
→ Learn

Validation
→ Decide

Test
→ Verify
```

---

## 1.6 Prevent Data Leakage

**Data leakage** occurs when information unavailable during real prediction influences training.

This can produce deceptively strong evaluation results.

Examples include:

```text
Scaling Using the Entire Dataset

Using Future Data to Predict the Past

Including Target-Derived Features

Using Test Data During Hyperparameter Tuning
```

A model affected by leakage may appear excellent during development and fail after deployment.

The fundamental rule is:

```text
Training Must Not Gain
Information It Would Not Have
During Real Inference
```

---

## 1.7 Preprocessing Must Respect the Split

Suppose input standardization uses:

```math
x'
=
\frac{x-\mu}{\sigma}
```

The values:

```math
\mu
```

and:

```math
\sigma
```

should be learned from the training data.

Then the same transformation is applied to validation and test data.

```text
Training Data
      ↓
Fit Preprocessing
      ↓
Learn μ and σ

Validation / Test
      ↓
Use Same μ and σ
```

We do not independently fit preprocessing using the test set.

---

## 1.8 Step 4 — Prepare the Inputs

Input preparation depends on the data type.

### Numerical Data

Possible operations:

```text
Scaling
Missing-Value Handling
Feature Transformation
```

### Images

Possible operations:

```text
Resize
Normalize Pixel Values
Augmentation
```

### Text

Possible operations:

```text
Tokenization
Vocabulary Construction
Embedding
Padding
Masking
```

### Time Series

Possible operations:

```text
Sequence Windowing
Scaling
Temporal Ordering
Missing-Value Handling
```

The model can only learn from the representation it receives.

---

## 1.9 Step 5 — Establish a Baseline

Before building an elaborate architecture, establish a simple baseline.

For example:

```text
Simple Model
      ↓
Train Successfully
      ↓
Record Validation Performance
```

A baseline answers:

```text
Does the pipeline work?

Can the model learn anything?

Is the complex model actually better?
```

Without a baseline, complexity has no meaningful reference point.

---

## 1.10 Baselines Need Not Be Neural Networks

For an image problem, a baseline might be a small CNN.

For tabular classification, it might instead be:

```text
Logistic Regression
Decision Tree
Gradient Boosting
```

For time-series forecasting, it could even be:

```text
Predict the Previous Value
```

A useful baseline need not be sophisticated.

It provides a minimum performance reference.

---

## 1.11 Step 6 — Choose the Architecture

The architecture should reflect the problem.

A rough guide is:

```text
Tabular / General Numerical Data
→ Dense Feed-Forward Network

Images
→ CNN

Sequential Data
→ RNN / LSTM / GRU

Very Long / Complex Sequence Relationships
→ Attention-Based Architectures
```

Within the chosen family, decisions include:

```text
Number of Layers
Layer Width
Activation Functions
Normalization
Dropout
Residual Connections
```

These are architectural hyperparameters.

---

## 1.12 Step 7 — Choose the Output Layer

The output layer must match the prediction task.

### Regression

Often:

```text
Linear Output
```

For a single target:

```math
\hat{y}
=
z
```

### Binary Classification

Typically:

```text
One Output Neuron
+
Sigmoid
```

### Multiclass Classification

Typically:

```text
One Output per Class
+
Softmax
```

The output representation and loss function must agree.

---

## 1.13 Output Layer and Loss Pairing

Common combinations are:

| Task | Output | Common Loss |
|---|---|---|
| Regression | Linear | MSE / MAE |
| Binary Classification | Sigmoid | Binary Cross-Entropy |
| Multiclass Classification | Softmax | Cross-Entropy |
| Multi-label Classification | Independent Sigmoids | Binary Cross-Entropy |

These pairings are not arbitrary.

The output activation represents the form of prediction, while the loss measures how wrong that prediction is.

---

## 1.14 Step 8 — Initialize the Parameters

Training begins from initialized parameters.

Good initialization helps preserve useful activation and gradient scales.

Common principles include:

```text
ReLU Networks
→ He Initialization

Tanh / Sigmoid Networks
→ Xavier / Glorot Initialization
```

Poor initialization can contribute to:

```text
Vanishing Activations
Exploding Activations
Vanishing Gradients
Exploding Gradients
Slow Training
```

Initialization creates the starting conditions for optimization.

---

## 1.15 Step 9 — Choose an Optimizer

Common choices include:

```text
SGD
SGD + Momentum
RMSProp
Adam
AdamW
```

The optimizer converts gradients into parameter updates.

Conceptually:

```text
Backpropagation
→ Calculates Gradients

Optimizer
→ Decides How to Use Them
```

Adam or AdamW often provides a practical starting point.

SGD with momentum can also perform extremely well when appropriately tuned.

---

## 1.16 Step 10 — Choose the Learning Rate

The learning rate controls update magnitude.

```math
\theta_{t+1}
=
\theta_t
-
\eta\nabla_\theta J
```

where:

```math
\eta
```

is the learning rate.

Its effect is fundamental:

```text
Too Small
→ Slow Learning

Too Large
→ Oscillation / Divergence

Appropriate
→ Efficient Convergence
```

When training behaves badly, learning rate should be among the first things inspected.

---

## 1.17 Step 11 — Choose Batch Size

Batch size controls how many examples contribute to each gradient estimate.

```text
Small Batch
→ Noisier Gradients
→ More Frequent Updates
→ Lower Memory Requirement

Large Batch
→ Smoother Gradients
→ Fewer Updates
→ Greater Memory Requirement
```

Batch size also interacts with:

- learning rate
- hardware utilization
- Batch Normalization
- generalization behaviour

It should therefore be treated as part of the overall training configuration.

---

## 1.18 Step 12 — Forward Propagation

For every mini-batch:

```text
Inputs
 ↓
Layer 1
 ↓
Activation
 ↓
Layer 2
 ↓
Activation
 ↓
...
 ↓
Output
 ↓
Prediction
```

Each layer performs some transformation.

For a dense layer:

```math
z^{[l]}
=
W^{[l]}a^{[l-1]}
+
b^{[l]}
```

followed by:

```math
a^{[l]}
=
g(z^{[l]})
```

CNNs and recurrent networks alter the structure of these transformations, but the overall principle remains forward computation.

---

## 1.19 Step 13 — Calculate the Loss

The model's prediction is compared with ground truth.

```text
Prediction
+
Ground Truth
      ↓
Loss Function
      ↓
Scalar Loss
```

For regression, an example is:

```math
J
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
```

For classification, cross-entropy is commonly used.

The loss provides the objective that training attempts to minimize.

---

## 1.20 Step 14 — Backpropagation

Backpropagation computes how the loss changes with respect to trainable parameters.

Conceptually:

```text
Loss
 ↓
Output Layer Gradient
 ↓
Hidden Layer Gradients
 ↓
Earlier Layer Gradients
 ↓
Parameter Gradients
```

Using the chain rule:

```math
\frac{\partial J}{\partial W}
```

tells us how changing a weight would affect the loss.

Backpropagation calculates the learning signal.

It does not itself update the parameters.

---

## 1.21 Step 15 — Optimizer Update

The optimizer uses the gradients to modify parameters.

For simple gradient descent:

```math
W
\leftarrow
W
-
\eta
\frac{\partial J}{\partial W}
```

and:

```math
b
\leftarrow
b
-
\eta
\frac{\partial J}{\partial b}
```

More advanced optimizers modify this rule using momentum or adaptive statistics.

Thus:

```text
Backpropagation
→ Find Direction

Optimizer
→ Take the Step
```

---

## 1.22 One Training Step

A single training step is:

```text
Mini-Batch
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
Parameter Update
```

This is repeated across mini-batches.

---

## 1.23 One Epoch

If the training set contains:

```math
N
```

examples and batch size is:

```math
B
```

the approximate number of training steps per epoch is:

```math
\frac{N}{B}
```

An epoch represents one complete pass through the training dataset.

Training therefore looks like:

```text
Epoch 1
→ Many Mini-Batch Updates

Epoch 2
→ Many Mini-Batch Updates

Epoch 3
→ Many Mini-Batch Updates

...
```

---

## 1.24 Monitor Training and Validation Loss

Two of the most informative quantities are:

```text
Training Loss
Validation Loss
```

They reveal different aspects of model behaviour.

### Training Loss

Measures how well the model fits data used for learning.

### Validation Loss

Measures how well the current model generalizes to unseen development data.

Their relationship is diagnostically useful.

---

## 1.25 Healthy Learning Pattern

A healthy early training pattern often looks like:

```text
Training Loss
↘

Validation Loss
↘
```

Both decrease.

This suggests that the model is:

```text
Learning Training Patterns
+
Improving Generalization
```

Training can continue while validation performance continues improving.

---

## 1.26 Underfitting

Suppose:

```text
Training Loss
→ High

Validation Loss
→ High
```

The model may be underfitting.

Possible causes include:

- insufficient model capacity
- too few epochs
- excessive regularization
- poor learning rate
- weak input representation

Conceptually:

```text
Model Cannot Even Fit
Training Data Well
        ↓
Underfitting
```

---

## 1.27 Responding to Underfitting

Possible actions include:

```text
Increase Model Capacity
        ↓
Add Layers / Units

Train Longer

Reduce Excessive Dropout

Reduce Excessive Weight Decay

Improve Input Features

Check Learning Rate
```

The correct response depends on the cause.

Blindly making the network larger is not always the answer.

---

## 1.28 Overfitting

Suppose:

```text
Training Loss
→ Very Low

Validation Loss
→ Significantly Higher
```

or validation loss begins rising while training loss continues falling.

The model may be overfitting.

Conceptually:

```text
Training Data
→ Memorized Extremely Well

Unseen Data
→ Performance Worsens
```

The model has learned patterns that do not generalize sufficiently.

---

## 1.29 Responding to Overfitting

Possible actions include:

```text
More Training Data

Data Augmentation

Dropout

L2 / Weight Decay

Early Stopping

Reduce Model Capacity

Improve Data Quality
```

The goal is not to maximize training performance.

The goal is to maximize useful performance on unseen data.

---

## 1.30 Bias and Variance Perspective

Underfitting and overfitting connect to the bias-variance trade-off.

Conceptually:

```text
Underfitting
→ High Bias

Overfitting
→ High Variance
```

A model with high bias is too restricted to represent the underlying pattern adequately.

A model with high variance responds too strongly to peculiarities of the training data.

The desired region is:

```text
Enough Capacity to Learn
+
Enough Constraint to Generalize
```

---

## 1.31 Early Stopping

Early stopping monitors validation performance.

```text
Train
 ↓
Validation Improves
 ↓
Continue

Validation Stops Improving
 ↓
Wait for Patience Period
 ↓
Stop Training
```

The best model checkpoint is often restored.

Early stopping therefore serves two purposes:

```text
Avoid Wasted Training
+
Reduce Overfitting
```

---

## 1.32 Model Checkpointing

During training, model parameters can be saved periodically.

A common strategy is:

```text
Validation Improves
      ↓
Save Model
```

If later epochs worsen validation performance:

```text
Restore Best Saved Model
```

This avoids assuming that the final epoch is necessarily the best model.

---

## 1.33 Learning Curves

Plots of metrics across epochs are called **learning curves**.

Common plots include:

```text
Training Loss vs Epoch
Validation Loss vs Epoch
Training Accuracy vs Epoch
Validation Accuracy vs Epoch
```

They can reveal:

- convergence
- overfitting
- underfitting
- instability
- insufficient training

Learning curves are among the most useful diagnostic tools in deep learning.

---

## 1.34 Signs of a Learning Rate That Is Too Large

Possible symptoms include:

```text
Loss Oscillates Wildly

Loss Increases

Loss Becomes NaN

Training Never Settles

Gradients / Parameters Become Extremely Large
```

A common response is:

```text
Reduce Learning Rate
```

but numerical instability or bad preprocessing should also be investigated.

---

## 1.35 Signs of a Learning Rate That Is Too Small

Possible symptoms include:

```text
Loss Decreases Extremely Slowly

Training Appears Stuck

Many Epochs Produce Tiny Improvements
```

If gradients exist but parameter updates are minuscule, the learning rate may simply be too small.

---

## 1.36 Vanishing Gradient Diagnosis

Possible signs include:

```text
Early Layers Learn Very Slowly

Gradient Magnitudes Near Zero

Deep Network Fails to Improve
```

Possible remedies include:

```text
ReLU-Like Activations

Better Initialization

Normalization

Residual Connections

LSTM / GRU for Recurrent Problems
```

The appropriate solution depends on the architecture.

---

## 1.37 Exploding Gradient Diagnosis

Possible signs include:

```text
Huge Gradient Norms

Sudden Loss Explosion

NaN / Infinity Values

Wild Parameter Updates
```

Possible responses include:

```text
Lower Learning Rate

Gradient Clipping

Better Initialization

Normalization

Inspect Data Scaling
```

Exploding gradients are especially familiar in recurrent networks but can occur elsewhere.

---

## 1.38 Gradient Clipping

Gradient clipping limits excessively large updates.

For norm clipping:

```math
g_{\text{clipped}}
=
g
\frac{c}{\|g\|}
```

when:

```math
\|g\|>c
```

where:

```math
c
```

is the clipping threshold.

This changes:

```text
Huge Gradient
→ Controlled Gradient
```

without changing ordinary gradients that already lie below the threshold.

---

## 1.39 Regularization as Part of the Workflow

Regularization should respond to evidence of overfitting rather than being added blindly.

Available tools include:

```text
L1 Regularization
L2 Regularization
Weight Decay
Dropout
Data Augmentation
Early Stopping
```

Regularization introduces constraints or noise that discourage the model from fitting training-specific details too aggressively.

---

## 1.40 Normalization as Part of the Workflow

Normalization can improve optimization.

Examples include:

```text
Input Standardization

Batch Normalization

Layer Normalization
```

These operate at different points.

```text
Input Scaling
→ Before / At Model Input

BatchNorm / LayerNorm
→ Inside the Network
```

Normalization and regularization should not be confused.

---

## 1.41 Training Mode and Evaluation Mode

Some layers behave differently during training and inference.

### Dropout

```text
Training
→ Random Units Dropped

Inference
→ All Units Used
```

### Batch Normalization

```text
Training
→ Batch Statistics

Inference
→ Stored Running Statistics
```

Therefore, the model must be placed into the correct mode when evaluating or predicting.

---

## 1.42 Hyperparameter Tuning

After establishing a working baseline, important hyperparameters can be tuned.

Examples include:

```text
Learning Rate
Batch Size
Layer Count
Layer Width
Dropout Rate
Weight Decay
Optimizer
Kernel Size
Number of Filters
Hidden-State Size
```

Search methods include:

```text
Manual Search
Grid Search
Random Search
Bayesian Optimization
Hyperband
```

The validation set guides these choices.

---

## 1.43 Tune the Important Things First

Not every hyperparameter deserves equal effort.

A useful priority is often:

```text
Learning Rate
      ↓
Model Capacity
      ↓
Regularization
      ↓
Optimizer / Batch Size
      ↓
Fine Architectural Choices
```

There is little value in carefully tuning a minor parameter while the learning rate is catastrophically wrong.

---

## 1.44 Evaluation Metrics Must Match the Goal

Loss is used for optimization, but the most meaningful evaluation metric depends on the application.

### Regression

Examples:

```text
MAE
MSE
RMSE
R²
```

### Classification

Examples:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
PR-AUC
```

The best metric depends on the real-world cost of different errors.

---

## 1.45 Loss and Evaluation Metric Are Not Necessarily the Same

A model may train using:

```text
Cross-Entropy Loss
```

while being evaluated using:

```text
F1 Score
```

This is perfectly valid.

The loss must be suitable for gradient-based optimization.

The evaluation metric must reflect the practical objective.

Thus:

```text
Loss
→ How the Model Learns

Metric
→ How We Judge It
```

---

## 1.46 Classification Thresholds

A binary classifier may output:

```math
P(y=1\mid x)
```

For example:

```math
0.73
```

A threshold converts this probability into a class.

With:

```math
\tau=0.5
```

we predict:

```text
Probability ≥ 0.5
→ Positive
```

But `0.5` is not a universal law.

The threshold can be selected based on:

- precision-recall trade-off
- false-positive cost
- false-negative cost
- operational requirements

The model learns probabilities or scores; the decision threshold is a separate deployment choice.

---

## 1.47 Error Analysis

A single metric does not explain *why* a model fails.

Inspect incorrect predictions.

For classification:

```text
False Positives
False Negatives
Frequently Confused Classes
```

For images:

```text
Poor Lighting?
Occlusion?
Rare Angles?
Incorrect Labels?
```

For sequences:

```text
Long Context?
Rare Tokens?
Sequence Length?
Ambiguous Examples?
```

Error analysis converts model failures into engineering information.

---

## 1.48 Data Problems vs Model Problems

Not every failure requires a more sophisticated architecture.

Poor performance may result from:

```text
Incorrect Labels
Missing Data
Biased Sampling
Insufficient Examples
Class Imbalance
Data Leakage
Distribution Mismatch
```

Sometimes:

```text
Better Data
>
Bigger Model
```

This is one of the most important practical lessons in machine learning.

---

## 1.49 Class Imbalance

Suppose:

```text
Class A
→ 99%

Class B
→ 1%
```

A model predicting:

```text
Always Class A
```

achieves:

```math
99\%
```

accuracy.

Yet it completely fails to detect Class B.

Therefore, accuracy alone can be misleading.

Useful responses may include:

```text
Precision / Recall / F1

Class Weights

Resampling

Threshold Adjustment

PR-AUC
```

The correct approach depends on the application.

---

## 1.50 Distribution Shift

The data encountered after deployment may differ from training data.

For example:

```text
Training Images
→ High-Quality Studio Images

Deployment Images
→ Mobile Phone Images in Poor Lighting
```

Even a model with excellent test performance may degrade if the deployment distribution changes.

This is called **distribution shift**.

Real-world ML engineering therefore extends beyond training accuracy.

---

## 1.51 Inference

After training, the model enters inference mode.

```text
New Input
    ↓
Same Preprocessing
    ↓
Trained Model
    ↓
Forward Propagation
    ↓
Prediction
```

There is normally:

```text
No Backpropagation
No Gradient Calculation
No Parameter Update
```

Inference is primarily a forward computation.

---

## 1.52 Training vs Inference

| Training | Inference |
|---|---|
| Forward propagation | Forward propagation |
| Loss calculation | Usually no training loss |
| Backpropagation | No backpropagation |
| Gradient calculation | Gradients unnecessary |
| Parameter updates | Parameters fixed |
| Dropout active | Dropout disabled |
| BatchNorm uses batch statistics | BatchNorm uses stored statistics |

The model learned during training is used, not relearned, during ordinary inference.

---

## 1.53 Save More Than the Weights

A deployable ML system may require:

```text
Model Architecture
Model Weights
Preprocessing Rules
Vocabulary
Class Mapping
Normalization Statistics
Hyperparameters
Software / Library Versions
```

Saving only a weight file may not be sufficient to reproduce predictions.

The preprocessing pipeline is part of the model system.

---

## 1.54 Reproducibility

Useful experiment records include:

```text
Dataset Version
Train / Validation / Test Split
Random Seed
Architecture
Optimizer
Learning Rate
Batch Size
Epochs
Regularization
Best Validation Metric
Final Test Metric
```

Without this information, reproducing a successful experiment can become surprisingly difficult.

---

## 1.55 Hardware Matters

Deep learning is computationally intensive because it performs large numbers of matrix and tensor operations.

Common hardware includes:

```text
CPU
GPU
Specialized Accelerators
```

GPUs are particularly effective because many neural-network operations can be parallelized.

Hardware limitations influence practical choices such as:

```text
Batch Size
Model Size
Image Resolution
Sequence Length
Training Duration
```

Engineering constraints therefore affect model design.

---

## 1.56 Debugging a Neural Network

When a network fails to train, debugging should be systematic.

A useful sequence is:

```text
1. Verify Input Data

2. Verify Labels

3. Verify Tensor Shapes

4. Verify Output Layer

5. Verify Loss Function

6. Try to Overfit a Tiny Dataset

7. Inspect Learning Rate

8. Inspect Gradients

9. Add Complexity Only After Basics Work
```

Randomly changing hyperparameters is usually a poor debugging strategy.

---

## 1.57 The Tiny-Dataset Overfit Test

One particularly useful debugging technique is to take a very small number of training examples.

Then attempt to deliberately overfit them.

Conceptually:

```text
Tiny Dataset
      ↓
Train Model
      ↓
Can Training Loss Become Very Small?
```

If the network cannot fit even a tiny dataset, something may be wrong with:

- implementation
- labels
- loss
- architecture
- optimization

A model should usually be able to memorize a sufficiently small training subset if it has adequate capacity.

---

## 1.58 Change One Thing at a Time

Suppose a model performs poorly and we simultaneously change:

```text
Learning Rate
Optimizer
Batch Size
Architecture
Dropout
Initialization
```

If performance improves, we do not know why.

A better experimental process is:

```text
Baseline
 ↓
Change One Important Variable
 ↓
Measure
 ↓
Record
 ↓
Continue
```

Controlled experiments make debugging and tuning much more informative.

---

## 1.59 Deep Learning Is Iterative

A realistic workflow is cyclical:

```text
Train
 ↓
Evaluate
 ↓
Inspect Errors
 ↓
Form Hypothesis
 ↓
Change Model / Data
 ↓
Train Again
```

This is closer to experimental science than to writing a deterministic program once.

Each iteration should answer a question.

For example:

```text
"Is the model underfitting?"

"Does stronger regularization help?"

"Are longer sequences causing failures?"

"Does augmentation improve generalization?"
```

---

## 1.60 Architecture Summary

The architectures studied in this handbook can now be placed side by side.

| Architecture | Core Idea | Best Known For |
|---|---|---|
| Feed-Forward NN | Layered nonlinear transformation | General prediction |
| CNN | Local connectivity + shared filters | Spatial / image data |
| RNN | Recurrent hidden state | Sequential data |
| LSTM | Gated cell-state memory | Longer dependencies |
| GRU | Simplified gated memory | Efficient sequence modelling |

They all remain neural networks trained using the same underlying gradient-based principles.

---

## 1.61 What Changes Across Architectures?

The fundamental training system stays remarkably similar.

```text
Input
 ↓
Parameterized Computation
 ↓
Prediction
 ↓
Loss
 ↓
Backpropagation
 ↓
Optimizer
 ↓
Updated Parameters
```

What changes is the **structure of the parameterized computation**.

### Dense Network

```text
Weighted Connections
```

### CNN

```text
Shared Spatial Filters
```

### RNN

```text
Shared Recurrent State
```

### LSTM / GRU

```text
Gated Recurrent State
```

This is the unifying view of the architectures.

---

## 1.62 The Complete Deep Learning Training Loop

Everything studied so far can be assembled into one process:

```text
Raw Data
   ↓
Preprocessing
   ↓
Train / Validation / Test Split
   ↓
Architecture Selection
   ↓
Parameter Initialization
   ↓
Mini-Batch
   ↓
Forward Propagation
   ↓
Prediction
   ↓
Loss
   ↓
Backpropagation
   ↓
Gradient Management
   ↓
Optimizer
   ↓
Parameter Update
   ↓
Repeat Across Batches
   ↓
Validation
   ↓
Diagnose
   ↓
Tune Hyperparameters
   ↓
Regularize if Needed
   ↓
Select Best Model
   ↓
Final Test Evaluation
   ↓
Inference
```

That is the complete conceptual journey from raw data to a trained neural-network model.

---

## 1.63 The Deep Learning Engineer's Mental Model

When looking at any neural network, ask five questions.

### 1. What goes in?

```text
Input Representation
```

### 2. What transformations happen?

```text
Architecture
Layers
Activations
```

### 3. What comes out?

```text
Output Representation
```

### 4. How is wrongness measured?

```text
Loss Function
```

### 5. How does the network improve?

```text
Backpropagation
+
Optimizer
```

Everything else improves, stabilizes, regularizes, or specializes this core system.

---

## 1.64 Key Takeaways

- Deep-learning development is an iterative engineering workflow rather than a single training operation.
- The problem and data structure should determine architecture choice.
- Neural networks are not automatically superior to classical ML models.
- Training, validation, and test sets serve different purposes.
- Data leakage can make evaluation results misleading.
- Preprocessing parameters should be learned from training data.
- A simple baseline should be established before adding complexity.
- The output layer and loss function must match the task.
- Initialization determines the starting conditions for optimization.
- Backpropagation calculates gradients; the optimizer uses them to update parameters.
- Learning rate is one of the most important training hyperparameters.
- Training and validation curves help diagnose model behaviour.
- High training and validation error can indicate underfitting.
- Low training error with poor validation performance can indicate overfitting.
- Regularization, augmentation, early stopping, and additional data can improve generalization.
- Gradient problems should be diagnosed rather than treated blindly.
- Loss functions and evaluation metrics serve different purposes.
- Classification thresholds need not always be `0.5`.
- Error analysis provides information that a single metric cannot.
- Data quality problems cannot always be solved by increasing model complexity.
- Accuracy can be misleading for imbalanced datasets.
- Deployment data may differ from training data because of distribution shift.
- Inference normally requires only forward propagation.
- Preprocessing artifacts are part of the deployable ML system.
- Reproducibility requires recording data, architecture, hyperparameters, and results.
- Hardware constraints influence practical model design.
- Deliberately overfitting a tiny dataset is a useful debugging technique.
- Controlled experiments are more informative than changing many variables simultaneously.
- Feed-forward networks, CNNs, RNNs, LSTMs, and GRUs use different architectures but share the same fundamental learning mechanism.

### Memory Hook

```text
Deep Learning Workflow:

Problem
 ↓
Data
 ↓
Split
 ↓
Preprocess
 ↓
Baseline
 ↓
Architecture
 ↓
Forward Pass
 ↓
Loss
 ↓
Backprop
 ↓
Optimizer
 ↓
Validation
 ↓
Diagnose
 ↓
Improve
 ↓
Test
 ↓
Inference


If Both Training
and Validation Are Bad:

→ Think Underfitting


If Training Is Great
but Validation Is Bad:

→ Think Overfitting


If Training Is Unstable:

→ Learning Rate
→ Gradients
→ Initialization
→ Normalization


Architecture:

Dense
→ General Relationships

CNN
→ Space

RNN
→ Sequence

LSTM / GRU
→ Sequence + Better Memory


Loss
→ How the Model Learns

Metric
→ How We Judge It


Backpropagation
→ Computes Gradients

Optimizer
→ Uses Gradients


The Engineer's Five Questions:

What Goes In?

What Happens Inside?

What Comes Out?

How Is Error Measured?

How Does It Learn?


Everything in this handbook
fits somewhere inside
those five questions.
```