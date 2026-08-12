# 05. Loss Functions

## 1.1 What Is a Loss Function?

A **loss function** measures how far a model's prediction is from the true target.

Suppose:

```math
y
```

is the true value and:

```math
\hat{y}
```

is the model's prediction.

A loss function computes:

```math
L(y,\hat{y})
```

The result is a numerical measure of prediction error.

Conceptually:

```text
Prediction
   ↓
Compare with Ground Truth
   ↓
Loss
```

A smaller loss generally indicates a better prediction.

---

## 1.2 Why Loss Is Necessary

A neural network cannot improve merely by knowing that a prediction is "wrong."

It needs a numerical signal that tells it **how wrong** the prediction is.

The training process therefore follows:

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss Function
  ↓
Numerical Error
  ↓
Backpropagation
  ↓
Parameter Update
```

The loss function provides the objective that training attempts to minimize.

---

## 1.3 Loss Function vs Evaluation Metric

A **loss function** is used directly during training.

An **evaluation metric** is primarily used to judge model performance.

For example, in regression:

```text
Training Loss
→ Mean Squared Error

Evaluation Metrics
→ RMSE
→ MAE
→ R²
```

In classification:

```text
Training Loss
→ Cross-Entropy

Evaluation Metrics
→ Accuracy
→ Precision
→ Recall
→ F1 Score
```

Sometimes the same quantity can serve both purposes, but the roles are conceptually different.

---

## 1.4 Loss for One Example vs Cost Over a Dataset

For one training example:

```math
L^{(i)}
=
L\left(y^{(i)},\hat{y}^{(i)}\right)
```

For `m` examples, the average loss can be written as:

```math
J
=
\frac{1}{m}
\sum_{i=1}^{m}
L\left(y^{(i)},\hat{y}^{(i)}\right)
```

The notation varies across textbooks.

A common distinction is:

```text
Loss
= error for one example

Cost / Objective
= aggregate loss over many examples
```

In practice, the terms are often used somewhat interchangeably.

---

## 1.5 Mean Squared Error

For regression, one of the most common loss functions is **Mean Squared Error (MSE)**.

For `m` examples:

```math
\operatorname{MSE}
=
\frac{1}{m}
\sum_{i=1}^{m}
\left(
y^{(i)}-\hat{y}^{(i)}
\right)^2
```

For one example:

```math
L
=
(y-\hat{y})^2
```

The error is squared, making all contributions non-negative.

Large errors receive disproportionately larger penalties.

---

## 1.6 Why Square the Error?

Suppose the prediction errors are:

```math
-2,\;2
```

If we simply average them:

```math
\frac{-2+2}{2}=0
```

the errors cancel each other.

Squaring avoids this:

```math
(-2)^2=4
```

and:

```math
2^2=4
```

The squared loss also makes large deviations more costly.

For example:

```math
2^2=4
```

but:

```math
10^2=100
```

Therefore, MSE strongly penalizes large errors.

---

## 1.7 Derivative of Squared Error

Consider:

```math
L=(y-\hat{y})^2
```

The derivative with respect to the prediction is:

```math
\frac{\partial L}{\partial \hat{y}}
=
2(\hat{y}-y)
```

This derivative tells us how changing the prediction would affect the loss.

If:

```math
\hat{y}>y
```

the gradient is positive.

If:

```math
\hat{y}<y
```

the gradient is negative.

This gradient becomes the starting point for backpropagation.

---

## 1.8 Mean Absolute Error

Another regression loss is **Mean Absolute Error (MAE)**:

```math
\operatorname{MAE}
=
\frac{1}{m}
\sum_{i=1}^{m}
\left|
y^{(i)}-\hat{y}^{(i)}
\right|
```

Unlike MSE, MAE does not square the errors.

Therefore, large errors do not receive disproportionately large penalties.

Conceptually:

```text
MSE
→ strongly punishes large errors

MAE
→ treats error magnitude more linearly
```

MAE is generally more robust to outliers than MSE.

---

## 1.9 MSE vs MAE

Consider an error of `10`.

Under MSE:

```math
10^2=100
```

Under MAE:

```math
|10|=10
```

This makes MSE more sensitive to large errors.

A useful comparison is:

| Property | MSE | MAE |
|---|---|---|
| Large errors | Strongly penalized | Linearly penalized |
| Outlier sensitivity | Higher | Lower |
| Smooth derivative | Yes | Not at zero |
| Common use | Regression | Robust regression |

---

## 1.10 Classification Requires a Different Loss

For classification, using ordinary squared error is often not ideal.

Suppose a binary classifier predicts:

```math
\hat{p}=0.9
```

for a sample whose true class is:

```math
y=1
```

The prediction is good.

But if the model predicts:

```math
\hat{p}=0.01
```

for the same example, the model is confidently wrong.

Classification losses should strongly penalize such confident mistakes.

This leads to **cross-entropy loss**.

---

## 1.11 Binary Cross-Entropy

For binary classification:

```math
y\in\{0,1\}
```

and:

```math
0<\hat{p}<1
```

Binary Cross-Entropy (BCE) is:

```math
L
=
-
\left[
y\log(\hat{p})
+
(1-y)\log(1-\hat{p})
\right]
```

This is also called **log loss**.

The formula contains two cases within one expression.

---

## 1.12 Binary Cross-Entropy When the True Class Is 1

If:

```math
y=1
```

then:

```math
1-y=0
```

so the loss becomes:

```math
L=-\log(\hat{p})
```

If the model predicts:

```math
\hat{p}\approx1
```

then:

```math
L\approx0
```

If it predicts:

```math
\hat{p}\approx0
```

then the loss becomes very large.

Therefore:

```text
True class = 1

High predicted probability
→ small loss

Low predicted probability
→ large loss
```

---

## 1.13 Binary Cross-Entropy When the True Class Is 0

If:

```math
y=0
```

then the loss becomes:

```math
L=-\log(1-\hat{p})
```

If:

```math
\hat{p}\approx0
```

the loss is small.

If:

```math
\hat{p}\approx1
```

the loss becomes very large.

Thus, BCE strongly penalizes confident incorrect predictions.

---

## 1.14 Why the Logarithm Is Useful

The logarithm creates a useful penalty shape.

For the correct class:

```math
-\log(\hat{p})
```

behaves approximately as follows:

```text
Predicted Probability    Loss

0.99                     very small
0.90                     small
0.50                     moderate
0.10                     large
0.01                     very large
```

The model is therefore encouraged not only to predict the correct class, but also to assign high probability to it.

---

## 1.15 Sigmoid and Binary Cross-Entropy

Binary classification commonly pairs:

```text
Output Activation
→ Sigmoid

Loss Function
→ Binary Cross-Entropy
```

The sigmoid produces:

```math
\hat{p}
=
\sigma(z)
```

where:

```math
0<\hat{p}<1
```

Binary cross-entropy then compares this predicted probability with the true binary label.

Together:

```text
Raw Output
   ↓
Sigmoid
   ↓
Probability
   ↓
Binary Cross-Entropy
   ↓
Loss
```

---

## 1.16 Categorical Cross-Entropy

For multiclass classification, suppose there are `K` classes.

The true label is represented using one-hot encoding:

```math
\mathbf{y}
=
[y_1,y_2,\ldots,y_K]
```

and the model produces probabilities:

```math
\hat{\mathbf{p}}
=
[\hat{p}_1,\hat{p}_2,\ldots,\hat{p}_K]
```

Categorical cross-entropy is:

```math
L
=
-
\sum_{k=1}^{K}
y_k\log(\hat{p}_k)
```

Because only one entry in the one-hot target is `1`, the expression simplifies to:

```math
L
=
-\log(\hat{p}_{\text{true class}})
```

The model is therefore penalized according to the probability assigned to the correct class.

---

## 1.17 Example of Categorical Cross-Entropy

Suppose the true class is `Dog`.

The one-hot target is:

```text
Cat   = 0
Dog   = 1
Horse = 0
```

and the model predicts:

```text
Cat   = 0.10
Dog   = 0.80
Horse = 0.10
```

The loss becomes:

```math
L=-\log(0.80)
```

If instead the model predicts:

```text
Cat   = 0.70
Dog   = 0.05
Horse = 0.25
```

then:

```math
L=-\log(0.05)
```

which is much larger.

The confident incorrect prediction therefore receives a strong penalty.

---

## 1.18 Softmax and Categorical Cross-Entropy

Multiclass classification commonly pairs:

```text
Output Activation
→ Softmax

Loss Function
→ Categorical Cross-Entropy
```

Softmax converts raw scores into probabilities:

```math
\hat{p}_k
=
\frac{e^{z_k}}
{\sum_j e^{z_j}}
```

such that:

```math
\sum_k \hat{p}_k=1
```

Cross-entropy then compares these probabilities with the true class.

---

## 1.19 Sparse Categorical Cross-Entropy

Categorical cross-entropy often assumes the true target is one-hot encoded.

For example:

```text
Class 2
→ [0, 1, 0, 0]
```

**Sparse categorical cross-entropy** allows the class to be represented directly as an integer:

```text
Class 2
→ 1
```

The underlying mathematical objective is essentially the same.

The difference is mainly how target labels are represented.

---

## 1.20 Loss for Multi-Label Classification

In multi-label classification, several labels may be true simultaneously.

For example:

```text
Person = 1
Car    = 1
Dog    = 0
Tree   = 1
```

Each label is treated as an independent binary classification problem.

Therefore, the output layer commonly uses multiple sigmoid units:

```text
Label 1 → Sigmoid
Label 2 → Sigmoid
Label 3 → Sigmoid
...
```

with binary cross-entropy applied independently across labels.

---

## 1.21 Loss Defines the Training Objective

The neural network contains parameters:

```math
\theta
=
\{
W^{[1]},
b^{[1]},
W^{[2]},
b^{[2]},
\ldots
\}
```

The loss therefore depends indirectly on these parameters:

```math
J(\theta)
```

Training attempts to find:

```math
\theta^*
```

such that:

```math
\theta^*
=
\operatorname*{arg\,min}_{\theta}
J(\theta)
```

In words:

> Find the parameter values that minimize the loss.

This is the optimization problem underlying neural-network training.

---

## 1.22 The Loss Landscape

Because the loss depends on many weights and biases, it can be imagined as a surface over parameter space.

For a highly simplified case with only two parameters:

```math
J(w_1,w_2)
```

we can imagine:

```text
High Loss
   /\        /\
  /  \______/  \
 /              \
        ↓
     Minimum
```

Real neural networks may contain millions or billions of parameters, so the true loss landscape exists in an extremely high-dimensional space.

Gradient-based optimization attempts to move through this space toward regions of lower loss.

---

## 1.23 Loss and Gradient

The central training question is:

```text
If I change this weight slightly,
what happens to the loss?
```

Mathematically:

```math
\frac{\partial L}{\partial w}
```

If the derivative is positive, increasing the weight increases the loss.

If the derivative is negative, increasing the weight decreases the loss.

Therefore, the gradient tells us which direction changes the loss.

This connects loss functions directly to gradient descent.

---

## 1.24 From Loss to Parameter Update

Suppose:

```math
\frac{\partial L}{\partial w}
```

is the gradient of the loss with respect to a weight.

Gradient descent updates the weight as:

```math
w_{\text{new}}
=
w_{\text{old}}
-
\eta
\frac{\partial L}{\partial w}
```

where:

```math
\eta
```

is the learning rate.

The negative sign means the parameter is moved in the direction that reduces the loss.

Thus:

```text
Prediction
   ↓
Loss
   ↓
Gradient
   ↓
Parameter Update
```

---

## 1.25 Loss Over a Mini-Batch

Neural networks commonly train using mini-batches.

Suppose a batch contains `B` examples.

The batch loss may be calculated as:

```math
J_{\text{batch}}
=
\frac{1}{B}
\sum_{i=1}^{B}
L^{(i)}
```

The gradients are then calculated from this aggregate loss.

The optimizer updates the model after each batch.

This is the basis of mini-batch gradient descent.

---

## 1.26 Regularization and the Objective Function

The training objective may include more than prediction loss.

For example:

```math
J
=
L_{\text{data}}
+
\lambda R(\theta)
```

where:

- `L_data` measures prediction error
- `R(θ)` is a regularization penalty
- `λ` controls the strength of regularization

This is directly related to the regularization concepts used in classical machine learning.

For example, L2 regularization adds a penalty related to squared weight magnitude.

---

## 1.27 Common Output-Loss Pairings

A practical summary is:

| Task | Output Activation | Typical Loss |
|---|---|---|
| Regression | Linear | MSE or MAE |
| Binary classification | Sigmoid | Binary Cross-Entropy |
| Multiclass classification | Softmax | Categorical Cross-Entropy |
| Multi-label classification | Multiple Sigmoids | Binary Cross-Entropy |

These pairings are common because the output representation and loss function complement each other mathematically.

---

## 1.28 Why Not Use Accuracy as the Loss?

Accuracy is useful for evaluating classification performance, but it is not suitable as a training loss.

Suppose the classification threshold is `0.5`.

Predictions:

```text
0.51 → class 1
0.99 → class 1
```

Both produce exactly the same accuracy result.

But:

```text
0.99
```

represents much greater confidence than:

```text
0.51
```

Accuracy also changes discontinuously when a prediction crosses the classification threshold.

Gradient-based optimization requires a function that changes smoothly enough to provide useful derivative information.

Cross-entropy provides that training signal.

---

## 1.29 Loss Functions and Maximum Likelihood

Cross-entropy is not an arbitrary formula chosen merely because it works.

It has a strong probabilistic interpretation.

For classification, minimizing cross-entropy corresponds closely to **maximizing the likelihood of the observed training labels** under the model.

Conceptually:

```text
Maximum Likelihood
        ↓
Make observed labels highly probable
        ↓
Equivalent optimization form
        ↓
Minimize Negative Log-Likelihood
        ↓
Cross-Entropy
```

This connects neural-network classification directly to probability and statistics.

---

## 1.30 The Complete Training Connection

Forward propagation gives:

```math
\hat{y}
=
F(\mathbf{x};\theta)
```

The loss function gives:

```math
L(y,\hat{y})
```

Combining them:

```math
L
\left(
y,
F(\mathbf{x};\theta)
\right)
```

The loss therefore depends on every parameter that contributed to the prediction.

Backpropagation calculates:

```math
\frac{\partial L}{\partial \theta}
```

and the optimizer uses these gradients to update the parameters.

This completes the conceptual bridge:

```text
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
Optimizer
        ↓
Updated Parameters
```

---

## 1.31 Key Takeaways

- A loss function measures the error between prediction and ground truth.
- Training attempts to minimize the loss.
- Loss is used directly for optimization; evaluation metrics primarily measure model performance.
- MSE and MAE are common regression losses.
- MSE penalizes large errors more strongly than MAE.
- Binary cross-entropy is commonly used for binary classification.
- Categorical cross-entropy is commonly used for multiclass classification.
- Cross-entropy strongly penalizes confident incorrect predictions.
- Sigmoid naturally pairs with binary cross-entropy.
- Softmax naturally pairs with categorical cross-entropy.
- Sparse categorical cross-entropy differs mainly in target-label representation.
- Multi-label classification commonly uses independent sigmoid outputs with binary cross-entropy.
- The loss can be viewed as a function of all trainable parameters.
- Gradients describe how changing parameters affects the loss.
- Gradient-based optimization updates parameters to reduce the loss.
- Cross-entropy has a probabilistic interpretation through maximum likelihood.

### Memory Hook

```text
Prediction
   ↓
Compare with Truth
   ↓
Loss
   ↓
Gradient
   ↓
Parameter Update

Regression
→ MSE / MAE

Binary Classification
→ Sigmoid + BCE

Multiclass Classification
→ Softmax + Cross-Entropy

Multi-label Classification
→ Multiple Sigmoids + BCE

Loss answers:
"How wrong are we?"

Gradient answers:
"Which way reduces that wrongness?"
```