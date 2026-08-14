# 12. Regularization in Neural Networks

## 1.1 Why Regularization Is Needed

Neural networks can contain very large numbers of trainable parameters.

A sufficiently large network may fit the training data extremely well, including:

- useful patterns
- noise
- accidental correlations
- training-specific irregularities

This can lead to **overfitting**.

Conceptually:

```text
Training Performance
→ Excellent

Validation Performance
→ Poor
```

The network has learned the training set better than it has learned the underlying general pattern.

**Regularization** refers to techniques used to reduce overfitting and improve generalization.

---

## 1.2 The Goal of Regularization

Without regularization, training primarily minimizes prediction loss:

```math
J(\theta)
=
L_{\text{data}}(\theta)
```

Regularization modifies the training process so that the model is discouraged from becoming unnecessarily complex.

A common form is:

```math
J(\theta)
=
L_{\text{data}}(\theta)
+
\lambda R(\theta)
```

where:

- `L_data` measures prediction error
- `R(θ)` is the regularization term
- `λ` controls regularization strength

Thus:

```text
Training Objective
=
Fit the Data
+
Control Complexity
```

---

## 1.3 Connection to Classical Machine Learning

The same idea appeared in classical machine learning.

For example:

```text
Linear Regression
→ Ridge
→ L2 Regularization

Linear Regression
→ Lasso
→ L1 Regularization
```

The principle remains the same in neural networks.

We deliberately constrain the model slightly so that it is less likely to memorize the training data.

This introduces a useful trade-off:

```text
Slightly Worse Training Fit
        ↓
Potentially Better Generalization
```

---

## 1.4 L2 Regularization

**L2 regularization** penalizes large weight values.

A common objective is:

```math
J
=
L_{\text{data}}
+
\frac{\lambda}{2}
\sum_i w_i^2
```

For an entire weight matrix:

```math
J
=
L_{\text{data}}
+
\frac{\lambda}{2}
\|W\|_2^2
```

The model is therefore encouraged to learn smaller weights unless large weights substantially improve prediction.

---

## 1.5 Why Penalize Large Weights?

Large weights can make a model highly sensitive to small changes in its inputs.

Suppose:

```math
z
=
w_1x_1+w_2x_2+b
```

If:

```math
|w_1|
```

is extremely large, even a small change in:

```math
x_1
```

can strongly change the output.

Regularization encourages the model to spread explanatory responsibility across parameters rather than depending excessively on a few very large values.

Conceptually:

```text
Very Large Weights
→ Highly Sensitive Model
→ Greater Risk of Fitting Noise

Moderate Weights
→ Smoother Behaviour
→ Better Generalization Potential
```

---

## 1.6 Effect of L2 on the Gradient

Suppose:

```math
J
=
L_{\text{data}}
+
\frac{\lambda}{2}w^2
```

Then:

```math
\frac{\partial J}{\partial w}
=
\frac{\partial L_{\text{data}}}{\partial w}
+
\lambda w
```

The update becomes:

```math
w
\leftarrow
w
-
\eta
\left(
\frac{\partial L_{\text{data}}}{\partial w}
+
\lambda w
\right)
```

Rearranging:

```math
w
\leftarrow
(1-\eta\lambda)w
-
\eta
\frac{\partial L_{\text{data}}}{\partial w}
```

The existing weight is slightly reduced before the ordinary gradient update is applied.

This explains the connection between L2 regularization and **weight decay**.

---

## 1.7 Weight Decay

Weight decay gradually reduces parameter magnitude during optimization.

Conceptually:

```text
Current Weight
     ↓
Small Shrinkage
     ↓
Gradient Update
```

A simplified update is:

```math
w
\leftarrow
(1-\eta\lambda)w
-
\eta
\frac{\partial L}{\partial w}
```

For plain SGD, L2 regularization and weight decay are closely related.

With adaptive optimizers such as Adam, the relationship is more subtle.

This is why optimizers such as **AdamW** explicitly decouple weight decay from the gradient calculation.

---

## 1.8 L1 Regularization

**L1 regularization** penalizes the absolute magnitude of weights.

The objective becomes:

```math
J
=
L_{\text{data}}
+
\lambda
\sum_i |w_i|
```

Unlike L2, L1 tends to encourage some weights to become exactly or approximately zero.

Therefore:

```text
L1
→ Encourages Sparsity

L2
→ Encourages Small Distributed Weights
```

---

## 1.9 L1 vs L2

A useful comparison is:

| Property | L1 | L2 |
|---|---|---|
| Penalty | Absolute weight | Squared weight |
| Encourages small weights | Yes | Yes |
| Encourages exact zeros | More strongly | Less strongly |
| Produces sparse models | Often | Usually not |
| Common in neural networks | Less common | Very common |

L2-style regularization is generally more common in deep neural networks.

---

## 1.10 Regularization Strength

The hyperparameter:

```math
\lambda
```

controls the strength of regularization.

If:

```math
\lambda=0
```

there is no regularization penalty.

If `λ` is small:

```text
Prediction Loss Dominates
→ Weak Regularization
```

If `λ` is large:

```text
Penalty Dominates
→ Strong Regularization
```

The correct value must balance model fit and generalization.

---

## 1.11 Too Little Regularization

If regularization is too weak:

```text
Large Model Capacity
        ↓
Training Data Fit Becomes Extremely Tight
        ↓
Noise May Be Learned
        ↓
Overfitting
```

Typical pattern:

```text
Training Loss
→ Very Low

Validation Loss
→ Much Higher
```

Increasing regularization may improve validation performance.

---

## 1.12 Too Much Regularization

Regularization can also be excessive.

If:

```math
\lambda
```

is too large, the optimizer may be strongly discouraged from learning useful parameter values.

Then:

```text
Model Becomes Too Constrained
        ↓
Cannot Fit Important Patterns
        ↓
Underfitting
```

Typical pattern:

```text
Training Loss
→ High

Validation Loss
→ High
```

Regularization therefore introduces a balance.

---

## 1.13 Bias-Variance Connection

Regularization can be understood using the **bias-variance trade-off**.

An overly flexible model may have:

```text
Low Bias
High Variance
```

It fits the training data extremely well but changes strongly with different training samples.

Regularization reduces flexibility.

This may increase bias slightly but reduce variance.

Conceptually:

```text
No Regularization
→ Low Bias
→ High Variance
→ Overfitting Risk

Moderate Regularization
→ Slightly Higher Bias
→ Lower Variance
→ Better Generalization
```

---

## 1.14 Early Stopping as Regularization

Early stopping can also act as a form of regularization.

During training:

```text
Training Loss
→ continues decreasing
```

while eventually:

```text
Validation Loss
→ begins increasing
```

Stopping near the best validation point prevents the model from continuing to specialize excessively to the training data.

Therefore:

```text
Early Stopping
→ limits effective training complexity
→ reduces overfitting
```

Although no explicit penalty is added to the loss, the effect is regularizing.

---

## 1.15 Data Augmentation

Another regularization strategy is **data augmentation**.

Instead of constraining the model directly, we increase the effective diversity of the training data.

For images, this might include:

```text
Original Image
   ↓
Flip
Rotate
Crop
Translate
Brightness Change
```

The network sees multiple plausible variations of the same underlying example.

This makes memorization more difficult and encourages more general features.

---

## 1.16 Why Data Augmentation Helps

Suppose a model only sees one exact image of an object.

It may learn details specific to that image.

If the model sees many transformed versions:

```text
Same Object
→ different position
→ different orientation
→ different crop
→ different lighting
```

it is forced to learn more stable features.

Therefore:

```text
Data Augmentation
→ More Training Diversity
→ Less Memorization
→ Better Generalization
```

The valid transformations depend on the problem domain.

---

## 1.17 Adding More Training Data

One of the strongest defenses against overfitting is simply obtaining more representative training data.

Conceptually:

```text
More Data
     ↓
More Examples of True Variation
     ↓
Harder to Memorize Individual Samples
     ↓
Better Generalization Potential
```

Regularization techniques are particularly important when model capacity is large relative to available data.

---

## 1.18 Model Capacity as a Form of Regularization

Architecture itself affects overfitting.

Consider:

```text
Network A
10 → 16 → 1
```

versus:

```text
Network B
10 → 1024 → 1024 → 1024 → 1
```

The second network has vastly greater capacity.

If the task and dataset are small, that capacity may be unnecessary.

Therefore:

```text
Smaller / Simpler Architecture
→ Implicit Complexity Control
```

Choosing an appropriately sized network is itself an important form of regularization.

---

## 1.19 Dropout

**Dropout** is one of the most important neural-network-specific regularization techniques.

During training, some neuron outputs are randomly set to zero.

Conceptually:

```text
Normal Layer

○ ○ ○ ○ ○ ○

During Dropout

○ × ○ ○ × ○
```

where `×` represents temporarily deactivated neurons.

Because the set of dropped neurons changes during training, the network cannot rely too heavily on any single path.

Dropout deserves its own chapter because its training and inference behaviour requires careful explanation.

---

## 1.20 Why Dropout Regularizes

Without dropout, neurons may become strongly dependent on particular other neurons.

This is sometimes described as **co-adaptation**.

With dropout:

```text
Neuron A Cannot Assume
Neuron B Will Always Be Present
```

because neuron B may disappear on the next training iteration.

The network is therefore encouraged to learn more robust, distributed representations.

Conceptually:

```text
Randomly Remove Units
        ↓
Prevent Excessive Dependence
        ↓
More Robust Features
        ↓
Better Generalization
```

---

## 1.21 Batch Normalization and Regularization

Batch normalization is primarily a training-stability and normalization technique.

However, the variation introduced by mini-batch statistics can also provide a mild regularizing effect.

Thus batch normalization may sometimes reduce the need for other regularization.

Its primary purpose, however, should not be thought of simply as:

```text
Batch Normalization
= Regularization
```

It serves broader optimization and activation-stability roles.

---

## 1.22 Noise as Regularization

Adding small amounts of noise during training can sometimes improve generalization.

Examples include noise applied to:

- input values
- hidden activations
- weights
- gradients

The principle is:

```text
Perfectly Memorizing Exact Training Values
        ↓
Becomes Harder

Learning Stable Patterns
        ↓
Becomes More Valuable
```

Dropout can itself be viewed as a structured form of noise injection.

---

## 1.23 Label Smoothing

For multiclass classification, the true one-hot target might normally be:

```math
\begin{bmatrix}
0 & 0 & 1 & 0
\end{bmatrix}
```

With **label smoothing**, it may instead become approximately:

```math
\begin{bmatrix}
0.03 & 0.03 & 0.91 & 0.03
\end{bmatrix}
```

The model is discouraged from becoming excessively confident.

Conceptually:

```text
Hard Target
→ "Correct class must have probability 1"

Smoothed Target
→ "Correct class should dominate,
   but absolute certainty is unnecessary"
```

This can improve generalization in some classification tasks.

---

## 1.24 Regularization and Training Loss

A regularized model may have slightly worse training loss than an unregularized model.

That is not necessarily a failure.

For example:

```text
Model A
Training Accuracy   = 100%
Validation Accuracy = 82%

Model B
Training Accuracy   = 95%
Validation Accuracy = 91%
```

Model B is clearly more useful despite fitting the training data less perfectly.

Therefore:

```text
Best Training Performance
≠
Best Model
```

Generalization matters.

---

## 1.25 Validation Data Guides Regularization

Regularization strength should be selected using validation performance.

A typical process is:

```text
Train with λ₁
→ Measure Validation Performance

Train with λ₂
→ Measure Validation Performance

Train with λ₃
→ Measure Validation Performance

Choose Best Validation Result
```

The test set should remain untouched during this selection process.

---

## 1.26 Regularization Does Not Fix Everything

Poor validation performance is not always caused by overfitting.

For example:

```text
Training Performance Poor
+
Validation Performance Poor
```

suggests underfitting rather than excessive variance.

Adding stronger regularization in this situation may make things worse.

Before regularizing more heavily, identify whether the actual problem is:

```text
Underfitting
or
Overfitting
```

---

## 1.27 Diagnosing the Situation

A useful simplified diagnostic is:

| Training Performance | Validation Performance | Likely Situation |
|---|---|---|
| Poor | Poor | Underfitting |
| Excellent | Much poorer | Overfitting |
| Good | Good and similar | Healthy generalization |

Regularization primarily targets the second case.

---

## 1.28 Common Regularization Techniques

Important techniques include:

```text
L2 Regularization / Weight Decay
L1 Regularization
Dropout
Early Stopping
Data Augmentation
More Training Data
Appropriate Model Size
Label Smoothing
Noise Injection
```

Not every model needs all of them.

The best choice depends on:

- architecture
- dataset size
- data type
- amount of overfitting
- computational constraints

---

## 1.29 Combining Regularization Techniques

Regularization techniques are often combined.

For example:

```text
ReLU Network
+
He Initialization
+
AdamW
+
Weight Decay
+
Data Augmentation
+
Early Stopping
```

However, blindly stacking every possible regularizer is not desirable.

Too much regularization can prevent useful learning.

The goal is:

```text
Enough Constraint
to Reduce Overfitting

but

Enough Capacity
to Learn the True Pattern
```

---

## 1.30 Regularization in the Training Objective

The conceptual training objective becomes:

```math
\text{Total Objective}
=
\text{Prediction Error}
+
\text{Complexity Penalty}
```

or:

```math
J(\theta)
=
L_{\text{data}}(\theta)
+
\lambda R(\theta)
```

Training still follows:

```text
Forward Propagation
        ↓
Prediction
        ↓
Regularized Loss
        ↓
Backpropagation
        ↓
Gradients
        ↓
Optimizer
        ↓
Updated Parameters
```

Regularization changes what the optimizer is encouraged to consider a good solution.

---

## 1.31 Key Takeaways

- Regularization reduces overfitting and improves generalization.
- Neural networks are particularly prone to overfitting because they can contain very large numbers of parameters.
- Regularization may modify the objective:

```math
J
=
L_{\text{data}}
+
\lambda R(\theta)
```

- L2 regularization penalizes squared weight magnitude.
- L2 encourages smaller, more distributed weights.
- Weight decay is closely related to L2 regularization.
- L1 regularization encourages sparse weights and can push some parameters toward zero.
- `λ` controls regularization strength.
- Too little regularization can permit overfitting.
- Too much regularization can cause underfitting.
- Regularization trades a small increase in bias for potentially lower variance.
- Early stopping can act as implicit regularization.
- Data augmentation increases effective training diversity.
- More representative training data is one of the strongest ways to combat overfitting.
- Appropriate model capacity helps control complexity.
- Dropout randomly disables neurons during training and is an important neural-network-specific regularizer.
- Label smoothing discourages excessive classification confidence.
- Training performance may become slightly worse while validation performance improves.
- Validation data should guide regularization choices.
- Regularization should target overfitting, not be applied blindly to every poorly performing model.

### Memory Hook

```text
Regularization
= Control Model Complexity

Goal:
Not Best Training Fit
but Best Generalization

L2
→ Penalize Large Weights
→ Small Distributed Weights

L1
→ Encourage Sparse Weights

Weight Decay
→ Gradually Shrink Weights

Early Stopping
→ Stop Before Memorization

Data Augmentation
→ More Training Variation

Dropout
→ Randomly Remove Neurons During Training

Too Little Regularization
→ Overfitting

Too Much Regularization
→ Underfitting

Core Trade-Off:

Fit the Pattern
without
Memorizing the Noise
```
