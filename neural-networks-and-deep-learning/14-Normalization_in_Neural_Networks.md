# 14. Normalization in Neural Networks

## 1.1 Why Normalization Matters

During training, values flowing through a neural network can have very different scales.

Consider a hidden layer:

```math
z^{[l]}
=
W^{[l]}a^{[l-1]}
+
b^{[l]}
```

As parameters change during training, the distribution of:

```math
z^{[l]}
```

can also change.

Poorly scaled activations can make optimization more difficult.

Normalization techniques attempt to keep intermediate values in more controlled numerical ranges.

Conceptually:

```text
Uncontrolled Activations
        ↓
Optimization Becomes Difficult
        ↓
Slower / Less Stable Training

Normalization
        ↓
Better-Controlled Activations
        ↓
Easier Optimization
```

---

## 1.2 Normalization Before Neural Networks

Normalization is not unique to neural networks.

We already encounter it when preprocessing input features.

For example, standardization transforms a feature:

```math
x'
=
\frac{x-\mu}{\sigma}
```

where:

```math
\mu
```

is the mean and:

```math
\sigma
```

is the standard deviation.

The transformed feature has approximately:

```math
\mu_{x'}=0
```

and:

```math
\sigma_{x'}=1
```

Neural-network normalization techniques extend a similar principle to values **inside the network**.

---

## 1.3 Input Normalization vs Internal Normalization

These should not be confused.

### Input Normalization

Applied before data enters the network:

```text
Raw Features
     ↓
Standardization / Scaling
     ↓
Neural Network
```

### Internal Normalization

Applied to intermediate values:

```text
Hidden Layer Computation
        ↓
Normalization
        ↓
Activation / Next Computation
```

Both can be useful simultaneously.

---

## 1.4 Why Scale Affects Optimization

Suppose one parameter operates on values around:

```math
0.001
```

while another operates on values around:

```math
10000
```

The resulting gradients can have very different scales.

This can make optimization behave unevenly.

A better-controlled numerical scale can produce a smoother optimization process.

Conceptually:

```text
Wildly Different Scales
→ Uneven Gradient Behaviour
→ Harder Optimization

Controlled Scales
→ Better Gradient Behaviour
→ Easier Optimization
```

---

## 1.5 Batch Normalization

**Batch Normalization**, commonly called **BatchNorm**, normalizes activations using statistics calculated from a mini-batch.

Suppose a mini-batch contains values:

```math
x_1,x_2,\ldots,x_m
```

BatchNorm first calculates the batch mean:

```math
\mu_B
=
\frac{1}{m}
\sum_{i=1}^{m}x_i
```

and batch variance:

```math
\sigma_B^2
=
\frac{1}{m}
\sum_{i=1}^{m}
(x_i-\mu_B)^2
```

---

## 1.6 Normalizing the Batch

Each value is then normalized:

```math
\hat{x}_i
=
\frac{x_i-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
```

where:

```math
\epsilon
```

is a small constant used for numerical stability.

The normalized values have approximately:

```math
\text{mean}\approx0
```

and:

```math
\text{variance}\approx1
```

within the mini-batch.

---

## 1.7 Why Not Stop at Mean Zero and Variance One?

Forcing every layer permanently to mean zero and variance one could unnecessarily restrict what the network can represent.

Therefore, BatchNorm introduces two trainable parameters:

```math
\gamma
```

and:

```math
\beta
```

The final output becomes:

```math
y_i
=
\gamma\hat{x}_i+\beta
```

where:

```math
\gamma
```

controls scaling and:

```math
\beta
```

controls shifting.

These parameters are learned through backpropagation.

---

## 1.8 Scale and Shift

BatchNorm therefore performs:

```text
Original Activation
       ↓
Subtract Batch Mean
       ↓
Divide by Batch Standard Deviation
       ↓
Normalized Activation
       ↓
Multiply by γ
       ↓
Add β
       ↓
Final Output
```

Mathematically:

```math
x
\rightarrow
\hat{x}
\rightarrow
\gamma\hat{x}+\beta
```

The network receives the benefits of normalization without losing the ability to learn an appropriate scale and offset.

---

## 1.9 Trainable vs Non-Trainable Quantities

BatchNorm contains different kinds of quantities.

### Learned Through Backpropagation

```math
\gamma,\beta
```

### Calculated from Data

During training:

```math
\mu_B,\sigma_B^2
```

are calculated from the current mini-batch.

Frameworks also maintain running estimates of the mean and variance for use during inference.

This distinction becomes important when switching between training and evaluation modes.

---

## 1.10 BatchNorm During Training

During training, BatchNorm uses statistics from the current mini-batch.

```text
Mini-Batch
    ↓
Calculate Batch Mean
    ↓
Calculate Batch Variance
    ↓
Normalize
    ↓
Scale and Shift
```

Therefore, the output for one example can depend slightly on the other examples present in the same mini-batch.

---

## 1.11 BatchNorm During Inference

During inference, we may predict only one example:

```text
One New Sample
```

Calculating useful batch statistics may then be impossible or unreliable.

Therefore, BatchNorm normally uses **running estimates** accumulated during training.

Conceptually:

```text
Training
→ Current Mini-Batch Statistics

Inference
→ Stored Running Statistics
```

This is why training mode and evaluation mode matter when using BatchNorm.

---

## 1.12 Running Mean and Variance

During training, frameworks maintain estimates such as:

```math
\mu_{\text{running}}
```

and:

```math
\sigma_{\text{running}}^2
```

These are updated gradually from training batches.

A simplified running-mean update might look like:

```math
\mu_{\text{running}}
\leftarrow
\alpha\mu_{\text{running}}
+
(1-\alpha)\mu_B
```

A similar process is used for variance.

These stored statistics approximate the broader training distribution.

---

## 1.13 Why BatchNorm Helps Training

BatchNorm can make training easier by keeping intermediate activation scales more controlled.

Benefits can include:

- faster convergence
- greater training stability
- reduced sensitivity to initialization
- ability to use larger learning rates in some cases
- improved gradient flow

It does not eliminate the need for good initialization or sensible optimization, but it can make the network more forgiving.

---

## 1.14 BatchNorm and Gradient Flow

Backpropagation depends on derivatives passing through many layers.

If activation scales become extreme, gradients may also behave poorly.

By keeping activations in more manageable ranges, BatchNorm can help maintain useful gradient flow.

Conceptually:

```text
Controlled Activations
        ↓
More Stable Derivatives
        ↓
More Stable Gradients
        ↓
Easier Training
```

This connects normalization to the vanishing and exploding gradient problems.

---

## 1.15 BatchNorm and Activation Functions

A common arrangement is:

```text
Linear / Dense Layer
        ↓
Batch Normalization
        ↓
Activation Function
```

For example:

```text
Dense
 ↓
BatchNorm
 ↓
ReLU
```

This allows BatchNorm to normalize the pre-activation values before ReLU is applied.

However, architecture conventions can vary.

The important point is that normalization becomes part of the computational graph.

---

## 1.16 BatchNorm Has Trainable Parameters

BatchNorm is not merely a fixed mathematical transformation.

Because:

```math
y
=
\gamma\hat{x}
+
\beta
```

the parameters:

```math
\gamma
```

and:

```math
\beta
```

are learned.

Backpropagation therefore computes:

```math
\frac{\partial J}{\partial \gamma}
```

and:

```math
\frac{\partial J}{\partial \beta}
```

along with gradients for the surrounding network parameters.

---

## 1.17 BatchNorm and Mini-Batch Size

BatchNorm relies on mini-batch statistics.

If the batch is extremely small, estimates of:

```math
\mu_B
```

and:

```math
\sigma_B^2
```

can become noisy.

For example:

```text
Batch Size = 256
→ Statistics based on many examples

Batch Size = 2
→ Statistics may fluctuate strongly
```

This is an important limitation of BatchNorm.

Other normalization techniques can work better when batch sizes are very small.

---

## 1.18 BatchNorm Has a Mild Regularizing Effect

Different mini-batches produce slightly different:

```math
\mu_B
```

and:

```math
\sigma_B^2
```

Therefore, the normalized representation of an example varies somewhat depending on the batch.

This introduces noise during training.

That noise can produce a mild regularizing effect.

Thus BatchNorm may sometimes reduce overfitting slightly.

However:

```text
BatchNorm
≠ Primarily a Regularization Technique
```

Its central role is improving optimization and training behaviour.

---

## 1.19 BatchNorm vs Dropout

Both can influence generalization, but their primary purposes differ.

| Batch Normalization | Dropout |
|---|---|
| Normalizes activations | Randomly removes activations |
| Primarily improves optimization | Primarily reduces overfitting |
| Uses batch statistics | Uses random masks |
| Has trainable γ and β | Usually has no trainable parameters |
| Behaviour differs during inference | Disabled during ordinary inference |

They can be used together, although not every architecture requires both.

---

## 1.20 Layer Normalization

**Layer Normalization**, or **LayerNorm**, uses a different normalization strategy.

Instead of calculating statistics across examples in a mini-batch, it calculates statistics across features of an individual example.

Suppose one example produces:

```math
\mathbf{x}
=
\begin{bmatrix}
x_1 & x_2 & \cdots & x_d
\end{bmatrix}
```

LayerNorm computes:

```math
\mu
=
\frac{1}{d}
\sum_{j=1}^{d}x_j
```

and:

```math
\sigma^2
=
\frac{1}{d}
\sum_{j=1}^{d}
(x_j-\mu)^2
```

---

## 1.21 LayerNorm Transformation

Each feature is normalized:

```math
\hat{x}_j
=
\frac{x_j-\mu}
{\sqrt{\sigma^2+\epsilon}}
```

and then scaled and shifted:

```math
y_j
=
\gamma_j\hat{x}_j+\beta_j
```

As with BatchNorm:

```math
\gamma
```

and:

```math
\beta
```

are trainable.

The major difference is **where the mean and variance are calculated**.

---

## 1.22 BatchNorm vs LayerNorm

The conceptual difference is:

```text
BatchNorm
→ Compare a feature across examples in the batch

LayerNorm
→ Compare features within one example
```

Suppose:

```text
Batch:

Example 1 → [x₁ x₂ x₃ x₄]
Example 2 → [x₁ x₂ x₃ x₄]
Example 3 → [x₁ x₂ x₃ x₄]
```

BatchNorm typically normalizes along the batch dimension for each feature.

LayerNorm typically normalizes across the features within each row.

---

## 1.23 Why LayerNorm Does Not Depend on Batch Size

LayerNorm calculates statistics independently for each example.

Therefore:

```text
Batch Size = 128
or
Batch Size = 1
```

does not fundamentally change its normalization mechanism.

This makes LayerNorm useful in architectures where:

- batches may be small
- sequence lengths vary
- examples are processed individually
- batch statistics are inconvenient

---

## 1.24 LayerNorm in Modern Deep Learning

LayerNorm is especially important in sequence models.

It is widely used in:

```text
Transformers
```

and is also useful in other architectures.

Transformer blocks commonly contain LayerNorm around attention and feed-forward components.

Therefore, although BatchNorm became famous through convolutional and feed-forward networks, LayerNorm is central to many modern language and generative models.

---

## 1.25 BatchNorm and CNNs

BatchNorm has historically been widely used in convolutional neural networks.

Conceptually:

```text
Convolution
    ↓
BatchNorm
    ↓
ReLU
    ↓
Next Convolution
```

Normalization can make deep convolutional networks substantially easier to train.

The exact dimensions over which statistics are calculated depend on the architecture and framework.

---

## 1.26 Other Normalization Methods

Several additional normalization techniques exist.

Examples include:

```text
Instance Normalization
Group Normalization
RMS Normalization
```

### Instance Normalization

Normalizes each example more independently and is commonly associated with some image-generation and style-transfer applications.

### Group Normalization

Divides channels into groups and normalizes within each group.

It can work well when batch sizes are too small for reliable BatchNorm statistics.

### RMSNorm

Normalizes based primarily on root-mean-square magnitude rather than subtracting the mean.

It is used in several modern transformer architectures.

---

## 1.27 Normalization Is Not the Same as Regularization

The terms sound similar but refer to different ideas.

### Normalization

```text
Control the Scale / Distribution of Values
```

### Regularization

```text
Control Model Complexity / Overfitting
```

Normalization may indirectly provide some regularization, but that is not its defining purpose.

Therefore:

```text
Normalization
≠
Regularization
```

This distinction is worth remembering.

---

## 1.28 Normalization Is Not the Same as Weight Initialization

Initialization determines parameter values **before training starts**.

Normalization modifies values **during forward computation**.

```text
Initialization
→ Starting parameter scale

Normalization
→ Activation scale during computation
```

They address related training-stability problems but at different stages.

---

## 1.29 Normalization Does Not Replace Input Preprocessing

Using BatchNorm or LayerNorm does not automatically mean raw input features should never be scaled.

Input preprocessing can still improve training.

For numerical tabular data, for example:

```text
Raw Inputs
     ↓
Feature Scaling
     ↓
Neural Network
     ↓
Internal Normalization if Used
```

These operations solve different parts of the problem.

---

## 1.30 Training Mode vs Evaluation Mode

BatchNorm makes the distinction between training and evaluation mode especially important.

During training:

```text
BatchNorm
→ Uses Current Batch Statistics
→ Updates Running Statistics
```

During inference:

```text
BatchNorm
→ Uses Stored Running Statistics
```

Dropout also changes behaviour:

```text
Training
→ Dropout ON

Inference
→ Dropout OFF
```

Thus:

```text
Training Mode
and
Evaluation Mode
```

are not merely framework bookkeeping.

They can actually change how the network computes its output.

---

## 1.31 What Happens If Evaluation Mode Is Forgotten?

Suppose a trained network containing BatchNorm and Dropout is left in training mode during prediction.

Then:

```text
Dropout
→ May Continue Randomly Removing Activations

BatchNorm
→ May Use Current Batch Statistics
```

Predictions can become inconsistent or incorrect.

This is why frameworks provide explicit mechanisms for switching a model between training and evaluation behaviour.

---

## 1.32 Normalization and the Training Loop

With normalization, a typical network might perform:

```text
Mini-Batch
    ↓
Dense / Convolution
    ↓
Normalization
    ↓
Activation
    ↓
Next Layer
    ↓
Prediction
    ↓
Loss
    ↓
Backpropagation
    ↓
Optimizer
```

Normalization participates directly in the forward and backward computational graph.

---

## 1.33 Choosing Between BatchNorm and LayerNorm

A simplified practical guide is:

```text
Conventional CNN / Feed-Forward Network
→ BatchNorm often useful

Transformer / Sequence Architecture
→ LayerNorm commonly used

Very Small Batch Sizes
→ LayerNorm or GroupNorm may be preferable
```

These are useful defaults rather than absolute rules.

Architecture design determines the appropriate choice.

---

## 1.34 Key Takeaways

- Normalization helps control numerical scales inside neural networks.
- Input normalization and internal normalization are different operations.
- BatchNorm calculates statistics using a mini-batch.
- BatchNorm first normalizes values using:

```math
\hat{x}
=
\frac{x-\mu_B}
{\sqrt{\sigma_B^2+\epsilon}}
```

- It then learns scale and shift parameters:

```math
y
=
\gamma\hat{x}+\beta
```

- `γ` and `β` are trainable parameters.
- During training, BatchNorm uses mini-batch statistics.
- During inference, BatchNorm normally uses stored running statistics.
- BatchNorm can improve optimization stability and convergence.
- Very small batch sizes can make BatchNorm statistics unreliable.
- BatchNorm can provide a mild secondary regularizing effect.
- LayerNorm calculates statistics across features within an individual example.
- LayerNorm does not fundamentally depend on batch size.
- LayerNorm is widely used in transformers and sequence models.
- GroupNorm can be useful when batch sizes are small.
- Normalization is not the same as regularization.
- Normalization is not the same as weight initialization.
- Internal normalization does not automatically replace input preprocessing.
- Training and evaluation modes matter because layers such as BatchNorm and Dropout behave differently between them.

### Memory Hook

```text
Normalization
→ Control Value Scale

Input Scaling
→ Before the Network

BatchNorm
→ Across the Batch

LayerNorm
→ Across Features of One Example

BatchNorm:

x
↓
Subtract Batch Mean
↓
Divide by Batch Std
↓
Scale by γ
↓
Shift by β

γ and β
→ Learned

BatchNorm Training
→ Batch Statistics

BatchNorm Inference
→ Running Statistics

LayerNorm
→ No Dependence on Batch Statistics

CNN
→ Often BatchNorm

Transformer
→ Often LayerNorm

Normalization
≠ Regularization

Core Idea:

Keep signals numerically well-behaved
so the network is easier to train.
```
