# 02. Activation Functions

## 1. Why Activation Functions Are Necessary

An artificial neuron first computes a weighted linear combination:

```math
z = \mathbf{w}^T\mathbf{x} + b
```

If the neuron simply outputs:

```math
a = z
```

then it performs only a **linear transformation**.

Even if many such layers are stacked together, the overall network remains linear.

For example:

```math
\mathbf{h} = W_1\mathbf{x} + \mathbf{b}_1
```

followed by:

```math
\mathbf{y} = W_2\mathbf{h} + \mathbf{b}_2
```

becomes:

```math
\mathbf{y}
=
W_2W_1\mathbf{x}
+
W_2\mathbf{b}_1
+
\mathbf{b}_2
```

which can still be rewritten as:

```math
\mathbf{y}=W\mathbf{x}+\mathbf{b}
```

Therefore, stacking linear layers alone does not allow the network to learn non-linear relationships.

An **activation function** introduces non-linearity:

```math
a = f(z)
```

so a neuron becomes:

```math
a = f(\mathbf{w}^T\mathbf{x}+b)
```

This allows neural networks to model complex decision boundaries and non-linear relationships.

---

## 2. What Does an Activation Function Do?

An activation function transforms the neuron's weighted input:

```math
z
```

into an output:

```math
a=f(z)
```

Different activation functions behave differently and are suitable for different parts of a neural network.

Common activation functions include:

- Step
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- Softmax

The activation function influences:

- whether the network can model non-linearity
- how gradients flow during training
- how quickly the network learns
- what range of values a neuron can output
- what type of problem an output layer can solve

---

## 3. Step Function

The classical perceptron uses a **step function**.

One common form is:

```math
f(z)
=
\begin{cases}
1, & z \geq 0 \\
0, & z < 0
\end{cases}
```

The output changes abruptly from `0` to `1`.

Conceptually:

```text
z < 0  →  0
z ≥ 0  →  1
```

This works for making a hard binary decision.

However, the step function is unsuitable for modern gradient-based neural-network training.

Its derivative is zero almost everywhere and undefined at the threshold.

Therefore, it does not provide useful gradient information for backpropagation.

---

## 4. Sigmoid Function

The **sigmoid function**, also called the **logistic function**, is:

```math
\sigma(z)=\frac{1}{1+e^{-z}}
```

Its output always lies between:

```math
0 < \sigma(z) < 1
```

Its behaviour is approximately:

```text
Large negative z → output near 0
z = 0            → output 0.5
Large positive z → output near 1
```

Because its output lies between `0` and `1`, sigmoid is useful when the output needs to represent a probability.

For binary classification:

```math
\hat{p}
=
\sigma(\mathbf{w}^T\mathbf{x}+b)
```

can represent:

```math
P(y=1\mid\mathbf{x})
```

This is exactly the same basic mechanism used in logistic regression.

---

## 5. Derivative of Sigmoid

The sigmoid function has a convenient derivative:

```math
\sigma'(z)
=
\sigma(z)(1-\sigma(z))
```

If:

```math
a=\sigma(z)
```

then:

```math
\frac{da}{dz}=a(1-a)
```

This makes differentiation mathematically convenient during backpropagation.

However, sigmoid has an important weakness.

For very large positive or negative values of `z`, the sigmoid curve becomes almost flat.

In these regions:

```math
\sigma'(z)\approx 0
```

This means the gradient becomes extremely small.

This phenomenon contributes to the **vanishing gradient problem**.

---

## 6. Sigmoid Saturation

When:

```math
z \gg 0
```

the sigmoid output approaches:

```math
\sigma(z)\approx 1
```

and when:

```math
z \ll 0
```

the output approaches:

```math
\sigma(z)\approx 0
```

In both cases, the derivative becomes small.

The neuron is then said to be **saturated**.

Conceptually:

```text
Very negative input
        ↓
Sigmoid output ≈ 0
        ↓
Gradient ≈ 0

Very positive input
        ↓
Sigmoid output ≈ 1
        ↓
Gradient ≈ 0
```

When many sigmoid layers are stacked, these small gradients can multiply together during backpropagation and become extremely small.

As a result, earlier layers may learn very slowly.

---

## 7. Tanh Function

The **hyperbolic tangent**, or `tanh`, activation function is:

```math
\tanh(z)
=
\frac{e^z-e^{-z}}
{e^z+e^{-z}}
```

Its output range is:

```math
-1 < \tanh(z) < 1
```

Unlike sigmoid, tanh is **zero-centred**.

```text
Large negative z → output near -1
z = 0            → output 0
Large positive z → output near 1
```

Its derivative is:

```math
\frac{d}{dz}\tanh(z)
=
1-\tanh^2(z)
```

Tanh was historically preferred over sigmoid in many hidden layers because its outputs are centred around zero.

However, it still suffers from saturation and vanishing gradients for large positive or negative inputs.

---

## 8. Sigmoid vs Tanh

Both functions are smooth, non-linear, and S-shaped.

Their main difference is output range:

```text
Sigmoid → 0 to 1
Tanh    → -1 to 1
```

Sigmoid:

```math
0 < \sigma(z) < 1
```

Tanh:

```math
-1 < \tanh(z) < 1
```

Tanh is zero-centred, which can make optimization easier in some situations.

However, both functions can saturate and produce very small gradients.

In modern feed-forward hidden layers, both have largely been replaced by ReLU-family activations.

---

## 9. ReLU

The **Rectified Linear Unit**, or **ReLU**, is one of the most widely used activation functions in deep neural networks.

It is defined as:

```math
\mathrm{ReLU}(z)=\max(0,z)
```

or equivalently:

```math
f(z)
=
\begin{cases}
0, & z \leq 0 \\
z, & z > 0
\end{cases}
```

Its behaviour is simple:

```text
Negative input → 0
Positive input → unchanged
```

Examples:

```text
ReLU(-5) = 0
ReLU(-1) = 0
ReLU(0)  = 0
ReLU(2)  = 2
ReLU(8)  = 8
```

Despite its simplicity, ReLU works extremely well in practice.

---

## 10. Derivative of ReLU

The derivative of ReLU is:

```math
f'(z)
=
\begin{cases}
0, & z < 0 \\
1, & z > 0
\end{cases}
```

At:

```math
z=0
```

the derivative is mathematically undefined.

In practical deep-learning implementations, a convenient value is assigned at this point, typically `0`.

For positive inputs:

```math
f'(z)=1
```

so gradients can propagate without becoming progressively smaller merely because of the activation function.

This is one reason ReLU enabled much deeper networks to train effectively.

---

## 11. Why ReLU Became Popular

ReLU has several advantages.

### 11.1. Computational Simplicity

It requires essentially:

```math
\max(0,z)
```

rather than exponentials such as those used by sigmoid and tanh.

### 11.2. Reduced Vanishing Gradient Problem

For positive inputs:

```math
f'(z)=1
```

so the gradient is not compressed toward zero.

### 11.3. Sparse Activations

Any negative input produces exactly zero.

Therefore, many neurons may be inactive for a given input.

This can produce sparse internal representations.

Because of these properties, ReLU became the standard default activation for many hidden layers.

---

## 12. The Dying ReLU Problem

ReLU also has a weakness.

For negative inputs:

```math
f'(z)=0
```

If a neuron's weighted input becomes negative for essentially all training examples, its output remains:

```math
0
```

and its gradient also remains:

```math
0
```

The neuron may therefore stop updating entirely.

This is known as the **dying ReLU problem**.

Conceptually:

```text
Neuron enters negative region
        ↓
Output = 0
        ↓
Gradient = 0
        ↓
Weights stop updating
        ↓
Neuron remains inactive
```

Poor initialization or excessively large learning rates can increase the likelihood of this problem.

---

## 13. Leaky ReLU

**Leaky ReLU** modifies ReLU by allowing a small negative slope.

It is defined as:

```math
f(z)
=
\begin{cases}
z, & z > 0 \\
\alpha z, & z \leq 0
\end{cases}
```

where `α` is a small positive constant.

For example:

```math
\alpha=0.01
```

would give:

```math
f(z)
=
\begin{cases}
z, & z > 0 \\
0.01z, & z \leq 0
\end{cases}
```

Unlike ordinary ReLU, negative inputs are not mapped exactly to zero.

Therefore, the gradient for negative values becomes:

```math
f'(z)=\alpha
```

rather than zero.

This reduces the risk of neurons becoming permanently inactive.

---

## 14. Softmax

**Softmax** is primarily used in the output layer of a **multiclass classification** network.

Suppose the network produces raw scores:

```math
z_1,z_2,\ldots,z_K
```

for `K` classes.

Softmax converts these scores into probabilities:

```math
P(y=k)
=
\frac{e^{z_k}}
{\sum_{j=1}^{K}e^{z_j}}
```

Each resulting value lies between `0` and `1`:

```math
0 < P(y=k) < 1
```

and all class probabilities sum to `1`:

```math
\sum_{k=1}^{K}P(y=k)=1
```

For example:

```text
Raw network output:

Cat    → 2.4
Dog    → 4.1
Horse  → 1.2

After Softmax:

Cat    → 0.15
Dog    → 0.79
Horse  → 0.06
```

The network can then choose the class with the highest probability.

---

## 15. Sigmoid vs Softmax for Classification

This distinction is extremely important.

### 15.1. Binary Classification

For two mutually exclusive classes, a single output neuron commonly uses sigmoid:

```math
\hat{p}=\sigma(z)
```

This represents the probability of one class.

The other class probability is:

```math
1-\hat{p}
```

### 15.2. Multiclass Classification

For several mutually exclusive classes, the output layer commonly uses softmax:

```math
P(y=k)
=
\frac{e^{z_k}}
{\sum_j e^{z_j}}
```

The outputs collectively sum to `1`.

### 15.3. Multi-label Classification

If several labels can independently be true at the same time, separate sigmoid outputs are commonly used.

For example, one photograph might simultaneously contain:

```text
Person  → yes
Car     → yes
Dog     → no
Tree    → yes
```

These labels are not mutually exclusive.

Therefore:

```text
Binary classification     → Sigmoid
Multiclass classification → Softmax
Multi-label classification → Multiple Sigmoids
```

---

## 16. Activation Functions in Hidden and Output Layers

The activation function used in hidden layers and output layers serves different purposes.

### 16.1. Hidden Layers

The main requirement is introducing useful non-linearity while allowing gradients to propagate effectively.

Common choices include:

```text
ReLU
Leaky ReLU
```

### 16.2. Output Layer

The choice depends strongly on the problem.

| Problem | Typical output activation |
|---|---|
| Regression | Linear / no activation |
| Binary classification | Sigmoid |
| Multiclass classification | Softmax |
| Multi-label classification | Sigmoid for each label |

For ordinary regression:

```math
\hat{y}=z
```

is often appropriate because the output may need to take unrestricted real values.

---

## 17. Linear Activation

A **linear activation function** is simply:

```math
f(z)=z
```

Its derivative is:

```math
f'(z)=1
```

A linear activation is generally not useful in hidden layers because it does not introduce non-linearity.

However, it is useful in output layers for regression.

For example, if a network predicts house prices:

```math
\hat{y}\in\mathbb{R}
```

we may not want to restrict the prediction to a fixed range such as `0` to `1`.

The final layer can therefore output its raw linear value.

---

## 18. Choosing an Activation Function

A useful practical rule is:

```text
Hidden layers
    ↓
ReLU by default

Problem with dead neurons
    ↓
Consider Leaky ReLU

Binary output
    ↓
Sigmoid

Multiclass output
    ↓
Softmax

Multi-label output
    ↓
Independent Sigmoids

Regression output
    ↓
Linear
```

This is not an absolute law, but it is a strong starting point for standard neural-network architectures.

---

## 19. Activation Functions and Gradient Flow

Activation functions are not merely output-shaping mechanisms.

They directly influence **gradient flow during backpropagation**.

Suppose the loss is:

```math
L
```

and a neuron has:

```math
a=f(z)
```

Using the chain rule:

```math
\frac{\partial L}{\partial z}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}
```

Since:

```math
\frac{\partial a}{\partial z}=f'(z)
```

the derivative of the activation function becomes part of the gradient.

Therefore:

```text
Activation Function
        ↓
Determines f'(z)
        ↓
Influences gradient size
        ↓
Influences learning
```

This is why the choice of activation function affects not only what the network can represent, but also how easily it can be trained.

---

## 20. Activation Function Comparison

| Activation | Output Range | Major Strength | Major Weakness | Common Use |
|---|---|---|---|---|
| Step | `0` or `1` | Simple classification | Not gradient-friendly | Classical perceptron |
| Sigmoid | `(0,1)` | Probability-like output | Vanishing gradients | Binary output |
| Tanh | `(-1,1)` | Zero-centred | Vanishing gradients | Some recurrent networks |
| ReLU | `[0,∞)` | Simple and effective | Dying ReLU | Hidden layers |
| Leaky ReLU | `(-∞,∞)` | Reduces dead neurons | Extra parameter/slope choice | Hidden layers |
| Softmax | `(0,1)` across classes | Multiclass probabilities | Mainly output-specific | Multiclass output |
| Linear | `(-∞,∞)` | Unrestricted output | No non-linearity | Regression output |

---

## 21. Key Takeaways

- Activation functions introduce **non-linearity** into neural networks.
- Without non-linear activations, multiple neural-network layers collapse into a single linear transformation.
- The classical perceptron uses a **step function**.
- Sigmoid maps values into the range `(0,1)` and is commonly used for binary-classification outputs.
- Tanh maps values into `(-1,1)` and is zero-centred.
- Sigmoid and tanh can suffer from saturation and vanishing gradients.
- ReLU outputs zero for negative values and passes positive values unchanged.
- ReLU became widely used because it is simple and allows stronger gradient flow for positive inputs.
- ReLU can suffer from the **dying ReLU problem**.
- Leaky ReLU maintains a small gradient for negative values.
- Softmax converts multiple class scores into probabilities that sum to `1`.
- Regression outputs commonly use no activation, equivalent to a linear activation.
- Activation functions affect both the network's expressive power and how gradients propagate during training.

### 21.1. Memory Hook

```text
Need non-linearity?
        ↓
Activation Function

Hidden layer?
        ↓
ReLU

Binary classification?
        ↓
Sigmoid

Multiclass classification?
        ↓
Softmax

Multi-label classification?
        ↓
Multiple Sigmoids

Regression?
        ↓
Linear Output

Sigmoid / Tanh
        ↓
Can Saturate
        ↓
Vanishing Gradients

ReLU
        ↓
Fast + Simple
        ↓
But Can Die
```
