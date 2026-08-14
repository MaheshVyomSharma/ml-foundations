# 09. Weight Initialization

## 1.1 Why Initialization Matters

Before training begins, a neural network needs initial values for its trainable weights.

For a layer:

```math
\mathbf{z}
=
W\mathbf{x}
+
\mathbf{b}
```

the matrix:

```math
W
```

must already contain values before the first forward pass can occur.

These initial values affect:

- how activations behave
- how gradients flow
- how quickly training starts
- whether neurons learn different features
- whether gradients vanish or explode

A poor initialization can make an otherwise valid network difficult to train.

---

## 1.2 Why Not Initialize All Weights to Zero?

Suppose every neuron in a hidden layer starts with identical weights:

```math
w_1=w_2=\cdots=w_n=0
```

and identical biases.

Because the neurons receive the same inputs, they produce the same outputs.

During backpropagation, they also receive identical gradients.

Therefore:

```text
Same Initialization
      ↓
Same Outputs
      ↓
Same Gradients
      ↓
Same Updates
      ↓
Neurons Remain Identical
```

The neurons fail to learn different features.

This is called the **symmetry problem**.

---

## 1.3 Breaking Symmetry

To break symmetry, weights are usually initialized using random values.

For example:

```math
w_i \sim \text{some random distribution}
```

Now different neurons begin with slightly different parameter values.

As a result:

```text
Different Weights
      ↓
Different Outputs
      ↓
Different Gradients
      ↓
Different Updates
      ↓
Different Learned Features
```

Randomness is therefore useful not because the model should remain random, but because it allows neurons to begin learning distinct representations.

---

## 1.4 Why Small Random Values?

Early neural-network implementations often used small random values such as:

```math
w \sim \mathcal{N}(0,0.01^2)
```

where:

```math
\mathcal{N}
```

denotes a normal distribution.

Very large initial weights can produce very large activations.

For activations such as sigmoid or tanh, this may push neurons into saturation.

For example:

```math
z \gg 0
```

can cause:

```math
\sigma(z)\approx1
```

and:

```math
\sigma'(z)\approx0
```

leading to weak gradients.

However, initializing weights **too small** can also cause gradients and activations to shrink.

Therefore, simply saying "use small random weights" is not enough.

The scale should depend on the network structure.

---

## 1.5 The Variance Problem

Consider a neuron:

```math
z
=
\sum_{i=1}^{n}
w_i x_i
```

If many inputs contribute to the sum, the variance of `z` may grow with the number of inputs.

If the weights are too large:

```text
Layer 1
→ moderate activations

Layer 2
→ larger activations

Layer 3
→ huge activations
```

This can contribute to exploding activations and gradients.

If weights are too small:

```text
Layer 1
→ moderate activations

Layer 2
→ smaller activations

Layer 3
→ almost zero
```

This can contribute to vanishing activations and gradients.

Good initialization tries to keep signal magnitudes reasonably stable across layers.

---

## 1.6 Fan-In and Fan-Out

Two important terms are:

```text
fan-in
= number of inputs entering a neuron or layer

fan-out
= number of outputs leaving a layer
```

For a dense layer with:

```math
n_{\text{in}}
```

inputs and:

```math
n_{\text{out}}
```

neurons:

```math
\text{fan-in}=n_{\text{in}}
```

and:

```math
\text{fan-out}=n_{\text{out}}
```

Modern initialization strategies scale their random weights according to these values.

---

## 1.7 Xavier / Glorot Initialization

**Xavier initialization**, also called **Glorot initialization**, was designed to keep activation and gradient variances reasonably stable across layers.

A common variance target is approximately:

```math
\mathrm{Var}(W)
\approx
\frac{2}
{n_{\text{in}}+n_{\text{out}}}
```

One common normal-distribution form is:

```math
W
\sim
\mathcal{N}
\left(
0,
\frac{2}
{n_{\text{in}}+n_{\text{out}}}
\right)
```

Another frequently used simplified form scales according to:

```math
\frac{1}{n_{\text{in}}}
```

The exact formulation depends on the framework and chosen distribution.

---

## 1.8 Why Xavier Initialization Works

The basic goal is:

```text
Input Signal Magnitude
        ↓
Layer Transformation
        ↓
Similar Output Magnitude
```

rather than:

```text
Signal
↓
Shrinks
↓
Shrinks
↓
Almost Zero
```

or:

```text
Signal
↓
Grows
↓
Grows
↓
Explodes
```

Xavier initialization is especially suitable for activation functions such as:

- sigmoid
- tanh

where preserving variance across layers is important.

---

## 1.9 Xavier Uniform Initialization

Instead of sampling weights from a normal distribution, Xavier initialization can also use a uniform distribution.

A common form is:

```math
W
\sim
U
\left(
-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},
\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}
\right)
```

where:

```math
U(a,b)
```

means values sampled uniformly between `a` and `b`.

Normal and uniform Xavier initializations pursue the same basic goal: keeping signal variance controlled.

---

## 1.10 He Initialization

**He initialization** is designed primarily for ReLU-family activations.

For ReLU, roughly half of the input values may become zero.

Therefore, a slightly larger variance than Xavier is useful.

A common form is:

```math
W
\sim
\mathcal{N}
\left(
0,
\frac{2}{n_{\text{in}}}
\right)
```

Equivalently, the standard deviation is:

```math
\sqrt{\frac{2}{n_{\text{in}}}}
```

This is known as **He normal initialization**.

---

## 1.11 Why He Initialization Fits ReLU

ReLU is:

```math
f(z)=\max(0,z)
```

Negative values are mapped to zero.

This reduces the variance of the activation signal.

He initialization compensates by using a larger initial weight variance:

```math
\frac{2}{n_{\text{in}}}
```

instead of roughly:

```math
\frac{1}{n_{\text{in}}}
```

The goal is again to prevent the forward signal from shrinking excessively across layers.

---

## 1.12 Xavier vs He Initialization

A practical rule is:

| Activation | Common Initialization |
|---|---|
| Sigmoid | Xavier / Glorot |
| Tanh | Xavier / Glorot |
| ReLU | He |
| Leaky ReLU | He-style |
| Linear | Xavier often reasonable |

A useful memory rule is:

```text
Sigmoid / Tanh
→ Xavier

ReLU family
→ He
```

This is a default guideline rather than an absolute law.

---

## 1.13 Bias Initialization

Biases are often initialized to zero:

```math
b=0
```

Unlike weights, zero biases usually do not create a symmetry problem.

Why?

Because neurons already have different randomly initialized weights.

Therefore, their outputs and gradients differ even if their biases begin identically.

In many standard architectures:

```text
Weights
→ carefully randomized

Biases
→ zero is usually fine
```

---

## 1.14 Can Biases Be Non-Zero?

Yes.

Bias initialization can sometimes be chosen deliberately.

For example:

- small positive biases may influence early ReLU activation
- output-layer biases can encode prior class probabilities
- specialized architectures may use custom bias initialization

However, zero remains a common and effective default.

---

## 1.15 Initialization and Sigmoid Saturation

Suppose sigmoid receives a very large positive input:

```math
z=10
```

Then:

```math
\sigma(z)\approx1
```

and:

```math
\sigma'(z)\approx0
```

Similarly, for a large negative input:

```math
z=-10
```

the derivative is also very small.

If initial weights are too large, many sigmoid neurons may begin training already saturated.

Then:

```text
Large Initial Weights
        ↓
Large |z|
        ↓
Sigmoid Saturation
        ↓
Tiny Gradients
        ↓
Slow Learning
```

Initialization therefore directly interacts with activation choice.

---

## 1.16 Initialization and ReLU

ReLU does not saturate for large positive values.

However, poor initialization can still create problems.

If many neurons receive negative pre-activations:

```math
z<0
```

then:

```math
\mathrm{ReLU}(z)=0
```

and:

```math
f'(z)=0
```

A poor initialization can therefore increase the risk of inactive neurons.

He initialization helps create a more suitable activation scale for ReLU networks.

---

## 1.17 Forward Signal Preservation

Suppose:

```math
a^{[l-1]}
```

is the activation entering a layer.

The next pre-activation is:

```math
z^{[l]}
=
W^{[l]}a^{[l-1]}
```

Good initialization attempts to keep:

```math
\mathrm{Var}(z^{[l]})
```

from changing dramatically from layer to layer.

Conceptually:

```text
Layer 1 variance
≈
Layer 2 variance
≈
Layer 3 variance
```

Perfect equality is not required.

The main objective is to avoid rapid collapse or explosion.

---

## 1.18 Backward Gradient Preservation

Initialization also affects backward propagation.

Gradients repeatedly pass through weight matrices:

```math
\delta^{[l-1]}
=
\left(
W^{[l]}
\right)^T
\delta^{[l]}
\odot
f'(z^{[l-1]})
```

If weight scales are inappropriate, gradients can:

```text
Shrink repeatedly
→ Vanishing Gradients
```

or:

```text
Grow repeatedly
→ Exploding Gradients
```

A good initialization therefore helps both:

```text
Forward Signal Flow
+
Backward Gradient Flow
```

---

## 1.19 Initialization Does Not Replace Training

Initialization only determines the starting point.

It does not contain the final learned solution.

Conceptually:

```text
Initialization
= Starting Position

Training
= Journey Through Parameter Space

Learned Weights
= Final Useful Region
```

Even an excellent initialization still requires:

- forward propagation
- loss calculation
- backpropagation
- optimization

---

## 1.20 Initialization Affects Convergence

Two identical networks with different random initializations can train differently.

One run may converge quickly.

Another may converge more slowly.

A third may arrive at a somewhat different solution.

This happens because neural-network optimization is generally non-convex.

Therefore:

```text
Different Starting Point
        ↓
Different Optimization Path
        ↓
Possibly Different Final Solution
```

This is one reason experiments often use random seeds for reproducibility.

---

## 1.21 Random Seed

A **random seed** controls the pseudo-random number generator used by software.

Setting the same seed can help reproduce:

- weight initialization
- data shuffling
- some stochastic operations

For example:

```text
seed = 42
```

does not make the model less random in principle.

It makes the random sequence repeatable.

This is useful during experimentation and debugging.

---

## 1.22 Reproducibility Is Not Always Exact

Even with a fixed random seed, exact reproducibility may not always occur.

Reasons include:

- parallel GPU operations
- non-deterministic kernels
- hardware differences
- library implementation differences

Therefore:

```text
Same Seed
→ improved reproducibility

Same Seed
≠ guaranteed bit-for-bit identical results
```

This is especially relevant in large deep-learning systems.

---

## 1.23 Initialization in Deep-Learning Frameworks

Modern frameworks generally provide sensible default initializers.

For example, a dense ReLU layer may automatically use a variance-scaled initialization.

Therefore, developers do not usually hand-code every random distribution.

However, understanding initialization remains important because it explains:

- why defaults exist
- why activation choice matters
- why some networks train poorly
- when custom initialization may help

---

## 1.24 Initialization and Deep Networks

The deeper the network, the more important initialization becomes.

Consider:

```text
Layer 1
↓
Layer 2
↓
Layer 3
↓
...
↓
Layer 50
```

A small variance distortion at each layer can accumulate dramatically.

For example:

```math
0.9^{50}\approx0.005
```

while:

```math
1.1^{50}\approx117
```

Repeated scaling can therefore cause large differences across deep networks.

Variance-aware initialization helps prevent this accumulation.

---

## 1.25 Why Initialization Became Important Historically

Early deep networks were difficult to train partly because:

- sigmoid and tanh activations saturated
- gradients vanished
- poor initialization amplified instability
- deep architectures compounded these effects

Improvements such as:

- better initialization
- ReLU activations
- normalization methods
- improved optimizers

made deep networks substantially easier to train.

Initialization is therefore one piece of a larger training-stability puzzle.

---

## 1.26 Initialization and Model Capacity

Initialization does not change the architecture's theoretical capacity.

A network with:

```text
10 → 100 → 100 → 1
```

has the same number of parameters regardless of initialization.

What initialization changes is:

```text
How easily that capacity can be learned
```

This distinction matters.

A model may have enough theoretical capacity yet still train poorly because optimization fails.

---

## 1.27 Initialization and Symmetry in Convolutional Networks

The same symmetry principle applies beyond dense layers.

Convolutional filters must also begin with different weights.

If every filter began identically:

```text
Filter 1
=
Filter 2
=
Filter 3
```

they would initially detect the same patterns and receive the same updates.

Random initialization encourages different filters to specialize in different features.

---

## 1.28 Zero Initialization Is Fine for Some Parameters

The rule:

```text
Never initialize anything to zero
```

is incorrect.

The better rule is:

```text
Do not initialize all trainable weights identically
```

Common zero-initialized quantities include:

- biases
- some normalization parameters
- certain architecture-specific parameters

The problem is not zero itself.

The problem is **symmetry among learnable units**.

---

## 1.29 Initialization Strategy Summary

A practical workflow is:

```text
Choose Activation
      ↓
Choose Matching Initialization
      ↓
Initialize Weights Randomly
      ↓
Initialize Biases
      ↓
Begin Training
```

Typical defaults:

```text
Tanh / Sigmoid
→ Xavier

ReLU / Leaky ReLU
→ He

Biases
→ Zero
```

---

## 1.30 Initialization and the Complete Training Loop

Initialization occurs before the normal training loop:

```text
Choose Architecture
      ↓
Initialize Parameters
      ↓
Take Batch
      ↓
Forward Propagation
      ↓
Loss
      ↓
Backpropagation
      ↓
Optimizer Step
      ↓
Repeat
```

The initialization determines where the optimization process begins.

Everything afterward determines how the parameters evolve.

---

## 1.31 Key Takeaways

- Neural networks require initial weight values before training begins.
- All weights in a layer should not be initialized identically.
- Identical initialization creates a symmetry problem.
- Random initialization allows neurons to learn different features.
- Weight scale matters, not just randomness.
- Weights that are too large can cause exploding activations or saturation.
- Weights that are too small can contribute to vanishing signals.
- `fan-in` is the number of inputs entering a layer.
- `fan-out` is the number of outputs produced by a layer.
- Xavier/Glorot initialization is commonly suited to sigmoid and tanh networks.
- He initialization is commonly suited to ReLU-family activations.
- Biases can often safely begin at zero.
- Good initialization helps preserve both forward activations and backward gradients.
- Initialization affects optimization speed and stability.
- Random seeds improve reproducibility.
- Initialization provides the starting point; it does not replace learning.
- Deeper networks are generally more sensitive to poor initialization.

### Memory Hook

```text
All Weights Equal
→ Symmetry
→ Neurons Learn the Same Thing

Random Weights
→ Break Symmetry

Sigmoid / Tanh
→ Xavier

ReLU Family
→ He

Bias
→ Usually Zero

Good Initialization
=
Stable Forward Signals
+
Stable Backward Gradients

Initialization
= Where Learning Starts
```
