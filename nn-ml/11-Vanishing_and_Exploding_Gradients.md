# 11. Vanishing and Exploding Gradients

## 1.1 Why Gradient Stability Matters

A neural network learns by propagating gradients backward through its layers.

For a deep network:

```text
Loss
 ↓
Output Layer
 ↓
Hidden Layer 3
 ↓
Hidden Layer 2
 ↓
Hidden Layer 1
```

Each layer modifies the gradient before passing it to the previous layer.

If gradients become progressively smaller, training can slow or stop.

If they become progressively larger, training can become unstable.

These two problems are called:

```text
Vanishing Gradients
Exploding Gradients
```

They are especially important in deep neural networks and recurrent neural networks.

---

## 1.2 Where the Problem Comes From

Backpropagation repeatedly applies the chain rule.

For a sequence of functions:

```math
y
=
f_3(f_2(f_1(x)))
```

the derivative is:

```math
\frac{dy}{dx}
=
\frac{dy}{df_3}
\frac{df_3}{df_2}
\frac{df_2}{df_1}
\frac{df_1}{dx}
```

In a neural network, many derivatives and weight terms are multiplied together as gradients move backward.

Therefore, repeated multiplication can dramatically change gradient magnitude.

---

## 1.3 Vanishing Gradients

A gradient **vanishes** when it becomes extremely small as it moves backward through the network.

Suppose several derivative terms are:

```math
0.2
```

Then:

```math
0.2^5
=
0.00032
```

and:

```math
0.2^{10}
\approx
1.0 \times 10^{-7}
```

A gradient that began at a useful magnitude can therefore become almost zero after passing through many layers.

Conceptually:

```text
Output Gradient
     ↓
0.2
     ↓
0.04
     ↓
0.008
     ↓
0.0016
     ↓
Almost Zero
```

---

## 1.4 Why Vanishing Gradients Are a Problem

Parameter updates depend on gradients:

```math
W
\leftarrow
W
-
\eta
\frac{\partial J}{\partial W}
```

If:

```math
\frac{\partial J}{\partial W}
\approx 0
```

then:

```math
\Delta W
\approx 0
```

The weights barely change.

Earlier layers may therefore learn extremely slowly.

In a deep network:

```text
Output Layers
→ learn reasonably well

Early Hidden Layers
→ receive tiny gradients
→ learn very slowly
```

This prevents the network from effectively learning useful representations in its deeper structure.

---

## 1.5 Sigmoid and Vanishing Gradients

The sigmoid function is:

```math
\sigma(z)
=
\frac{1}{1+e^{-z}}
```

Its derivative is:

```math
\sigma'(z)
=
\sigma(z)(1-\sigma(z))
```

The maximum derivative of sigmoid is:

```math
0.25
```

Therefore:

```math
0 < \sigma'(z) \leq 0.25
```

If several sigmoid derivatives are multiplied:

```math
0.25^5
\approx
0.00098
```

the gradient can shrink rapidly.

This is one major reason early deep networks using sigmoid activations were difficult to train.

---

## 1.6 Sigmoid Saturation Makes It Worse

For large positive or negative values of `z`:

```math
\sigma(z)
\approx
1
```

or:

```math
\sigma(z)
\approx
0
```

and therefore:

```math
\sigma'(z)
\approx
0
```

The neuron is said to be **saturated**.

Thus:

```text
Large |z|
   ↓
Sigmoid Saturation
   ↓
Derivative ≈ 0
   ↓
Gradient Shrinks
   ↓
Vanishing Gradient
```

Poor weight initialization can make this problem worse by producing large pre-activation values at the beginning of training.

---

## 1.7 Tanh and Vanishing Gradients

The tanh function is:

```math
\tanh(z)
=
\frac{e^z-e^{-z}}
{e^z+e^{-z}}
```

Its derivative is:

```math
\frac{d}{dz}\tanh(z)
=
1-\tanh^2(z)
```

Although tanh is zero-centred and often behaves better than sigmoid, it also saturates for large positive or negative inputs.

Therefore:

```math
\tanh'(z)
\approx0
```

in saturated regions.

Tanh networks can therefore also suffer from vanishing gradients.

---

## 1.8 ReLU and Gradient Flow

ReLU is:

```math
f(z)
=
\max(0,z)
```

For positive inputs:

```math
f'(z)=1
```

Therefore, ReLU does not automatically shrink the gradient in its positive region.

Conceptually:

```text
Sigmoid
→ derivative often < 1
→ repeated shrinking

ReLU, z > 0
→ derivative = 1
→ gradient passes through activation
```

This is one reason ReLU became important in deep feed-forward networks.

---

## 1.9 ReLU Does Not Eliminate Every Problem

For negative inputs:

```math
f'(z)=0
```

Therefore, a ReLU neuron can completely block the gradient if it remains in the negative region.

This produces the **dying ReLU problem**.

Thus ReLU trades one issue for another:

```text
Sigmoid / Tanh
→ widespread gradient shrinkage

ReLU
→ better positive gradient flow
→ possible dead neurons
```

Variants such as Leaky ReLU reduce this risk.

---

## 1.10 Exploding Gradients

An **exploding gradient** occurs when repeated multiplication causes gradient magnitude to grow excessively.

Suppose several terms are:

```math
2
```

Then:

```math
2^{10}
=
1024
```

If they are:

```math
3
```

then:

```math
3^{10}
=
59049
```

A moderate gradient can therefore become enormous.

Conceptually:

```text
Small Gradient
     ↓
2
     ↓
4
     ↓
8
     ↓
16
     ↓
32
     ↓
Very Large
```

---

## 1.11 Why Exploding Gradients Are a Problem

The parameter update is:

```math
W_{\text{new}}
=
W_{\text{old}}
-
\eta
\frac{\partial J}{\partial W}
```

If:

```math
\left|
\frac{\partial J}{\partial W}
\right|
```

becomes extremely large, the parameter update can also become enormous.

Consequences include:

- unstable training
- rapidly changing weights
- oscillating loss
- divergence
- numerical overflow
- `NaN` values

Conceptually:

```text
Huge Gradient
   ↓
Huge Update
   ↓
Weights Jump Far
   ↓
Loss Becomes Unstable
```

---

## 1.12 Signs of Exploding Gradients

Possible symptoms include:

```text
Loss suddenly becomes enormous

Loss oscillates wildly

Weights grow rapidly

Gradient norms become very large

Training produces NaN or Inf
```

These symptoms do not uniquely prove exploding gradients, but they are strong clues.

---

## 1.13 Deep Networks Amplify the Problem

Suppose a network contains many layers:

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

During backpropagation, gradients pass through all these transformations.

Even small systematic scaling effects can accumulate.

For example:

```math
0.9^{50}
\approx
0.005
```

while:

```math
1.1^{50}
\approx
117
```

Therefore:

```text
Repeated Factor < 1
→ Vanishing

Repeated Factor > 1
→ Exploding
```

This is a simplified intuition, but it captures the core mechanism.

---

## 1.14 Weight Matrices Also Affect Gradient Magnitude

Backpropagation through a layer contains terms such as:

```math
\delta^{[l-1]}
=
\left(W^{[l]}\right)^T
\delta^{[l]}
\odot
f'(z^{[l-1]})
```

Therefore, gradient magnitude depends on both:

```text
Activation Derivatives
+
Weight Matrix Magnitudes
```

If weight matrices repeatedly amplify signals, gradients may explode.

If they repeatedly shrink signals, gradients may vanish.

This is why weight initialization is closely connected to gradient stability.

---

## 1.15 Weight Initialization as a Solution

Good initialization attempts to keep signal variance reasonably stable.

For sigmoid and tanh networks:

```text
Xavier / Glorot Initialization
```

is commonly used.

For ReLU-family activations:

```text
He Initialization
```

is commonly used.

The goal is:

```text
Forward Activations
→ remain reasonably scaled

Backward Gradients
→ remain reasonably scaled
```

Initialization does not eliminate every gradient problem, but it greatly improves training stability.

---

## 1.16 Gradient Clipping

Gradient clipping directly addresses exploding gradients.

Suppose the gradient is:

```math
g
=
\nabla_{\theta}J
```

and its norm exceeds a threshold:

```math
\|g\|>c
```

The gradient can be rescaled:

```math
g_{\text{clipped}}
=
c
\frac{g}{\|g\|}
```

so that:

```math
\|g_{\text{clipped}}\|=c
```

The optimizer then uses the clipped gradient.

Conceptually:

```text
Huge Gradient
     ↓
Clip Magnitude
     ↓
Controlled Gradient
     ↓
Optimizer Update
```

---

## 1.17 Clipping by Value

Another form clips each gradient component independently.

For example:

```math
g_i
=
\begin{cases}
c, & g_i > c \\
g_i, & -c \leq g_i \leq c \\
-c, & g_i < -c
\end{cases}
```

This limits individual values.

However, **gradient norm clipping** is often preferable because it preserves the overall gradient direction while limiting its magnitude.

---

## 1.18 Learning Rate and Gradient Instability

Even a reasonable gradient can cause large parameter changes if the learning rate is too high.

The update magnitude depends on:

```math
\eta
\nabla J
```

Therefore:

```text
Large Gradient
+
Large Learning Rate
=
Very Large Parameter Update
```

Reducing the learning rate can sometimes stabilize training.

However, if the underlying gradients are genuinely exploding, learning-rate reduction alone may not fully solve the problem.

---

## 1.19 Normalization Helps Training Stability

Normalization techniques can help keep activation distributions under better control.

Examples include:

```text
Batch Normalization
Layer Normalization
```

These techniques reduce uncontrolled shifts in activation scale and can make optimization more stable.

They do not directly replace gradient clipping or initialization, but they can reduce conditions that contribute to unstable gradients.

---

## 1.20 Residual Connections

Very deep networks often use **residual connections**, also called skip connections.

Instead of learning only:

```math
y=F(x)
```

a residual block may compute:

```math
y=F(x)+x
```

The direct path:

```math
x \rightarrow y
```

allows information and gradients to bypass some transformations.

Conceptually:

```text
x ────────────────┐
│                 ↓
│              Addition
│                 ↓
└→ Layers → F(x) ─┘
```

Residual connections greatly improved the trainability of very deep neural networks.

---

## 1.21 Why Skip Connections Help Gradient Flow

For:

```math
y=F(x)+x
```

the derivative is:

```math
\frac{\partial y}{\partial x}
=
\frac{\partial F(x)}{\partial x}
+
1
```

The direct `+1` path gives the gradient an alternative route backward.

Even if:

```math
\frac{\partial F(x)}{\partial x}
```

becomes small, the identity path can still transmit gradient information.

This makes very deep networks easier to optimize.

---

## 1.22 Vanishing Gradients in Recurrent Neural Networks

The problem becomes particularly severe in recurrent neural networks because the same transformation is repeatedly applied across time.

Conceptually:

```text
Time 1
 ↓
Time 2
 ↓
Time 3
 ↓
Time 4
 ↓
...
```

Backpropagation through many time steps repeatedly multiplies derivatives.

Therefore, information from distant time steps can vanish before it influences earlier states.

This makes learning long-term dependencies difficult.

---

## 1.23 LSTM and GRU

Architectures such as:

```text
LSTM
GRU
```

were developed partly to improve long-term gradient flow in recurrent networks.

They contain gating mechanisms that create more controlled pathways for information and gradients.

These architectures will be examined later when recurrent neural networks are covered.

---

## 1.24 Gradient Norm

A useful diagnostic quantity is the **gradient norm**.

For gradient vector:

```math
g
=
\begin{bmatrix}
g_1 \\
g_2 \\
\vdots \\
g_n
\end{bmatrix}
```

its Euclidean norm is:

```math
\|g\|_2
=
\sqrt{
g_1^2
+
g_2^2
+
\cdots
+
g_n^2
}
```

Monitoring gradient norms can reveal whether gradients are:

```text
Very Small
→ possible vanishing gradients

Very Large
→ possible exploding gradients
```

---

## 1.25 Vanishing Gradients vs Small Useful Gradients

A small gradient is not automatically a problem.

Near a well-optimized solution:

```math
\nabla J
\approx0
```

is expected.

Vanishing gradients are problematic when:

```text
Early Layers
→ receive tiny gradients

while

Model Has Not Learned Well
```

The issue is therefore not merely the numerical size of a gradient, but whether that size prevents useful learning.

---

## 1.26 Exploding Gradients vs Large Useful Gradients

Similarly, a large gradient may sometimes be useful when the model is far from a good solution.

The problem occurs when gradients become so large that updates are unstable.

Therefore:

```text
Large Gradient
≠ automatically exploding

Tiny Gradient
≠ automatically vanishing
```

Context matters.

---

## 1.27 Common Causes of Vanishing Gradients

Important causes include:

- very deep networks
- repeated derivatives smaller than `1`
- sigmoid saturation
- tanh saturation
- poor weight initialization
- long recurrent sequences

Conceptually:

```text
Many Small Multipliers
        ↓
Gradient Shrinks
        ↓
Early Layers Stop Learning
```

---

## 1.28 Common Causes of Exploding Gradients

Important causes include:

- deep computational chains
- large weight values
- poor initialization
- unstable recurrent dynamics
- excessive learning rates

Conceptually:

```text
Many Large Multipliers
        ↓
Gradient Grows
        ↓
Huge Parameter Updates
```

---

## 1.29 Common Solutions

A useful summary is:

| Problem | Common Techniques |
|---|---|
| Vanishing gradients | ReLU-family activations |
| Vanishing gradients | Xavier / He initialization |
| Vanishing gradients | Residual connections |
| Vanishing gradients in RNNs | LSTM / GRU |
| Exploding gradients | Gradient clipping |
| Exploding gradients | Better initialization |
| Exploding gradients | Lower learning rate |
| Both | Normalization techniques |
| Both | Better architecture design |

Modern deep networks often combine several of these techniques.

---

## 1.30 Gradient Stability and Activation Choice

Activation functions directly affect backpropagation through:

```math
\frac{\partial a}{\partial z}
=
f'(z)
```

Therefore:

```text
Activation Function
        ↓
Activation Derivative
        ↓
Gradient Flow
        ↓
Training Stability
```

This is why activation choice is not merely about introducing non-linearity.

It also determines how easily a network can be trained.

---

## 1.31 Gradient Stability and Initialization

Initialization determines the scale of the initial weights.

The weights influence:

```text
Forward Activation Magnitudes
+
Backward Gradient Magnitudes
```

Therefore:

```text
Activation Choice
+
Weight Initialization
+
Network Depth
+
Optimizer
+
Learning Rate
```

all interact to determine training stability.

These concepts should not be viewed independently.

---

## 1.32 Gradient Stability in the Training Loop

The complete relationship is:

```text
Forward Propagation
       ↓
Activations
       ↓
Loss
       ↓
Backpropagation
       ↓
Gradients
       ↓
Check Gradient Scale
       ↓
Optimizer
       ↓
Parameter Update
```

If gradients vanish:

```text
Updates Too Small
→ Learning Stalls
```

If gradients explode:

```text
Updates Too Large
→ Training Becomes Unstable
```

Successful deep learning requires keeping gradients in a useful numerical range.

---

## 1.33 Key Takeaways

- Neural networks rely on gradients flowing backward through many layers.
- Backpropagation repeatedly multiplies derivatives through the chain rule.
- Gradients can shrink exponentially across deep computational chains.
- Extremely small gradients cause the **vanishing gradient problem**.
- Early layers may learn very slowly when gradients vanish.
- Sigmoid and tanh are especially vulnerable because they can saturate.
- ReLU improves gradient flow for positive activations.
- ReLU can still suffer from dead neurons.
- Repeated multiplication by large values can cause **exploding gradients**.
- Exploding gradients can produce unstable updates, divergence, and numerical errors.
- Weight matrices as well as activation derivatives affect gradient magnitude.
- Xavier and He initialization help preserve activation and gradient scale.
- Gradient clipping is a major technique for controlling exploding gradients.
- Lower learning rates may improve stability.
- Normalization techniques can help stabilize activations and optimization.
- Residual connections provide alternative paths for gradient flow.
- LSTM and GRU architectures help address long-term gradient problems in recurrent networks.
- Gradient norms can be monitored to diagnose instability.
- Small gradients are not automatically vanishing gradients, and large gradients are not automatically exploding gradients.

### Memory Hook

```text
Backpropagation
= Multiply Derivatives Backward

Repeated Values < 1
→ Gradient Shrinks
→ Vanishing Gradient

Repeated Values > 1
→ Gradient Grows
→ Exploding Gradient

Vanishing:
→ ReLU
→ Better Initialization
→ Residual Connections
→ LSTM / GRU

Exploding:
→ Gradient Clipping
→ Better Initialization
→ Smaller Learning Rate

Core Problem:

Vanishing
= Too Little Learning Signal

Exploding
= Too Much Learning Signal
```