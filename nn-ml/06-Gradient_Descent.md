# 06. Gradient Descent

## 1.1 What Is Gradient Descent?

**Gradient descent** is an optimization algorithm used to reduce a model's loss by adjusting its trainable parameters.

A neural network contains parameters such as:

```math
W^{[1]}, W^{[2]}, \ldots
```

and:

```math
b^{[1]}, b^{[2]}, \ldots
```

The loss depends on these parameters:

```math
J(\theta)
```

where:

```math
\theta
```

represents all trainable parameters.

The goal of training is to find parameter values that minimize the loss:

```math
\theta^*
=
\mathrm{arg\,min}_{\theta} J(\theta)
```

Gradient descent provides a practical way to search for these values.

---

## 1.2 The Core Idea

Suppose the loss depends on one weight:

```math
J(w)
```

We want to determine how changing `w` affects the loss.

The derivative:

```math
\frac{dJ}{dw}
```

provides this information.

If:

```math
\frac{dJ}{dw} > 0
```

then increasing `w` increases the loss.

To reduce the loss, we should decrease `w`.

If:

```math
\frac{dJ}{dw} < 0
```

then increasing `w` decreases the loss.

To reduce the loss, we should increase `w`.

This leads to the gradient-descent update:

```math
w_{\text{new}}
=
w_{\text{old}}
-
\eta
\frac{dJ}{dw}
```

where:

```math
\eta
```

is the **learning rate**.

---

## 1.3 Why the Negative Gradient?

The gradient points in the direction of **steepest increase** of the loss.

Therefore:

```math
\nabla J
```

points uphill.

To reduce the loss, we move in the opposite direction:

```math
-\nabla J
```

Hence the update rule:

```math
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta \nabla J(\theta)
```

The negative sign does **not** mean that the gradient itself is always negative.

The gradient can be positive or negative.

The rule simply says:

> Move opposite to the direction in which the loss increases most rapidly.

This distinction is fundamental.

---

## 1.4 Gradient vs Derivative

For a function of one variable:

```math
J(w)
```

we use a derivative:

```math
\frac{dJ}{dw}
```

For a function of many parameters:

```math
J(w_1,w_2,\ldots,w_n)
```

we use partial derivatives:

```math
\frac{\partial J}{\partial w_1},
\frac{\partial J}{\partial w_2},
\ldots,
\frac{\partial J}{\partial w_n}
```

These partial derivatives form the **gradient vector**:

```math
\nabla J
=
\begin{bmatrix}
\frac{\partial J}{\partial w_1} \\
\frac{\partial J}{\partial w_2} \\
\vdots \\
\frac{\partial J}{\partial w_n}
\end{bmatrix}
```

So:

```text
Derivative
→ one variable

Gradient
→ many variables
```

---

## 1.5 Geometric Intuition

Imagine the loss as a landscape.

```text
High Loss
       /\
      /  \
     /    \
    /      \
   /        \
  /          \
 /            \
       ↓
   Low Loss
```

The current parameter values place the model somewhere on this landscape.

The gradient tells us the direction of steepest uphill movement.

Gradient descent moves in the opposite direction.

```text
Gradient
→ uphill

Negative Gradient
→ downhill
```

The process repeats until the model reaches a region of relatively low loss.

---

## 1.6 One-Dimensional Example

Suppose:

```math
J(w)=w^2
```

Then:

```math
\frac{dJ}{dw}=2w
```

Assume:

```math
w=4
```

Then:

```math
\frac{dJ}{dw}=8
```

With learning rate:

```math
\eta=0.1
```

the update is:

```math
w_{\text{new}}
=
4
-
0.1(8)
```

Therefore:

```math
w_{\text{new}}=3.2
```

The parameter moves closer to:

```math
w=0
```

where the loss is minimized.

---

## 1.7 What If the Gradient Is Negative?

Suppose:

```math
w=-4
```

Then:

```math
\frac{dJ}{dw}
=
2(-4)
=
-8
```

The update becomes:

```math
w_{\text{new}}
=
-4
-
0.1(-8)
```

Therefore:

```math
w_{\text{new}}
=
-3.2
```

Again, the parameter moves toward zero.

This demonstrates why the gradient itself does not need to be negative.

The update formula automatically moves in the correct direction.

---

## 1.8 Gradient Descent in Neural Networks

A neural network may contain thousands, millions, or billions of parameters.

For each parameter, we need to compute how the loss changes with respect to that parameter.

For example:

```math
\frac{\partial J}{\partial W^{[1]}}
```

```math
\frac{\partial J}{\partial b^{[1]}}
```

```math
\frac{\partial J}{\partial W^{[2]}}
```

and so on.

The parameters are then updated:

```math
W^{[l]}
\leftarrow
W^{[l]}
-
\eta
\frac{\partial J}{\partial W^{[l]}}
```

and:

```math
b^{[l]}
\leftarrow
b^{[l]}
-
\eta
\frac{\partial J}{\partial b^{[l]}}
```

This happens across all trainable layers.

---

## 1.9 The Learning Rate

The **learning rate** controls the size of each parameter update.

It is usually represented by:

```math
\eta
```

The update rule is:

```math
\theta
\leftarrow
\theta
-
\eta\nabla J(\theta)
```

The learning rate determines how far the optimizer moves in each step.

---

## 1.10 Learning Rate Too Small

If the learning rate is too small:

```text
Current Position
      ↓
 • → • → • → • → • → Minimum
```

the model may approach the minimum very slowly.

Consequences can include:

- very slow training
- many epochs required
- unnecessary computation
- possible stagnation in poorly conditioned regions

---

## 1.11 Learning Rate Too Large

If the learning rate is too large:

```text
Minimum
   ↓
\        /
 \      /
  •────• 
   \  /
    •
```

the optimizer may overshoot the minimum repeatedly.

The loss may:

- oscillate
- become unstable
- fail to converge
- even increase dramatically

Therefore:

```text
Learning rate too small
→ slow

Learning rate too large
→ unstable
```

---

## 1.12 Choosing the Learning Rate

The learning rate is one of the most important hyperparameters in neural-network training.

Typical values might be around:

```math
0.1,\;0.01,\;0.001,\;0.0001
```

but there is no universally correct value.

The appropriate learning rate depends on:

- architecture
- optimizer
- dataset
- normalization
- batch size
- parameter initialization

Modern optimizers can also adapt learning rates during training.

---

## 1.13 Batch Gradient Descent

In **batch gradient descent**, the gradient is computed using the entire training dataset before one parameter update.

Suppose the dataset contains `m` examples.

The cost is:

```math
J
=
\frac{1}{m}
\sum_{i=1}^{m}
L^{(i)}
```

The gradient is calculated from all `m` examples.

Then one update is performed.

Conceptually:

```text
Entire Dataset
      ↓
Compute Loss
      ↓
Compute Gradient
      ↓
One Parameter Update
```

This produces a stable gradient but may be computationally expensive for large datasets.

---

## 1.14 Stochastic Gradient Descent

In **Stochastic Gradient Descent (SGD)**, one training example is used for each parameter update.

```text
One Example
    ↓
Forward Pass
    ↓
Loss
    ↓
Gradient
    ↓
Update
```

Then the next example is processed.

SGD performs many updates and can be computationally efficient per update.

However, individual gradients are noisy.

The loss may therefore fluctuate considerably during training.

---

## 1.15 Mini-Batch Gradient Descent

**Mini-batch gradient descent** lies between batch gradient descent and pure SGD.

A small group of examples is processed together.

For a batch size `B`:

```math
J_{\text{batch}}
=
\frac{1}{B}
\sum_{i=1}^{B}
L^{(i)}
```

The workflow becomes:

```text
Mini-Batch
   ↓
Forward Pass
   ↓
Batch Loss
   ↓
Backpropagation
   ↓
Gradient
   ↓
Parameter Update
```

Then the next mini-batch is processed.

This is the standard approach used in most modern deep-learning systems.

---

## 1.16 Comparing the Three Forms

| Method | Data Per Update | Main Advantage | Main Limitation |
|---|---:|---|---|
| Batch Gradient Descent | Entire dataset | Stable gradient | Expensive |
| SGD | 1 sample | Frequent updates | Very noisy |
| Mini-Batch Gradient Descent | Small batch | Efficient and practical | Batch size must be chosen |

In modern deep learning, when people casually say **SGD**, they may sometimes refer to mini-batch SGD rather than literally one sample at a time.

---

## 1.17 Why Mini-Batches Work Well

Mini-batches provide a useful compromise.

They allow:

- efficient matrix computation
- GPU acceleration
- more frequent parameter updates
- lower memory requirements than full-batch training
- some useful noise in the gradient

The slight randomness in mini-batch gradients can sometimes help optimization avoid poor regions of the loss landscape.

---

## 1.18 Batch Size

The **batch size** is the number of examples processed before one parameter update.

Common values include:

```math
16,\;32,\;64,\;128,\;256
```

though much larger batch sizes are also used.

A smaller batch produces noisier gradient estimates.

A larger batch produces smoother gradient estimates.

Conceptually:

```text
Small Batch
→ noisier gradients
→ more frequent updates

Large Batch
→ smoother gradients
→ fewer updates per epoch
```

---

## 1.19 Epoch

An **epoch** means one complete pass through the entire training dataset.

Suppose:

```math
N=10{,}000
```

training samples exist and:

```math
B=100
```

is the batch size.

Then the number of batches per epoch is:

```math
\frac{10{,}000}{100}=100
```

So one epoch contains approximately:

```text
100 forward passes
100 backward passes
100 parameter updates
```

assuming each batch produces one update.

---

## 1.20 Epoch vs Batch vs Iteration

These terms are closely related.

### Epoch

One complete pass through the training dataset.

### Batch

A subset of training examples processed together.

### Iteration

One parameter-update step.

If:

```math
N
```

is the number of samples and:

```math
B
```

is the batch size, then approximately:

```math
\text{iterations per epoch}
=
\frac{N}{B}
```

For example:

```math
N=1000
```

and:

```math
B=100
```

gives:

```math
10
```

iterations per epoch.

---

## 1.21 Why Multiple Epochs?

A single pass through the training data is usually not enough.

The model begins with poorly chosen initial parameters.

Each epoch gradually improves them.

Conceptually:

```text
Epoch 1
→ poor model

Epoch 5
→ better

Epoch 20
→ better still

Too many epochs
→ possible overfitting
```

Training therefore continues until the model has learned sufficiently useful parameters.

---

## 1.22 Gradient Descent Is Iterative

Gradient descent does not normally jump directly to the optimum.

Instead:

```text
Initialize Parameters
        ↓
Compute Prediction
        ↓
Compute Loss
        ↓
Compute Gradient
        ↓
Update Parameters
        ↓
Repeat
```

Mathematically:

```math
\theta_{t+1}
=
\theta_t
-
\eta\nabla J(\theta_t)
```

where:

```math
t
```

represents the optimization step.

---

## 1.23 Convex vs Non-Convex Optimization

For some classical models, the loss function may be **convex**.

A simplified convex loss looks like:

```text
\          /
 \        /
  \      /
   \____/
```

It contains one global minimum.

Neural-network loss functions are generally **non-convex** because many layers and non-linear transformations interact.

Conceptually:

```text
      /\        /\
  ___/  \______/  \___
     \__      __/
        \____/
```

The landscape may contain:

- valleys
- saddle points
- flat regions
- many local structures

Training neural networks is therefore a more complicated optimization problem than simple convex optimization.

---

## 1.24 Local Minimum and Global Minimum

A **global minimum** is the lowest possible value of the loss.

A **local minimum** is lower than nearby points but may not be the lowest point overall.

Conceptually:

```text
      Local Minimum
           ↓
   \      / \         /
    \____/   \       /
              \_____/
                 ↑
          Global Minimum
```

In high-dimensional neural networks, the distinction is more complicated than this simple picture suggests.

Modern optimization often focuses less on finding the exact mathematical global minimum and more on finding a parameter region that gives good generalization.

---

## 1.25 Saddle Points

A **saddle point** is a point where the gradient may be zero even though the point is not a minimum.

For example, the function:

```math
f(x,y)=x^2-y^2
```

has a saddle point at:

```math
(0,0)
```

Along one direction the function increases, while along another it decreases.

High-dimensional neural-network optimization can contain many such regions.

This is one reason a zero or tiny gradient does not always mean that the best solution has been reached.

---

## 1.26 Plateaus

A **plateau** is a relatively flat region of the loss surface.

In such a region:

```math
\nabla J \approx 0
```

so updates become small.

Training may slow considerably.

Plateaus can arise from:

- activation saturation
- poor initialization
- certain regions of the loss landscape

Optimization techniques such as momentum and adaptive learning rates can help navigate these regions.

---

## 1.27 Gradient Descent and Backpropagation Are Not the Same Thing

These concepts are related but distinct.

**Backpropagation** computes gradients.

For example:

```math
\frac{\partial J}{\partial W}
```

**Gradient descent** uses those gradients to update parameters:

```math
W
\leftarrow
W
-
\eta
\frac{\partial J}{\partial W}
```

Therefore:

```text
Backpropagation
= Calculate gradients

Gradient Descent
= Use gradients to update parameters
```

This distinction is extremely important.

---

## 1.28 Gradient Descent and the Optimizer

Gradient descent is the basic optimization idea.

Modern neural networks often use more advanced **optimizers**, such as:

- SGD
- SGD with Momentum
- RMSProp
- Adam

These methods still use gradients, but modify how parameter updates are calculated.

For example, plain gradient descent uses:

```math
\theta
\leftarrow
\theta
-
\eta\nabla J
```

while Adam additionally tracks information about previous gradients and gradient magnitudes.

The fundamental principle remains:

```text
Gradient
→ information about loss direction

Optimizer
→ rule for using that information
```

---

## 1.29 Gradient Descent and Feature Scaling

Input scaling can strongly affect optimization.

Suppose one feature has values around:

```math
0.01
```

while another has values around:

```math
100000
```

Their influence on gradients can differ dramatically.

This may make the loss landscape poorly conditioned.

Feature normalization or standardization often makes gradient-based optimization easier.

This is one reason preprocessing remains important in neural networks.

---

## 1.30 Parameter Update for a Whole Layer

For one layer:

```math
W^{[l]}
```

is a matrix.

Its gradient has the same shape:

```math
\frac{\partial J}{\partial W^{[l]}}
```

The entire matrix is updated:

```math
W^{[l]}
\leftarrow
W^{[l]}
-
\eta
\frac{\partial J}{\partial W^{[l]}}
```

Similarly:

```math
b^{[l]}
\leftarrow
b^{[l]}
-
\eta
\frac{\partial J}{\partial b^{[l]}}
```

Every individual weight and bias therefore receives its own gradient-based adjustment.

---

## 1.31 Gradient Descent Across the Network

Consider:

```text
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
Output
```

Backpropagation computes gradients for:

```text
Layer 3 parameters
Layer 2 parameters
Layer 1 parameters
```

Then the optimizer updates all of them.

Thus one training step is:

```text
Forward Pass
     ↓
Loss
     ↓
Backpropagation
     ↓
Gradients for Every Parameter
     ↓
Optimizer
     ↓
Updated Network
```

---

## 1.32 Training Loss Curve

During successful training, the loss often decreases over time.

Conceptually:

```text
Loss
│\
│ \
│  \
│   \
│    \__
│       \___
│
└────────────── Training Steps
```

However, the curve is rarely perfectly smooth.

Mini-batch training introduces fluctuations.

A loss curve may therefore look more like:

```text
Loss
│\_
│  \__
│     \_/\
│         \__
│            \_
└────────────── Training Steps
```

The overall trend matters more than every individual fluctuation.

---

## 1.33 When Gradient Descent Has Converged

Training may be considered to have converged when improvements become very small.

For example:

```math
J_{t+1}\approx J_t
```

or:

```math
\|\nabla J\|\approx0
```

In practice, training is often stopped using criteria such as:

- validation loss stops improving
- maximum number of epochs reached
- learning rate becomes extremely small
- early stopping condition is triggered

Convergence does not necessarily mean the mathematically perfect optimum has been found.

---

## 1.34 The Complete Optimization Loop

The complete neural-network training loop can now be written as:

```text
1. Initialize weights and biases

2. Take a mini-batch

3. Forward propagation
        ↓
   Produce predictions

4. Compute loss

5. Backpropagation
        ↓
   Compute gradients

6. Optimizer step
        ↓
   Update parameters

7. Repeat for all batches

8. Repeat for multiple epochs
```

In compact form:

```text
Forward
→ Loss
→ Backward
→ Update
→ Repeat
```

This is the central training loop of a neural network.

---

## 1.35 Key Takeaways

- Gradient descent is an optimization method used to reduce the loss.
- The gradient points in the direction of steepest increase in loss.
- Gradient descent moves in the opposite direction.
- The gradient itself is not always negative.
- The update rule is:

```math
\theta
\leftarrow
\theta
-
\eta\nabla J(\theta)
```

- The learning rate controls the update size.
- A learning rate that is too small causes slow training.
- A learning rate that is too large can cause instability or divergence.
- Batch gradient descent uses the entire dataset per update.
- SGD uses one example per update.
- Mini-batch gradient descent uses a subset of examples and is the standard practical approach.
- An epoch is one complete pass through the training dataset.
- An iteration is one parameter-update step.
- Neural-network optimization is generally non-convex.
- Loss landscapes can contain local minima, saddle points, and plateaus.
- Backpropagation computes gradients.
- Gradient descent or another optimizer uses those gradients to update parameters.
- Feature scaling can make gradient-based optimization more effective.
- Neural-network training repeatedly performs forward propagation, loss calculation, backpropagation, and parameter updates.

### Memory Hook

```text
Gradient
= Direction of Steepest Increase

Negative Gradient
= Direction of Descent

Update:
new = old - learning_rate × gradient

Backpropagation
= Find the Gradients

Gradient Descent
= Use the Gradients

Epoch
= One Full Pass Through Data

Batch
= Samples Processed Together

Iteration
= One Parameter Update

Training Loop:
Forward
→ Loss
→ Backward
→ Update
→ Repeat
```
