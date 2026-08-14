# 07. Backpropagation

## 1.1 What Is Backpropagation?

**Backpropagation** is the algorithm used to compute how much each trainable parameter in a neural network contributed to the final loss.

It works by propagating error information **backward** through the network using the chain rule of calculus.

The overall training flow is:

```text
Input
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

Forward propagation computes values.

Backpropagation computes derivatives.

---

## 1.2 Why Backpropagation Is Needed

A neural network may contain many weights and biases.

Suppose the loss is:

```math
J
```

and the network contains a weight:

```math
w
```

To update that weight, gradient descent needs:

```math
\frac{\partial J}{\partial w}
```

For every trainable parameter, we need to know:

> If this parameter changes slightly, how much will the loss change?

Backpropagation calculates these derivatives efficiently.

---

## 1.3 Backpropagation vs Gradient Descent

These two concepts are closely related but are not the same.

Backpropagation computes:

```math
\frac{\partial J}{\partial \theta}
```

Gradient descent then uses this quantity:

```math
\theta
\leftarrow
\theta
-
\eta
\frac{\partial J}{\partial \theta}
```

Therefore:

```text
Backpropagation
= Compute gradients

Gradient Descent
= Use gradients to update parameters
```

Backpropagation answers:

```text
How did each parameter affect the loss?
```

Gradient descent answers:

```text
How should we change each parameter?
```

---

## 1.4 The Chain Rule Is the Core Idea

Suppose:

```math
y=f(u)
```

and:

```math
u=g(x)
```

Then:

```math
y=f(g(x))
```

The chain rule gives:

```math
\frac{dy}{dx}
=
\frac{dy}{du}
\frac{du}{dx}
```

This means:

> To measure the effect of `x` on `y`, multiply the derivative along each intermediate step.

Neural networks contain many nested functions, so the chain rule is exactly what we need.

---

## 1.5 A Single Neuron Example

Consider one neuron:

```math
z=wx+b
```

followed by activation:

```math
a=f(z)
```

and loss:

```math
J=L(y,a)
```

The computation is:

```text
x
↓
z = wx + b
↓
a = f(z)
↓
J = L(y,a)
```

To determine how the weight affects the loss, we need:

```math
\frac{\partial J}{\partial w}
```

Using the chain rule:

```math
\frac{\partial J}{\partial w}
=
\frac{\partial J}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w}
```

This is the essence of backpropagation.

---

## 1.6 Breaking Down the Chain

For:

```math
z=wx+b
```

we have:

```math
\frac{\partial z}{\partial w}=x
```

Since:

```math
a=f(z)
```

we have:

```math
\frac{\partial a}{\partial z}=f'(z)
```

And the loss gives:

```math
\frac{\partial J}{\partial a}
```

Therefore:

```math
\frac{\partial J}{\partial w}
=
\frac{\partial J}{\partial a}
f'(z)
x
```

Every factor represents one part of the dependency chain.

---

## 1.7 Backpropagation Through the Bias

For:

```math
z=wx+b
```

the derivative with respect to the bias is:

```math
\frac{\partial z}{\partial b}=1
```

Therefore:

```math
\frac{\partial J}{\partial b}
=
\frac{\partial J}{\partial a}
\frac{\partial a}{\partial z}
```

or:

```math
\frac{\partial J}{\partial b}
=
\frac{\partial J}{\partial a}
f'(z)
```

The bias is updated using its own gradient just like a weight.

---

## 1.8 Why We Move Backward

During forward propagation:

```text
Input
→ Hidden Layer 1
→ Hidden Layer 2
→ Output
→ Loss
```

The loss depends directly on the output.

The output depends on the previous layer.

That layer depends on the layer before it.

Therefore, to determine the contribution of an early-layer weight, we must trace the dependency backward:

```text
Loss
← Output
← Hidden Layer 2
← Hidden Layer 1
← Input
```

This backward traversal gives the algorithm its name.

---

## 1.9 Two-Layer Network

Consider a simple network:

```math
z_1=w_1x+b_1
```

```math
a_1=f(z_1)
```

```math
z_2=w_2a_1+b_2
```

```math
\hat{y}=g(z_2)
```

and:

```math
J=L(y,\hat{y})
```

The forward flow is:

```text
x
↓
z₁
↓
a₁
↓
z₂
↓
ŷ
↓
J
```

To calculate the effect of `w₁` on the loss:

```math
\frac{\partial J}{\partial w_1}
```

we must trace through every intermediate variable.

---

## 1.10 Chain Rule Through Multiple Layers

The derivative becomes:

```math
\frac{\partial J}{\partial w_1}
=
\frac{\partial J}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial z_2}
\frac{\partial z_2}{\partial a_1}
\frac{\partial a_1}{\partial z_1}
\frac{\partial z_1}{\partial w_1}
```

This long expression may look complicated, but conceptually it simply means:

```text
Effect of w₁ on z₁
×
Effect of z₁ on a₁
×
Effect of a₁ on z₂
×
Effect of z₂ on prediction
×
Effect of prediction on loss
```

The effects multiply along the computational path.

---

## 1.11 Local Gradients

A useful way to understand backpropagation is through **local gradients**.

Each operation only needs to know its own derivative.

For example:

```math
z=wx+b
```

knows:

```math
\frac{\partial z}{\partial w}=x
```

An activation:

```math
a=f(z)
```

knows:

```math
\frac{\partial a}{\partial z}=f'(z)
```

The loss function knows:

```math
\frac{\partial J}{\partial a}
```

Backpropagation combines these local gradients using the chain rule.

Therefore:

```text
Local Derivatives
      ↓
Chain Rule
      ↓
Global Gradient
```

---

## 1.12 Computational Graph View

Consider:

```math
z=wx+b
```

```math
a=f(z)
```

```math
J=L(y,a)
```

The computational graph is:

```text
x ──┐
    × w
w ──┘
     ↓
    + b
     ↓
     z
     ↓
    f(z)
     ↓
     a
     ↓
   Loss
     ↓
     J
```

Forward propagation moves downward through this graph.

Backpropagation moves upward through it, calculating derivatives.

```text
Forward:
values flow →

Backward:
gradients flow ←
```

---

## 1.13 Backpropagation at the Output Layer

The process starts at the loss.

Suppose:

```math
J=L(y,\hat{y})
```

The first derivative is:

```math
\frac{\partial J}{\partial \hat{y}}
```

This measures how sensitive the loss is to the network's prediction.

The derivative is then propagated into the output layer.

---

## 1.14 Example with Mean Squared Error

Suppose:

```math
J=(y-\hat{y})^2
```

Then:

```math
\frac{\partial J}{\partial \hat{y}}
=
2(\hat{y}-y)
```

If:

```math
\hat{y}>y
```

the derivative is positive.

If:

```math
\hat{y}<y
```

the derivative is negative.

This becomes the starting error signal for backpropagation.

---

## 1.15 Backpropagation Through an Activation Function

Suppose:

```math
a=f(z)
```

and we already know:

```math
\frac{\partial J}{\partial a}
```

Then:

```math
\frac{\partial J}{\partial z}
=
\frac{\partial J}{\partial a}
\frac{\partial a}{\partial z}
```

Since:

```math
\frac{\partial a}{\partial z}
=
f'(z)
```

we obtain:

```math
\frac{\partial J}{\partial z}
=
\frac{\partial J}{\partial a}
f'(z)
```

This is why activation-function derivatives matter during training.

---

## 1.16 ReLU During Backpropagation

For ReLU:

```math
f(z)=\max(0,z)
```

the derivative is approximately:

```math
f'(z)
=
\begin{cases}
0, & z<0 \\
1, & z>0
\end{cases}
```

Therefore:

```math
\frac{\partial J}{\partial z}
=
\frac{\partial J}{\partial a}
f'(z)
```

If:

```math
z>0
```

the gradient passes through unchanged.

If:

```math
z<0
```

the gradient becomes zero.

This explains both:

- ReLU's strong gradient flow for positive values
- the dying ReLU problem for persistently negative values

---

## 1.17 Sigmoid During Backpropagation

For sigmoid:

```math
\sigma(z)
=
\frac{1}{1+e^{-z}}
```

the derivative is:

```math
\sigma'(z)
=
\sigma(z)(1-\sigma(z))
```

Thus:

```math
\frac{\partial J}{\partial z}
=
\frac{\partial J}{\partial a}
\sigma(z)(1-\sigma(z))
```

When sigmoid saturates:

```math
\sigma(z)\approx0
```

or:

```math
\sigma(z)\approx1
```

its derivative approaches zero.

This causes the gradient to shrink as it moves backward through many layers.

---

## 1.18 The Error Signal

A commonly used intermediate quantity is:

```math
\delta
=
\frac{\partial J}{\partial z}
```

This is often called the **error signal** for a neuron or layer.

It captures how sensitive the loss is to that layer's pre-activation value.

For layer `l`:

```math
\delta^{[l]}
=
\frac{\partial J}{\partial z^{[l]}}
```

This quantity is extremely useful because gradients for both weights and biases can be derived from it.

---

## 1.19 Weight Gradient from the Error Signal

For a neuron:

```math
z=wa+b
```

the derivative is:

```math
\frac{\partial z}{\partial w}=a
```

Therefore:

```math
\frac{\partial J}{\partial w}
=
\delta a
```

For a whole layer, the matrix form is:

```math
\frac{\partial J}{\partial W^{[l]}}
=
\delta^{[l]}
\left(
a^{[l-1]}
\right)^T
```

This gives the gradient for every weight connecting the previous layer to the current layer.

---

## 1.20 Bias Gradient from the Error Signal

Because:

```math
z=Wa+b
```

and:

```math
\frac{\partial z}{\partial b}=1
```

the bias gradient is simply related to:

```math
\delta
```

For a single example:

```math
\frac{\partial J}{\partial b^{[l]}}
=
\delta^{[l]}
```

For a batch, the gradients are aggregated across examples.

---

## 1.21 Propagating Error to the Previous Layer

Once we have:

```math
\delta^{[l]}
```

we need to determine the error signal for the previous layer.

Conceptually:

```text
Current Layer Error
       ↓
Current Weights
       ↓
Previous Layer
```

The previous layer's error depends on:

- the current layer's error
- the weights connecting the layers
- the derivative of the previous activation function

A common form is:

```math
\delta^{[l-1]}
=
\left(
W^{[l]}
\right)^T
\delta^{[l]}
\odot
f'^{[l-1]}
\left(
z^{[l-1]}
\right)
```

where:

```math
\odot
```

denotes element-wise multiplication.

---

## 1.22 Why the Weight Matrix Is Transposed

During the forward pass:

```math
z^{[l]}
=
W^{[l]}a^{[l-1]}+b^{[l]}
```

The weights carry information from the previous layer to the current layer.

During backpropagation, the gradient must move in the reverse direction.

Therefore:

```math
\left(W^{[l]}\right)^T
```

naturally appears when propagating gradients backward.

Conceptually:

```text
Forward:
Previous Layer
   ↓ W
Current Layer

Backward:
Current Layer
   ↓ Wᵀ
Previous Layer
```

---

## 1.23 Backpropagation Through a Whole Network

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
  ↓
Loss
```

Backpropagation proceeds:

```text
Loss
  ↓
Output Gradient
  ↓
Layer 3 Gradients
  ↓
Layer 2 Gradients
  ↓
Layer 1 Gradients
```

At each layer:

1. compute the layer's error signal
2. compute weight gradients
3. compute bias gradients
4. propagate the error to the previous layer

---

## 1.24 Forward Values Must Be Remembered

Backpropagation often needs values calculated during forward propagation.

For example:

```math
a^{[l-1]}
```

is required when calculating:

```math
\frac{\partial J}{\partial W^{[l]}}
```

Similarly:

```math
z^{[l]}
```

may be needed to calculate:

```math
f'(z^{[l]})
```

Therefore, deep-learning frameworks typically store intermediate forward-pass values so they can be reused during the backward pass.

This is one reason training requires more memory than inference.

---

## 1.25 Automatic Differentiation

Modern frameworks such as PyTorch and TensorFlow generally calculate gradients automatically.

The developer specifies the forward computation:

```text
Input
→ Layers
→ Prediction
→ Loss
```

The framework builds a computational graph describing how the output depends on the parameters.

It then applies automatic differentiation to compute:

```math
\frac{\partial J}{\partial \theta}
```

for all trainable parameters.

This process is often referred to as **autograd** or **automatic differentiation**.

---

## 1.26 Backpropagation Is Not Numerical Guessing

Backpropagation does not estimate gradients by repeatedly changing weights and observing what happens.

Instead, it computes derivatives analytically through the computational graph.

For example, it does not normally perform:

```text
Try w + tiny amount
Measure loss

Try w - tiny amount
Measure loss

Approximate derivative
```

That technique is called **numerical differentiation**.

Backpropagation uses the chain rule and is vastly more efficient for large neural networks.

---

## 1.27 Numerical Gradient Checking

Although numerical differentiation is too slow for normal training, it can be useful for checking whether backpropagation has been implemented correctly.

For a small value:

```math
\epsilon
```

a numerical derivative can be approximated as:

```math
\frac{\partial J}{\partial w}
\approx
\frac{
J(w+\epsilon)-J(w-\epsilon)
}{
2\epsilon
}
```

This numerical value can be compared with the gradient produced by backpropagation.

This technique is called **gradient checking**.

---

## 1.28 Why Backpropagation Is Efficient

A naive method could separately calculate each parameter's effect on the loss.

For a network with millions of parameters, this would be extremely expensive.

Backpropagation reuses intermediate derivatives.

Once the gradient reaching a layer is known, the same information can be used to calculate gradients for many parameters in that layer.

This reuse makes backpropagation computationally efficient.

---

## 1.29 Shared Dependency

Suppose many weights contribute to the same intermediate neuron.

Instead of recomputing the entire derivative path from the loss separately for each weight, backpropagation first computes:

```math
\frac{\partial J}{\partial z}
```

Once this value is known:

```math
\frac{\partial J}{\partial w_i}
=
\frac{\partial J}{\partial z}
\frac{\partial z}{\partial w_i}
```

for every incoming weight.

The expensive upstream derivative is therefore reused.

---

## 1.30 Backpropagation and Batches

With a mini-batch, the loss is usually averaged across several examples:

```math
J
=
\frac{1}{B}
\sum_{i=1}^{B}
L^{(i)}
```

Backpropagation then computes gradients of this batch loss.

For example:

```math
\frac{\partial J}{\partial W}
```

represents the aggregate effect of the batch on that weight matrix.

The optimizer performs one parameter update using these gradients.

---

## 1.31 Backpropagation Does Not Update Parameters

A subtle but important point:

Backpropagation itself does **not** change weights.

It computes quantities such as:

```math
\frac{\partial J}{\partial W}
```

and:

```math
\frac{\partial J}{\partial b}
```

The optimizer performs the update:

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
→ gradient calculation

Optimizer
→ parameter modification
```

---

## 1.32 Vanishing Gradients

During backpropagation, derivatives from many layers are multiplied together.

Suppose several derivatives are smaller than `1`.

For example:

```math
0.2\times0.2\times0.2\times0.2
=
0.0016
```

As the gradient travels backward through many layers, it may become extremely small.

This is the **vanishing gradient problem**.

Earlier layers then receive tiny updates and learn very slowly.

Sigmoid and tanh activations can contribute strongly to this problem.

---

## 1.33 Exploding Gradients

The opposite can also occur.

Suppose several factors are larger than `1`:

```math
3\times3\times3\times3
=
81
```

Repeated multiplication can make gradients extremely large.

This is known as the **exploding gradient problem**.

Consequences may include:

- unstable training
- huge parameter updates
- rapidly increasing loss
- numerical overflow

Vanishing and exploding gradients become especially important in deep and recurrent neural networks.

---

## 1.34 Why ReLU Helped Deep Learning

For positive inputs, ReLU has derivative:

```math
f'(z)=1
```

Therefore, the activation function itself does not shrink the gradient in that region.

This helped make deeper feed-forward networks easier to train compared with networks using many sigmoid or tanh layers.

ReLU does not eliminate every gradient problem, but it substantially improves gradient flow in many architectures.

---

## 1.35 Backpropagation and the Learning Rate

Backpropagation determines the gradient:

```math
\nabla J
```

The optimizer multiplies it by the learning rate:

```math
\eta\nabla J
```

Therefore, these concepts have different roles:

```text
Backpropagation
→ direction and sensitivity

Learning Rate
→ size of update

Optimizer
→ update rule
```

If gradients are correct but the learning rate is badly chosen, training can still fail.

---

## 1.36 Full Training Example

Consider:

```text
Input
   ↓
Dense Layer
   ↓
ReLU
   ↓
Dense Layer
   ↓
Sigmoid
   ↓
Prediction
```

Training proceeds as follows.

### Forward Pass

```math
z^{[1]}
=
W^{[1]}x+b^{[1]}
```

```math
a^{[1]}
=
\mathrm{ReLU}(z^{[1]})
```

```math
z^{[2]}
=
W^{[2]}a^{[1]}+b^{[2]}
```

```math
\hat{y}
=
\sigma(z^{[2]})
```

### Loss

```math
J=L(y,\hat{y})
```

### Backward Pass

Compute:

```math
\frac{\partial J}{\partial W^{[2]}}
```

```math
\frac{\partial J}{\partial b^{[2]}}
```

then propagate backward and compute:

```math
\frac{\partial J}{\partial W^{[1]}}
```

```math
\frac{\partial J}{\partial b^{[1]}}
```

### Update

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

Then the next forward pass begins.

---

## 1.37 The Complete Learning Loop

The whole process is:

```text
Initialize Parameters
        ↓
Take Mini-Batch
        ↓
Forward Propagation
        ↓
Prediction
        ↓
Compute Loss
        ↓
Backpropagation
        ↓
Compute Gradients
        ↓
Optimizer Step
        ↓
Update Parameters
        ↓
Repeat
```

This loop is repeated across batches and epochs.

---

## 1.38 Forward Propagation vs Backpropagation

| Forward Propagation | Backpropagation |
|---|---|
| Moves input toward output | Moves gradient information backward |
| Computes activations | Computes derivatives |
| Produces prediction | Determines parameter sensitivity |
| Uses weights | Computes gradients for weights |
| Required for training and inference | Required mainly during training |

A compact distinction is:

```text
Forward:
What did the network compute?

Backward:
Why did it produce that error?
```

---

## 1.39 Backpropagation vs Optimizer

| Backpropagation | Optimizer |
|---|---|
| Computes gradients | Updates parameters |
| Uses chain rule | Uses optimization rule |
| Produces ∂J/∂W | Produces new W |
| Same basic mechanism across many optimizers | Can be SGD, Adam, RMSProp, etc. |

The relationship is:

```text
Backpropagation
      ↓
Gradients
      ↓
Optimizer
      ↓
New Parameters
```

---

## 1.40 Key Takeaways

- Backpropagation computes gradients of the loss with respect to trainable parameters.
- It propagates error information backward through the network.
- The **chain rule** is the mathematical foundation of backpropagation.
- Each operation contributes a local derivative.
- Local derivatives are multiplied to determine total parameter sensitivity.
- Backpropagation begins at the loss and moves toward earlier layers.
- Activation-function derivatives directly influence gradient flow.
- The error signal is commonly represented as:

```math
\delta
=
\frac{\partial J}{\partial z}
```

- Weight gradients depend on the error signal and the previous layer's activation.
- Bias gradients are directly related to the error signal.
- Intermediate forward-pass values are required during backpropagation.
- Modern deep-learning frameworks perform backpropagation through automatic differentiation.
- Backpropagation is much more efficient than numerical differentiation.
- Gradient checking can numerically verify an implementation.
- Backpropagation computes gradients but does not update parameters.
- The optimizer uses the gradients to update parameters.
- Repeated multiplication of derivatives can cause vanishing or exploding gradients.
- ReLU helps gradient flow for positive activations.

### Memory Hook

```text
Forward
= Values Move Forward

Backward
= Gradients Move Backward

Chain Rule
= Multiply Effects Along the Path

Backpropagation
= Find ∂Loss / ∂Parameter

Gradient Descent
= Use That Gradient

Optimizer
= Decide the Update Rule

Core Flow:

Forward
→ Loss
→ Backprop
→ Gradients
→ Optimizer
→ Updated Weights
```
