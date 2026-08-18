# 04. Forward Propagation

## 1.1 What Is Forward Propagation?

**Forward propagation** is the process of passing input data through a neural network, layer by layer, until the final prediction is produced.

At each layer, the network performs two basic operations:

```text
Linear Transformation
        ↓
Activation Function
```

For layer `l`:

```math
\mathbf{z}^{[l]}
=
W^{[l]}\mathbf{a}^{[l-1]}
+
\mathbf{b}^{[l]}
```

followed by:

```math
\mathbf{a}^{[l]}
=
f^{[l]}
\left(
\mathbf{z}^{[l]}
\right)
```

The output of one layer becomes the input to the next.

Forward propagation therefore means:

```text
Input
  ↓
Layer 1 Computation
  ↓
Layer 2 Computation
  ↓
...
  ↓
Output Prediction
```

---

## 1.2 Input as the First Activation

For convenience, the input itself is often treated as the activation of layer zero:

```math
\mathbf{a}^{[0]}=\mathbf{x}
```

Then the first hidden layer computes:

```math
\mathbf{z}^{[1]}
=
W^{[1]}\mathbf{a}^{[0]}
+
\mathbf{b}^{[1]}
```

and:

```math
\mathbf{a}^{[1]}
=
f^{[1]}
\left(
\mathbf{z}^{[1]}
\right)
```

Since:

```math
\mathbf{a}^{[0]}=\mathbf{x}
```

this is equivalent to:

```math
\mathbf{a}^{[1]}
=
f^{[1]}
\left(
W^{[1]}\mathbf{x}
+
\mathbf{b}^{[1]}
\right)
```

This notation becomes useful when describing deep networks compactly.

---

## 1.3 The Two Computations Inside Each Layer

Every standard dense layer performs two conceptual stages.

### Linear Step

```math
\mathbf{z}^{[l]}
=
W^{[l]}\mathbf{a}^{[l-1]}
+
\mathbf{b}^{[l]}
```

This calculates a weighted combination of the previous layer's outputs.

### Activation Step

```math
\mathbf{a}^{[l]}
=
f^{[l]}
\left(
\mathbf{z}^{[l]}
\right)
```

This introduces non-linearity.

Therefore:

```text
Previous Activations
        ↓
Weights + Bias
        ↓
z
        ↓
Activation Function
        ↓
a
```

The distinction between `z` and `a` is important:

```text
z = value before activation
a = value after activation
```

---

## 1.4 Example: One Hidden Layer

Consider a neural network with:

```text
2 inputs
   ↓
3 hidden neurons
   ↓
1 output neuron
```

The input is:

```math
\mathbf{x}
=
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
```

The hidden layer has:

```math
W^{[1]}
\in
\mathbb{R}^{3\times2}
```

and:

```math
\mathbf{b}^{[1]}
\in
\mathbb{R}^{3}
```

The hidden layer computes:

```math
\mathbf{z}^{[1]}
=
W^{[1]}\mathbf{x}
+
\mathbf{b}^{[1]}
```

followed by:

```math
\mathbf{a}^{[1]}
=
f^{[1]}
\left(
\mathbf{z}^{[1]}
\right)
```

The output layer then computes:

```math
z^{[2]}
=
W^{[2]}\mathbf{a}^{[1]}
+
b^{[2]}
```

followed by:

```math
\hat{y}
=
f^{[2]}(z^{[2]})
```

The entire forward pass is therefore:

```text
x
↓
W¹x + b¹
↓
Activation
↓
a¹
↓
W²a¹ + b²
↓
Output Activation
↓
ŷ
```

---

## 1.5 Numerical Example

Suppose the input is:

```math
\mathbf{x}
=
\begin{bmatrix}
2 \\
1
\end{bmatrix}
```

and the first hidden neuron has:

```math
\mathbf{w}
=
\begin{bmatrix}
0.5 \\
-1
\end{bmatrix}
```

with bias:

```math
b=0.2
```

The neuron computes:

```math
z
=
0.5(2)
+
(-1)(1)
+
0.2
```

Therefore:

```math
z=0.2
```

If the neuron uses ReLU:

```math
a=\mathrm{ReLU}(0.2)
```

so:

```math
a=0.2
```

This value is then passed forward to neurons in the next layer.

Every neuron in the network performs essentially this same operation.

---

## 1.6 Forward Propagation Through Multiple Layers

For a deeper network:

```text
Input
  ↓
Hidden Layer 1
  ↓
Hidden Layer 2
  ↓
Hidden Layer 3
  ↓
Output
```

the computation becomes:

```math
\mathbf{a}^{[1]}
=
f^{[1]}
\left(
W^{[1]}\mathbf{x}
+
\mathbf{b}^{[1]}
\right)
```

then:

```math
\mathbf{a}^{[2]}
=
f^{[2]}
\left(
W^{[2]}\mathbf{a}^{[1]}
+
\mathbf{b}^{[2]}
\right)
```

then:

```math
\mathbf{a}^{[3]}
=
f^{[3]}
\left(
W^{[3]}\mathbf{a}^{[2]}
+
\mathbf{b}^{[3]}
\right)
```

and finally:

```math
\hat{\mathbf{y}}
=
f^{[4]}
\left(
W^{[4]}\mathbf{a}^{[3]}
+
\mathbf{b}^{[4]}
\right)
```

Each layer transforms one representation into another.

---

## 1.7 Function Composition View

A neural network can also be viewed as a composition of functions.

Suppose each layer represents a transformation:

```math
f_1,\;f_2,\;f_3
```

Then the network computes:

```math
\hat{y}
=
f_3
\left(
f_2
\left(
f_1(\mathbf{x})
\right)
\right)
```

This is a **composition of functions**.

The output of one function becomes the input to the next.

This perspective becomes extremely important during backpropagation because derivatives must pass backward through this chain of functions using the **chain rule**.

---

## 1.8 Why Forward Propagation Is Called "Forward"

The term refers to the direction in which information flows.

```text
Inputs
  ↓
Hidden Layers
  ↓
Output
```

The data moves from earlier layers toward later layers.

No parameter updates happen during the forward pass itself.

Forward propagation simply computes:

```text
Given current weights and biases,
what prediction does the network produce?
```

Learning happens only after the prediction is compared with the true target and the resulting error is propagated backward.

---

## 1.9 Forward Propagation During Training

During training, the network follows this sequence:

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss Calculation
  ↓
Backpropagation
  ↓
Parameter Update
```

Forward propagation therefore produces the prediction required to calculate the loss.

For a training sample:

```math
(\mathbf{x},y)
```

the network first computes:

```math
\hat{y}=F(\mathbf{x})
```

where `F` represents the entire neural network.

The loss is then calculated as:

```math
L(y,\hat{y})
```

That loss tells the network how wrong the prediction is.

---

## 1.10 Forward Propagation During Inference

Forward propagation is also used after training.

During **inference**, the network receives unseen input and produces a prediction.

The process is:

```text
New Input
   ↓
Forward Propagation
   ↓
Prediction
```

No loss calculation is required unless we are evaluating the model against known labels.

No backpropagation is performed.

No weights are updated.

Therefore:

```text
Training
= Forward Pass
+ Loss
+ Backward Pass
+ Parameter Update

Inference
= Forward Pass Only
```

---

## 1.11 Batch Forward Propagation

Neural networks rarely process only one sample at a time.

Suppose we have a batch containing `m` examples.

Instead of representing one input as:

```math
\mathbf{x}
```

we can represent the batch as a matrix:

```math
X
=
\begin{bmatrix}
| & | & & | \\
\mathbf{x}^{(1)}
&
\mathbf{x}^{(2)}
&
\cdots
&
\mathbf{x}^{(m)}
\\
| & | & & |
\end{bmatrix}
```

Then the entire layer can be computed simultaneously:

```math
Z^{[l]}
=
W^{[l]}A^{[l-1]}
+
\mathbf{b}^{[l]}
```

followed by:

```math
A^{[l]}
=
f^{[l]}(Z^{[l]})
```

This allows many training examples to be processed using efficient matrix operations.

---

## 1.12 Why Vectorization Matters

Consider processing 1,000 training samples.

A naive implementation might loop through them one by one:

```text
Sample 1
Sample 2
Sample 3
...
Sample 1000
```

A vectorized implementation instead performs operations on the entire batch:

```math
Z = WX + b
```

This is substantially more efficient on modern hardware.

CPUs, GPUs, and specialized AI accelerators are designed to perform large matrix operations quickly.

Therefore, neural-network computation relies heavily on **vectorization**.

---

## 1.13 Matrix Shapes During a Batch Forward Pass

Suppose:

```math
n_{\text{in}}
```

is the number of input features,

```math
n_{\text{out}}
```

is the number of neurons in the current layer,

and:

```math
m
```

is the batch size.

Then:

```math
A^{[l-1]}
\in
\mathbb{R}^{n_{\text{in}}\times m}
```

and:

```math
W^{[l]}
\in
\mathbb{R}^{n_{\text{out}}\times n_{\text{in}}}
```

Therefore:

```math
W^{[l]}A^{[l-1]}
\in
\mathbb{R}^{n_{\text{out}}\times m}
```

and thus:

```math
Z^{[l]}
\in
\mathbb{R}^{n_{\text{out}}\times m}
```

Each column represents one training example.

Each row represents one neuron.

---

## 1.14 Broadcasting the Bias

The bias vector has shape:

```math
\mathbf{b}^{[l]}
\in
\mathbb{R}^{n_{\text{out}}}
```

Yet:

```math
Z^{[l]}
\in
\mathbb{R}^{n_{\text{out}}\times m}
```

The same bias vector must therefore be added to every sample in the batch.

Conceptually:

```text
Sample 1 → + b
Sample 2 → + b
Sample 3 → + b
...
Sample m → + b
```

Numerical libraries and deep-learning frameworks usually perform this automatically using **broadcasting**.

---

## 1.15 Hidden Activations as Learned Features

The hidden activations:

```math
\mathbf{a}^{[1]},
\mathbf{a}^{[2]},
\ldots
```

can be interpreted as increasingly transformed versions of the original input.

For example:

```text
Raw Input
   ↓
a¹
   ↓
a²
   ↓
a³
   ↓
Prediction
```

Each activation vector can be thought of as a learned feature representation.

This is why hidden layers are sometimes described as **feature extractors**.

The network is not merely moving numbers forward.

It is progressively transforming the representation of the data.

---

## 1.16 Forward Propagation in Binary Classification

Consider the architecture:

```text
10 inputs
    ↓
Dense(8, ReLU)
    ↓
Dense(4, ReLU)
    ↓
Dense(1, Sigmoid)
```

The first layer computes:

```math
\mathbf{a}^{[1]}
=
\mathrm{ReLU}
\left(
W^{[1]}\mathbf{x}
+
\mathbf{b}^{[1]}
\right)
```

The second:

```math
\mathbf{a}^{[2]}
=
\mathrm{ReLU}
\left(
W^{[2]}\mathbf{a}^{[1]}
+
\mathbf{b}^{[2]}
\right)
```

The output layer computes:

```math
\hat{p}
=
\sigma
\left(
W^{[3]}\mathbf{a}^{[2]}
+
b^{[3]}
\right)
```

The result:

```math
0 < \hat{p} < 1
```

can be interpreted as the model's estimated probability of the positive class.

---

## 1.17 Forward Propagation in Multiclass Classification

Suppose the network predicts one of four classes.

The output layer produces four raw values:

```math
z_1,\;z_2,\;z_3,\;z_4
```

Softmax converts them into:

```math
p_1,\;p_2,\;p_3,\;p_4
```

such that:

```math
p_1+p_2+p_3+p_4=1
```

The prediction is usually:

```math
\hat{y}
=
\mathrm{arg\,max}_k p_k
```

meaning that the class with the largest predicted probability is selected.

---

## 1.18 Forward Propagation in Regression

For regression, the final layer often uses a linear activation.

For example:

```text
Input
  ↓
Dense(64, ReLU)
  ↓
Dense(32, ReLU)
  ↓
Dense(1, Linear)
```

The final computation is:

```math
\hat{y}
=
W^{[3]}\mathbf{a}^{[2]}
+
b^{[3]}
```

No sigmoid or softmax is required because the output may need to take unrestricted numerical values.

---

## 1.19 Parameters Are Fixed During the Forward Pass

During a particular forward pass, the values of:

```math
W^{[1]},
W^{[2]},
\ldots
```

and:

```math
\mathbf{b}^{[1]},
\mathbf{b}^{[2]},
\ldots
```

are treated as fixed.

The network simply evaluates the current function represented by those parameters.

The weights are changed only after:

```text
Forward Pass
   ↓
Loss
   ↓
Backpropagation
   ↓
Optimizer
   ↓
Weight Update
```

This distinction is useful:

```text
Forward propagation
= use current parameters

Backpropagation
= calculate how parameters should change

Optimization
= actually update parameters
```

---

## 1.20 Computational Graph Intuition

A forward pass can also be represented as a **computational graph**.

For a single neuron:

```math
z = wx+b
```

followed by:

```math
a=f(z)
```

and a loss:

```math
L=L(y,a)
```

the graph is:

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
 Loss(y,a)
     ↓
     L
```

During forward propagation, values move through this graph from input toward loss.

During backpropagation, derivatives move through the same graph in the opposite direction.

This computational-graph view underlies automatic differentiation systems used in frameworks such as PyTorch and TensorFlow.

---

## 1.21 Forward Pass vs Backward Pass

These two ideas should be kept clearly separate.

### Forward Pass

Computes values:

```text
Input
  ↓
Activations
  ↓
Prediction
  ↓
Loss
```

### Backward Pass

Computes gradients:

```text
Loss
  ↓
Output Layer Gradients
  ↓
Hidden Layer Gradients
  ↓
Earlier Layer Gradients
```

Therefore:

```text
Forward
= What did the network predict?

Backward
= Which parameters caused the error,
  and by how much?
```

---

## 1.22 Forward Propagation and the Entire Neural Network

A neural network can be viewed as one large function:

```math
\hat{y}=F(\mathbf{x};\theta)
```

where:

```math
\theta
```

represents all trainable parameters:

```math
\theta
=
\left\{
W^{[1]},
\mathbf{b}^{[1]},
W^{[2]},
\mathbf{b}^{[2]},
\ldots
\right\}
```

Forward propagation evaluates:

```math
F(\mathbf{x};\theta)
```

for the current parameter values.

Training attempts to find parameter values that make:

```math
F(\mathbf{x};\theta)
```

produce predictions close to the true targets.

---

## 1.23 Key Takeaways

- **Forward propagation** passes data from the input layer toward the output layer.
- Each dense layer performs a linear transformation followed by an activation function.
- The pre-activation value is commonly written as:

```math
\mathbf{z}^{[l]}
=
W^{[l]}\mathbf{a}^{[l-1]}
+
\mathbf{b}^{[l]}
```

- The activation is:

```math
\mathbf{a}^{[l]}
=
f^{[l]}
\left(
\mathbf{z}^{[l]}
\right)
```

- The input is commonly written as:

```math
\mathbf{a}^{[0]}=\mathbf{x}
```

- A neural network is a composition of many functions.
- Forward propagation uses the current weights and biases but does not update them.
- During training, the forward pass produces the prediction required to compute the loss.
- During inference, only the forward pass is required.
- Batches allow multiple examples to be processed simultaneously.
- Vectorization makes neural-network computation efficient.
- Hidden activations can be interpreted as learned feature representations.
- Forward propagation computes values; backpropagation computes gradients.
- The complete network can be written as:

```math
\hat{y}=F(\mathbf{x};\theta)
```

### Memory Hook

```text
Forward Propagation
=
Input
↓
Weighted Sum
↓
Activation
↓
Next Layer
↓
Prediction

At Every Layer:

z = Wa + b
a = f(z)

Training:
Forward
→ Loss
→ Backward
→ Update

Inference:
Forward Only
```

---
