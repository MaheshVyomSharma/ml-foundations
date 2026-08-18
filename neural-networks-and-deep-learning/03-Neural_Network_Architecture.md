# 03. Neural Network Architecture

## 1.1 From Neurons to Layers

A neural network is built by organizing many artificial neurons into **layers**.

A typical feed-forward neural network contains three broad types of layers:

- **Input layer**
- **Hidden layer(s)**
- **Output layer**

Conceptually:

```text
Input Layer
    ↓
Hidden Layer
    ↓
Hidden Layer
    ↓
Output Layer
```

Each layer receives values from the previous layer, performs a transformation, and passes its outputs forward.

For a hidden layer:

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
f\left(\mathbf{z}^{[l]}\right)
```

where:

- `l` denotes the layer number
- `W` contains the weights
- `b` contains the biases
- `a` contains the activations
- `f` is the activation function

---

## 1.2 The Input Layer

The **input layer** represents the features supplied to the network.

Suppose the dataset contains four features:

```math
x_1,\;x_2,\;x_3,\;x_4
```

Then the network input can be represented as:

```math
\mathbf{x}
=
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4
\end{bmatrix}
```

The input layer does not usually perform any learned computation.

Its role is to provide the feature values to the first hidden layer.

Therefore:

```text
Input layer
= representation of the input features
```

If the model receives 20 numerical features, the input dimension is typically 20.

If it receives a 28 × 28 grayscale image flattened into a vector:

```math
28 \times 28 = 784
```

the input dimension would be 784.

---

## 1.3 Hidden Layers

A **hidden layer** is a layer between the input and output layers.

It is called hidden because its values are neither the original input nor the final prediction.

Each hidden neuron performs:

```math
z = \mathbf{w}^T\mathbf{x}+b
```

followed by:

```math
a=f(z)
```

When many neurons operate together:

```math
\mathbf{a}
=
f(W\mathbf{x}+\mathbf{b})
```

The activations produced by one hidden layer become the inputs to the next layer.

Conceptually:

```text
Raw Features
     ↓
Hidden Representation 1
     ↓
Hidden Representation 2
     ↓
Prediction
```

Hidden layers allow the network to learn intermediate representations of the data.

---

## 1.4 The Output Layer

The **output layer** produces the network's final prediction.

Its structure depends on the task.

### Regression

For a single continuous target:

```text
Output neurons = 1
Activation      = Linear
```

Example:

```math
\hat{y}\in\mathbb{R}
```

### Binary Classification

For two classes:

```text
Output neurons = 1
Activation      = Sigmoid
```

giving:

```math
0 < \hat{p} < 1
```

### Multiclass Classification

For `K` mutually exclusive classes:

```text
Output neurons = K
Activation      = Softmax
```

giving:

```math
P(y=1),P(y=2),\ldots,P(y=K)
```

with:

```math
\sum_{k=1}^{K}P(y=k)=1
```

### Multi-label Classification

For `K` independent labels:

```text
Output neurons = K
Activation      = Sigmoid on each output
```

---

## 1.5 Fully Connected Layers

A **fully connected layer**, also called a **dense layer**, connects every neuron in one layer to every neuron in the next layer.

Suppose one layer contains 3 neurons and the next contains 4 neurons.

Then each of the 4 neurons receives all 3 outputs from the previous layer.

Conceptually:

```text
Previous Layer       Next Layer

    ○ ─────────────→ ○
    │╲─────────────→ ○
    │ ╲────────────→ ○
    │  ╲───────────→ ○

    ○ ─────────────→ ○
    │╲─────────────→ ○
    │ ╲────────────→ ○
    │  ╲───────────→ ○

    ○ ─────────────→ ○
      ...
```

Instead of storing individual weights separately, we represent the connections using a matrix.

---

## 1.6 Weight Matrix Dimensions

Suppose a layer receives:

```math
n_{\text{in}}
```

inputs and contains:

```math
n_{\text{out}}
```

neurons.

The weight matrix has shape:

```math
W
\in
\mathbb{R}^{n_{\text{out}}\times n_{\text{in}}}
```

The bias vector has shape:

```math
\mathbf{b}
\in
\mathbb{R}^{n_{\text{out}}}
```

If:

```math
\mathbf{x}
\in
\mathbb{R}^{n_{\text{in}}}
```

then:

```math
W\mathbf{x}
\in
\mathbb{R}^{n_{\text{out}}}
```

and therefore:

```math
\mathbf{z}
=
W\mathbf{x}+\mathbf{b}
```

has one value for every neuron in the layer.

---

## 1.7 Example of Layer Dimensions

Suppose the input contains 5 features and the first hidden layer contains 3 neurons.

Then:

```math
\mathbf{x}\in\mathbb{R}^{5}
```

and:

```math
W^{[1]}
\in
\mathbb{R}^{3\times5}
```

with:

```math
\mathbf{b}^{[1]}
\in
\mathbb{R}^{3}
```

The layer computes:

```math
\mathbf{z}^{[1]}
=
W^{[1]}\mathbf{x}
+
\mathbf{b}^{[1]}
```

resulting in:

```math
\mathbf{z}^{[1]}
\in
\mathbb{R}^{3}
```

After activation:

```math
\mathbf{a}^{[1]}
=
f(\mathbf{z}^{[1]})
```

there are 3 outputs.

If the next layer contains 2 neurons:

```math
W^{[2]}
\in
\mathbb{R}^{2\times3}
```

because it receives 3 values and produces 2.

This gives the dimensional flow:

```text
5 inputs
   ↓
3 hidden neurons
   ↓
2 hidden neurons
```

---

## 1.8 Parameters in a Dense Layer

For a fully connected layer with:

```math
n_{\text{in}}
```

inputs and:

```math
n_{\text{out}}
```

neurons, the number of weights is:

```math
n_{\text{in}}\times n_{\text{out}}
```

Each output neuron also has one bias.

Therefore, the total number of trainable parameters is:

```math
n_{\text{in}}n_{\text{out}}
+
n_{\text{out}}
```

or:

```math
n_{\text{out}}(n_{\text{in}}+1)
```

Example:

A layer with 100 inputs and 50 neurons contains:

```math
100\times50 = 5000
```

weights and:

```math
50
```

biases.

Total:

```math
5000+50=5050
```

trainable parameters.

---

## 1.9 Depth

The **depth** of a neural network refers broadly to the number of successive computational layers.

A network with one hidden layer may look like:

```text
Input
  ↓
Hidden
  ↓
Output
```

A deeper network might look like:

```text
Input
  ↓
Hidden 1
  ↓
Hidden 2
  ↓
Hidden 3
  ↓
Hidden 4
  ↓
Output
```

Increasing depth allows the network to construct increasingly complex representations by composing transformations.

Conceptually:

```text
Simple representation
        ↓
More complex representation
        ↓
Higher-level representation
        ↓
Prediction
```

The term **deep learning** largely refers to learning with networks containing multiple such representation layers.

---

## 1.10 Width

The **width** of a neural network refers to the number of neurons in a layer.

For example:

```text
Layer 1 → 64 neurons
Layer 2 → 128 neurons
Layer 3 → 64 neurons
```

The network has varying widths across layers.

More neurons allow a layer to learn more features or patterns simultaneously.

However:

```text
More neurons
   ↓
More parameters
   ↓
More computational cost
   ↓
Greater risk of overfitting
```

Width is therefore a hyperparameter rather than something that should simply be maximized.

---

## 1.11 Depth vs Width

Both depth and width increase model capacity, but they do so differently.

### Width

A wider layer can learn many patterns at the same level of representation.

### Depth

A deeper network can compose simpler representations into progressively more complex ones.

Conceptually:

```text
Width
→ more features at one level

Depth
→ more levels of transformation
```

A useful intuition is:

```text
Wide network
= more neurons working in parallel

Deep network
= more transformations performed sequentially
```

Modern neural-network architectures often use both.

---

## 1.12 Representation Learning

One of the most important ideas in deep learning is **representation learning**.

Instead of manually designing every useful feature, the network learns intermediate features automatically.

For an image-recognition network, the conceptual progression might be:

```text
Pixels
  ↓
Edges
  ↓
Textures
  ↓
Shapes
  ↓
Object Parts
  ↓
Objects
```

Earlier layers often learn simpler patterns.

Later layers combine them into more abstract representations.

For structured tabular data, the learned representations may be less visually interpretable, but the same principle applies.

Each layer transforms the input representation into another representation that may be more useful for prediction.

---

## 1.13 Feed-Forward Neural Networks

A **feed-forward neural network** is one in which information flows from input toward output without forming cycles.

```text
Input
  ↓
Hidden Layer 1
  ↓
Hidden Layer 2
  ↓
Output
```

The output from a later layer does not feed directly back into an earlier layer.

A common fully connected feed-forward network is called a **Multilayer Perceptron**, or **MLP**.

Despite the name, modern MLP neurons generally use differentiable activation functions such as ReLU rather than classical perceptron step functions.

---

## 1.14 Multilayer Perceptron

A **Multilayer Perceptron (MLP)** typically consists of:

- an input layer
- one or more fully connected hidden layers
- an output layer

For example:

```text
Input: 10 features
       ↓
Dense: 64 neurons + ReLU
       ↓
Dense: 32 neurons + ReLU
       ↓
Output: 1 neuron + Sigmoid
```

This architecture could be used for binary classification.

Mathematically:

```math
\mathbf{a}^{[1]}
=
f_1
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
f_2
\left(
W^{[2]}\mathbf{a}^{[1]}
+
\mathbf{b}^{[2]}
\right)
```

and finally:

```math
\hat{y}
=
f_3
\left(
W^{[3]}\mathbf{a}^{[2]}
+
\mathbf{b}^{[3]}
\right)
```

This nested sequence of transformations is the essence of a feed-forward neural network.

---

## 1.15 Why Layers Need Different Weights

Every layer has its own weight matrix and bias vector.

For example:

```math
W^{[1]},\mathbf{b}^{[1]}
```

belong to layer 1, while:

```math
W^{[2]},\mathbf{b}^{[2]}
```

belong to layer 2.

These parameters are learned independently because different layers perform different transformations.

If every layer used identical parameters, the network would lose much of its ability to learn different representations.

During training, all trainable weights and biases across the network are adjusted to reduce the loss.

---

## 1.16 Network Architecture as a Hyperparameter

The structure of the network is not normally learned automatically in a basic neural-network training process.

Choices such as:

- number of hidden layers
- neurons per layer
- activation functions
- output-layer structure

are architectural **hyperparameters**.

For example:

```text
Architecture A

Input
 ↓
32 neurons
 ↓
Output
```

versus:

```text
Architecture B

Input
 ↓
128 neurons
 ↓
64 neurons
 ↓
32 neurons
 ↓
Output
```

Both networks may solve the same task, but they differ substantially in capacity and computational cost.

Architecture design therefore involves balancing:

```text
Model Capacity
     ↕
Training Cost
     ↕
Generalization
```

---

## 1.17 Underfitting and Overfitting

A network that is too small may lack sufficient capacity to model the data.

This can cause **underfitting**.

```text
Too little capacity
        ↓
Cannot capture important patterns
        ↓
Poor training performance
        ↓
Poor test performance
```

A network that is unnecessarily large may memorize training-specific patterns.

This can contribute to **overfitting**.

```text
Very high capacity
        ↓
Fits training data extremely well
        ↓
May learn noise
        ↓
Poorer generalization
```

Therefore:

```text
Bigger network ≠ automatically better network
```

Architecture must match the complexity of the problem and available data.

---

## 1.18 Universal Approximation Intuition

A neural network with sufficient capacity and suitable non-linear activation functions can approximate a very broad class of functions.

This idea is associated with the **universal approximation theorem**.

The practical interpretation is not:

```text
One neural network can magically solve everything
```

Rather, it tells us that neural networks have extremely powerful representational capacity.

However, the theorem does not guarantee:

- that training will be easy
- that the required network will be small
- that enough data is available
- that optimization will find the best parameters
- that the model will generalize well

Representational capability and practical trainability are different issues.

---

## 1.19 Network Architecture Example

Consider a binary-classification problem with 8 input features.

Suppose the network architecture is:

```text
Input
8 features
   ↓
Hidden Layer 1
16 neurons
ReLU
   ↓
Hidden Layer 2
8 neurons
ReLU
   ↓
Output Layer
1 neuron
Sigmoid
```

The first weight matrix has shape:

```math
W^{[1]}
\in
\mathbb{R}^{16\times8}
```

The second:

```math
W^{[2]}
\in
\mathbb{R}^{8\times16}
```

The output layer:

```math
W^{[3]}
\in
\mathbb{R}^{1\times8}
```

The forward computation is:

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

then:

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

and:

```math
\hat{y}
=
\sigma
\left(
W^{[3]}\mathbf{a}^{[2]}
+
b^{[3]}
\right)
```

The network has converted 8 raw input features into several successive learned representations before producing its final probability.

---

## 1.20 Architecture Notation

A convenient shorthand for the previous network is:

```text
8 → 16 → 8 → 1
```

meaning:

```text
8 inputs
   ↓
16 hidden neurons
   ↓
8 hidden neurons
   ↓
1 output neuron
```

Including activations:

```text
8
↓
Dense(16, ReLU)
↓
Dense(8, ReLU)
↓
Dense(1, Sigmoid)
```

This notation is commonly useful when describing architectures compactly.

---

## 1.21 Key Takeaways

- Neural networks organize artificial neurons into layers.
- The three main layer types are input, hidden, and output layers.
- The input layer represents the original features.
- Hidden layers learn intermediate representations.
- The output layer structure depends on the prediction task.
- A fully connected or dense layer connects every neuron to every neuron in the next layer.
- A whole layer can be represented using matrix multiplication.
- The number of trainable parameters in a dense layer is:

```math
n_{\text{out}}(n_{\text{in}}+1)
```

- **Depth** refers to successive computational layers.
- **Width** refers to the number of neurons in a layer.
- More depth and width increase model capacity but also increase computational cost and overfitting risk.
- Feed-forward networks pass information from input toward output without cycles.
- An MLP is a feed-forward network containing one or more fully connected hidden layers.
- Network architecture is primarily a hyperparameter choice.
- Deep networks learn hierarchical or progressively transformed representations of their inputs.
- A larger neural network is not automatically a better neural network.

### Memory Hook

```text
Neuron
   ↓
Layer
   ↓
Network

Input Layer
= Features

Hidden Layers
= Learned Representations

Output Layer
= Prediction

Width
= Neurons per Layer

Depth
= Number of Successive Layers

Dense Layer
= Every Input Connected
  to Every Neuron

MLP
= Feed-Forward
+ Fully Connected
+ Hidden Layers
```
---
