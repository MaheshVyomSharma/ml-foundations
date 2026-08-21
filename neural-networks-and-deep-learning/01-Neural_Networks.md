# 01. Neural Networks — Big Picture & Biological Inspiration

## 1. What Is a Neural Network?

An **Artificial Neural Network (ANN)** is a machine learning model composed of interconnected computational units called **artificial neurons**.

Each neuron receives one or more inputs, performs a mathematical operation on them, and produces an output.

At its simplest, a neuron computes:

```math
z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
```

or, in vector form:

```math
z = \mathbf{w}^T\mathbf{x} + b
```

where:

- `x` represents the input features
- `w` represents the weights associated with those features
- `b` is the bias
- `z` is the weighted linear combination

The neuron usually then passes `z` through an **activation function**:

```math
a = f(z)
```

where `a` becomes the neuron's output.

This tiny operation is the fundamental building block of even very large neural networks.

---

## 2. The Connection to Classical Machine Learning

A neural network should not be viewed as something completely separate from classical machine learning.

Consider linear regression:

```math
\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
```

This is essentially the same weighted computation performed inside an artificial neuron.

The important difference is that neural networks typically apply a **non-linear activation function** after this linear operation:

```math
a = f(\mathbf{w}^T\mathbf{x} + b)
```

This gives us a useful progression:

```text
Linear Regression
        ↓
Weighted Linear Combination
        ↓
Artificial Neuron
        ↓
Activation Function
        ↓
Multiple Connected Neurons
        ↓
Neural Network
        ↓
Deep Neural Network
```

The mathematical foundations from classical ML therefore carry directly into neural networks.

---

## 3. Why Do We Need Neural Networks?

A linear model can only directly represent relationships of the form:

```math
\hat{y} = \mathbf{w}^T\mathbf{x} + b
```

Geometrically, this produces a **linear decision boundary**.

In two dimensions, that boundary is a line:

```math
w_1x_1 + w_2x_2 + b = 0
```

In three dimensions, it becomes a plane.

In higher dimensions, it is called a **hyperplane**.

Many real-world relationships, however, are highly non-linear.

Examples include:

- recognizing objects in images
- understanding speech
- detecting complex patterns in sensor data
- predicting relationships involving many interacting variables
- understanding natural language

Neural networks address this by combining many neurons with **non-linear activation functions**.

A network can therefore learn much more complicated mappings:

```math
\mathbf{x} \rightarrow \mathbf{y}
```

rather than being restricted to a single linear relationship.

---

## 4. Why Multiple Linear Layers Alone Are Not Enough

Suppose one neural-network layer performs:

```math
\mathbf{h} = W_1\mathbf{x} + \mathbf{b}_1
```

and another performs:

```math
\mathbf{y} = W_2\mathbf{h} + \mathbf{b}_2
```

Substituting the first equation into the second:

```math
\mathbf{y}
=
W_2(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2
```

Expanding:

```math
\mathbf{y}
=
W_2W_1\mathbf{x}
+
W_2\mathbf{b}_1
+
\mathbf{b}_2
```

The entire expression can still be rewritten as:

```math
\mathbf{y} = W\mathbf{x} + \mathbf{b}
```

So stacking linear transformations does **not** make the network fundamentally more powerful.

Ten linear layers can mathematically collapse into one linear transformation.

This is why **activation functions are essential**.

They introduce non-linearity between layers:

```math
\mathbf{h} = f(W_1\mathbf{x}+\mathbf{b}_1)
```

Now the layers can no longer simply collapse into a single linear transformation.

This allows the network to represent complex non-linear relationships.

---

## 5. Biological Inspiration

Artificial neural networks were loosely inspired by biological neurons.

A biological neuron contains structures such as:

- **dendrites** — receive signals
- **cell body (soma)** — integrates incoming signals
- **axon** — carries the output signal
- **synapses** — connections between neurons

A simplified analogy is:

| Biological neuron | Artificial neuron |
|---|---|
| Dendrites | Inputs |
| Synaptic strength | Weights |
| Soma | Weighted summation |
| Firing behaviour | Activation function |
| Axon | Output |

Conceptually:

```text
Biological neuron

Signals
   ↓
Dendrites
   ↓
Cell body
   ↓
Neuron fires
   ↓
Axon
```

Artificial neuron:

```text
Inputs
   ↓
Weights
   ↓
Weighted Sum + Bias
   ↓
Activation Function
   ↓
Output
```

The analogy is useful for intuition, but artificial neural networks are **mathematical models**, not realistic simulations of biological brains.

Modern neural-network design is driven primarily by mathematics, optimization, computational efficiency, and empirical performance.

---

## 6. From One Neuron to a Network

A single neuron receives several inputs:

```math
x_1,x_2,\ldots,x_n
```

and computes:

```math
z = \sum_{i=1}^{n} w_ix_i + b
```

followed by:

```math
a = f(z)
```

Now imagine several neurons receiving the same inputs.

Each neuron has its own weights and bias:

```text
x₁ ─┬──→ neuron 1
x₂ ─┼──→ neuron 2
x₃ ─┼──→ neuron 3
x₄ ─┴──→ neuron 4
```

Together these neurons form a **layer**.

The outputs of that layer can then become the inputs to another layer:

```text
Input Layer
     ↓
Hidden Layer
     ↓
Hidden Layer
     ↓
Output Layer
```

A network containing multiple intermediate layers is generally called a **deep neural network**.

This is the origin of the term **deep learning**: the model learns through multiple successive layers of representation.

---

## 7. What Does a Neural Network Actually Learn?

A neural network primarily learns its **weights and biases**.

Suppose a neuron computes:

```math
z = w_1x_1+w_2x_2+b
```

Initially, the weights do not contain useful knowledge.

The network makes a prediction:

```math
\hat{y}
```

The prediction is compared with the true target:

```math
y
```

A **loss function** measures how wrong the prediction is:

```math
L(y,\hat{y})
```

Training then repeatedly adjusts the weights and biases in a direction that reduces this loss.

The basic learning cycle is:

```text
Input
  ↓
Prediction
  ↓
Compare with Ground Truth
  ↓
Calculate Loss
  ↓
Determine how parameters contributed to the error
  ↓
Update Weights and Biases
  ↓
Repeat
```

The mechanisms responsible for these steps are **forward propagation, loss functions, backpropagation, and optimization**.

---

## 8. Parameters vs Hyperparameters

An important distinction in neural networks is between **parameters** and **hyperparameters**.

### 8.1. Parameters

Parameters are values learned automatically from training data.

Examples:

- weights
- biases

If a neuron contains:

```math
z = w_1x_1+w_2x_2+b
```

then:

```math
w_1,\;w_2,\;b
```

are trainable parameters.

### 8.2. Hyperparameters

Hyperparameters are choices that control the model or training process.

Examples include:

- number of hidden layers
- number of neurons per layer
- learning rate
- batch size
- number of epochs
- activation function
- optimizer
- regularization strength

A useful distinction is:

```text
Parameters      → learned by the model
Hyperparameters → chosen/configured for the model
```

---

## 9. What Makes Deep Learning Different?

Traditional machine-learning workflows often depend heavily on **feature engineering**.

For example, when classifying an image using classical ML, we might manually derive features describing:

- edges
- shapes
- textures
- colours

and then provide those engineered features to a classifier.

Deep neural networks can learn useful intermediate representations directly from the data.

For an image network, the progression might loosely resemble:

```text
Pixels
  ↓
Edges
  ↓
Textures
  ↓
Shapes
  ↓
Object parts
  ↓
Objects
```

These representations are not normally programmed manually. They emerge through training.

This ability to perform **representation learning** is one of the major strengths of deep learning.

---

## 10. Neural Networks Are Still Machine Learning

Despite their complexity, neural networks follow the same fundamental supervised-learning framework encountered in classical ML:

```text
Training Data
     ↓
Model
     ↓
Prediction
     ↓
Loss / Error
     ↓
Optimization
     ↓
Improved Model
```

Many familiar ideas therefore remain relevant:

- train/validation/test splits
- regression and classification
- loss functions
- overfitting
- regularization
- feature scaling
- model evaluation
- hyperparameter tuning
- generalization

Deep learning extends these ideas rather than replacing them.

---

## 11. Core Mathematical View

At the heart of a neural network are concepts already encountered in the mathematical foundations of ML.

### 11.1. Linear Algebra

Inputs and weights are represented using vectors and matrices:

```math
\mathbf{z} = W\mathbf{x}+\mathbf{b}
```

### 11.2. Functions

Activation functions transform the weighted result:

```math
\mathbf{a}=f(\mathbf{z})
```

### 11.3. Calculus

Derivatives tell us how changing a parameter affects the loss:

```math
\frac{\partial L}{\partial w}
```

### 11.4. Probability and Statistics

Classification outputs are frequently interpreted probabilistically:

```math
P(y=k\mid\mathbf{x})
```

### 11.5. Optimization

Parameters are updated to reduce the loss:

```math
w_{\text{new}}
=
w_{\text{old}}
-
\eta\frac{\partial L}{\partial w}
```

where `η` is the **learning rate**.

Deep learning therefore brings together:

```text
Linear Algebra
      +
Calculus
      +
Probability & Statistics
      +
Optimization
      ↓
Neural Networks
```

---

## 12. The Complete Picture

The fundamental neural-network pipeline can be summarized as:

```text
Input Features
      ↓
Weighted Sum + Bias
      ↓
Activation Function
      ↓
Hidden Representations
      ↓
More Neural Layers
      ↓
Prediction
      ↓
Loss Function
      ↓
Backpropagation
      ↓
Gradient-Based Optimization
      ↓
Updated Weights and Biases
```

Training repeats this process many times until the network learns parameters that generalize sufficiently well to unseen data.

---

## 13. Key Takeaways

- An artificial neuron computes a weighted sum of its inputs plus a bias.
- The basic computation strongly resembles linear and logistic regression.
- Activation functions introduce the non-linearity required to learn complex relationships.
- Multiple linear layers without activation functions still collapse into a linear transformation.
- Multiple neurons form layers, and multiple layers form neural networks.
- Deep learning refers to neural networks containing multiple successive representation-learning layers.
- Neural networks primarily learn **weights and biases**.
- Training minimizes a **loss function** by adjusting these parameters.
- Parameters are learned; hyperparameters configure the architecture and training process.
- Neural networks are mathematical models loosely inspired by biological neurons, not simulations of the brain.
- Deep learning builds directly upon linear algebra, calculus, probability, statistics, and optimization.
- Classical ML concepts such as overfitting, regularization, evaluation, and generalization remain fully relevant.

### 13.1. Memory Hook

```text
Neuron
= Weighted Sum
+ Bias
+ Activation

Network
= Many Connected Neurons

Learning
= Prediction
+ Loss
+ Gradients
+ Parameter Updates

Deep Learning
= Neural Networks
+ Multiple Representation Layers
```
---

## 14. Perceptron and Artificial Neuron

## 15. The Artificial Neuron

The **artificial neuron** is the fundamental computational unit of a neural network.

It receives several input values, assigns a weight to each input, adds a bias, and passes the resulting value through an activation function.

For inputs:

```math
x_1, x_2, \ldots, x_n
```

with corresponding weights:

```math
w_1, w_2, \ldots, w_n
```

the neuron first calculates:

```math
z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
```

or equivalently:

```math
z = \sum_{i=1}^{n} w_i x_i + b
```

The result is then passed through an activation function:

```math
a = f(z)
```

Therefore, the complete neuron can be represented as:

```math
a = f\left(\sum_{i=1}^{n} w_i x_i + b\right)
```

The basic flow is:

```text
Inputs
  ↓
Weights
  ↓
Weighted Sum + Bias
  ↓
Activation Function
  ↓
Output
```

---

## 16. What Do the Weights Represent?

A **weight** determines how strongly an input influences the neuron's output.

Consider:

```math
z = 2x_1 - 0.5x_2 + 4x_3 + b
```

Here:

- `x₁` has weight `2`
- `x₂` has weight `-0.5`
- `x₃` has weight `4`

The magnitude of a weight represents the strength of its influence.

A larger absolute weight generally means that changes in that input have a stronger effect on the neuron.

The sign represents the direction of the influence:

```text
Positive weight → pushes z upward
Negative weight → pushes z downward
Weight near zero → little influence
```

Weights are not normally assigned manually. They are **learned during training**.

---

## 17. What Does the Bias Do?

The **bias** is an additional trainable parameter that allows the neuron to shift its response independently of the input values.

Without bias:

```math
z = \mathbf{w}^T\mathbf{x}
```

With bias:

```math
z = \mathbf{w}^T\mathbf{x} + b
```

Consider a two-dimensional decision boundary:

```math
w_1x_1 + w_2x_2 + b = 0
```

Without the bias:

```math
w_1x_1 + w_2x_2 = 0
```

the boundary is constrained to pass through the origin.

Adding `b` allows the boundary to shift away from the origin.

This is directly analogous to the intercept in linear regression:

```math
y = mx + c
```

where `c` allows the line to move vertically rather than forcing it through `(0,0)`.

### 17.1. Memory Hook

```text
Weights → control orientation and influence
Bias    → allows shifting
```

---

## 18. Vector Representation

Instead of writing:

```math
z = w_1x_1+w_2x_2+\cdots+w_nx_n+b
```

we can represent the inputs and weights as vectors:

```math
\mathbf{x}
=
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
```

and:

```math
\mathbf{w}
=
\begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_n
\end{bmatrix}
```

Then:

```math
z = \mathbf{w}^T\mathbf{x}+b
```

because:

```math
\mathbf{w}^T\mathbf{x}
=
w_1x_1+w_2x_2+\cdots+w_nx_n
```

This vector representation becomes extremely important because neural networks perform these calculations for many neurons simultaneously using **matrix operations**.

---

## 19. The Perceptron

The **perceptron** is one of the earliest artificial-neuron models.

It was introduced by Frank Rosenblatt in the 1950s as a model for binary classification.

A classical perceptron calculates:

```math
z = \mathbf{w}^T\mathbf{x}+b
```

and applies a **step function**.

One common form is:

```math
f(z)
=
\begin{cases}
1, & z \geq 0 \\
0, & z < 0
\end{cases}
```

Therefore:

```math
\hat{y}
=
\begin{cases}
1, & \mathbf{w}^T\mathbf{x}+b \geq 0 \\
0, & \mathbf{w}^T\mathbf{x}+b < 0
\end{cases}
```

The perceptron therefore acts as a binary classifier.

---

## 20. Perceptron as a Decision Boundary

The perceptron switches between its two output classes when:

```math
\mathbf{w}^T\mathbf{x}+b=0
```

For two features:

```math
w_1x_1+w_2x_2+b=0
```

Rearranging:

```math
x_2
=
-\frac{w_1}{w_2}x_1
-
\frac{b}{w_2}
```

This is the equation of a straight line.

The perceptron therefore creates a **linear decision boundary**.

Points on one side satisfy:

```math
\mathbf{w}^T\mathbf{x}+b > 0
```

while points on the other side satisfy:

```math
\mathbf{w}^T\mathbf{x}+b < 0
```

The labels assigned to the two sides can be swapped. What fundamentally matters is that the boundary separates the classes.

In higher-dimensional spaces, this boundary becomes a **hyperplane**.

---

## 21. Linear Separability

A dataset is **linearly separable** if a single straight line, plane, or hyperplane can completely separate its classes.

For example:

```text
○ ○ ○       × × ×
 ○ ○         × ×
○ ○ ○       × × ×
      ↑
A straight boundary can separate them
```

A perceptron can learn this type of classification problem.

However, consider a pattern such as XOR:

```text
Class 0        Class 1

(0,0)            (0,1)

(1,1)            (1,0)
```

The classes are arranged diagonally.

No single straight line can separate them correctly.

Therefore, a single perceptron cannot solve XOR.

This limitation became historically important because it demonstrated that **single-layer perceptrons cannot represent arbitrary non-linear decision boundaries**.

---

## 22. Why Multiple Neurons Matter

A single perceptron gives us one linear boundary:

```math
\mathbf{w}^T\mathbf{x}+b=0
```

But multiple neurons can learn different boundaries.

Conceptually:

```text
Neuron 1 → learns boundary A
Neuron 2 → learns boundary B
Neuron 3 → learns boundary C
              ↓
        combine outputs
              ↓
      complex decision region
```

Once these neurons are connected through layers with suitable non-linear activation functions, the network can combine simple boundaries into much more complicated ones.

This is one of the central ideas behind neural networks:

> Complex non-linear behaviour can emerge from many relatively simple computational units working together.

---

## 23. Perceptron vs Logistic Regression

The perceptron and logistic regression have very similar internal linear computations.

Both begin with:

```math
z = \mathbf{w}^T\mathbf{x}+b
```

The difference is what happens afterward.

### 23.1. Perceptron

Uses a hard step function:

```math
\hat{y}
=
\begin{cases}
1, & z \geq 0 \\
0, & z < 0
\end{cases}
```

Its output directly represents a class.

### 23.2. Logistic Regression

Uses the sigmoid function:

```math
\sigma(z)=\frac{1}{1+e^{-z}}
```

giving:

```math
\hat{p}=\sigma(\mathbf{w}^T\mathbf{x}+b)
```

Its output lies between `0` and `1` and can be interpreted as a probability under the model.

A threshold can then convert that probability into a class prediction.

Therefore:

```text
Perceptron
Linear combination → Step → Class

Logistic Regression
Linear combination → Sigmoid → Probability → Threshold → Class
```

This connection is important because a sigmoid neuron is mathematically very close to logistic regression.

---

## 24. Perceptron Learning

The perceptron learns by modifying its weights when it makes classification mistakes.

A simplified weight-update rule is:

```math
w_i
\leftarrow
w_i
+
\eta(y-\hat{y})x_i
```

and the bias can similarly be updated as:

```math
b
\leftarrow
b
+
\eta(y-\hat{y})
```

where:

- `η` is the learning rate
- `y` is the true class
- `ŷ` is the predicted class
- `xᵢ` is the corresponding input

If the prediction is correct:

```math
y-\hat{y}=0
```

so no correction is necessary.

If the prediction is incorrect, the weights are adjusted in a direction intended to improve the classification.

This gives the fundamental learning idea:

```text
Predict
   ↓
Compare with truth
   ↓
Wrong?
   ↓
Adjust parameters
   ↓
Predict again
```

Modern neural networks use considerably more sophisticated learning mechanisms, but the basic principle survives.

---

## 25. Artificial Neuron vs Classical Perceptron

The terms **artificial neuron** and **perceptron** are sometimes used loosely as though they mean exactly the same thing.

It is useful to keep a distinction.

A classical perceptron specifically uses a threshold or step activation for binary classification.

A modern artificial neuron is the more general structure:

```math
a=f(\mathbf{w}^T\mathbf{x}+b)
```

where `f` could be:

- sigmoid
- tanh
- ReLU
- Leaky ReLU
- another activation function
- or even no activation in certain output layers

Therefore:

```text
Perceptron
   ↓
Specific early artificial-neuron model

Artificial Neuron
   ↓
General computational building block
of modern neural networks
```

---

## 26. One Neuron vs an Entire Layer

For one neuron:

```math
z = \mathbf{w}^T\mathbf{x}+b
```

For several neurons, each neuron has its own weights and bias.

Instead of calculating them separately, we collect their weights into a matrix:

```math
\mathbf{z}=W\mathbf{x}+\mathbf{b}
```

Then apply the activation function:

```math
\mathbf{a}=f(\mathbf{z})
```

This single expression represents the computation performed by an **entire neural-network layer**.

This is where the linear algebra foundation becomes extremely useful.

Instead of thinking:

```text
Neuron 1 calculation
Neuron 2 calculation
Neuron 3 calculation
...
Neuron 1000 calculation
```

we think:

```math
\mathbf{a}=f(W\mathbf{x}+\mathbf{b})
```

Modern deep-learning frameworks perform these matrix operations efficiently on CPUs, GPUs, and other accelerators.

---

## 27. Key Takeaways

- An artificial neuron receives inputs, multiplies them by weights, adds a bias, and applies an activation function.
- **Weights** determine the strength and direction of input influence.
- **Bias** allows the neuron's response or decision boundary to shift.
- A classical **perceptron** uses a step function and performs binary classification.
- A single perceptron creates a **linear decision boundary**.
- A perceptron can solve linearly separable problems but cannot solve inherently non-linearly separable problems such as XOR.
- Multiple neurons and non-linear activation functions allow neural networks to construct complex decision boundaries.
- Logistic regression and the perceptron both begin with the same weighted linear combination.
- Logistic regression uses sigmoid; the classical perceptron uses a hard threshold.
- Perceptron learning adjusts weights in response to classification errors.
- Modern artificial neurons generalize the original perceptron idea.
- Multiple neurons can be calculated simultaneously using matrix operations.

### 27.1. Memory Hook

```text
Artificial Neuron
= Inputs
× Weights
+ Bias
→ Activation

Perceptron
= Artificial Neuron
+ Step Function

One Perceptron
= One Linear Boundary

Many Neurons
+ Non-linearity
= Complex Boundaries

One Neuron:
z = wᵀx + b

One Layer:
z = Wx + b
```
---
