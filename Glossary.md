# Machine Learning Foundations Glossary

This glossary is a dictionary-style reference for important terms used throughout the project.

Each entry separates meanings that depend on the subject area.

> **Sample version:** This first version contains a small sample for review before the full glossary is developed.

---

## Activation Function

### Neural Networks

A function applied to a neuron’s pre-activation value to produce its output. Activation functions introduce nonlinearity, allowing stacked layers to represent nonlinear relationships.

Common examples include sigmoid, hyperbolic tangent, ReLU, leaky ReLU, and softmax.

**Reference:** [Activation Functions](neural-networks-and-deep-learning/02-Activation_Functions.md)

### Related Terms

`neuron` · `ReLU` · `sigmoid` · `softmax`

## Affine Transformation

### Linear Algebra and Machine Learning

A transformation of the form:

```math
f(\mathbf{x}) = A\mathbf{x} + \mathbf{b}
```

It combines a linear transformation with a translation. A nonzero bias means the origin does not necessarily remain fixed.

**Reference:** [Linear Algebra](math-for-ml/01-Linear_Algebra.md#machine-learning-connection)

### Related Terms

`bias` · `linear transformation` · `matrix multiplication`

## Attention

### Neural Networks

A mechanism that assigns different weights to positions in an input sequence so that a representation can selectively use the most relevant information.

Self-attention compares queries, keys, and values from the same sequence. Multi-head attention performs several such comparisons in parallel.

**Reference:** [Attention Mechanism](neural-networks-and-deep-learning/20-Attention_Mechanism.md)

### Related Terms

`query` · `key` · `value` · `Transformer`

## Backpropagation

### Neural Networks

An algorithm that applies the chain rule from the output layer back through earlier layers to compute gradients of the loss with respect to weights and biases.

Those gradients are then used by an optimization algorithm to update the parameters.

**Reference:** [Backpropagation](neural-networks-and-deep-learning/07-Backpropagation.md)

### Related Terms

`chain rule` · `gradient` · `loss function` · `gradient descent`

## Batch

### Neural Networks and Optimization

A subset of training examples processed together to compute a loss or gradient update.

Full-batch gradient descent uses the entire dataset, stochastic gradient descent uses one example, and mini-batch gradient descent uses a small subset.

**Reference:** [Calculus](math-for-ml/02-Calculus.md#gradient-descent), [Training a Neural Network](neural-networks-and-deep-learning/08-Training_a_Neural_Network.md)

### Related Terms

`epoch` · `gradient descent` · `stochastic gradient descent`

## Bias

### Neural Networks

A trainable scalar or vector added to a weighted input before the activation function. For a single neuron,

```math
z = \mathbf{w}^{T}\mathbf{x} + b
```

The bias lets the neuron shift its response or decision boundary away from the origin. Each output neuron generally has its own bias.

**Reference:** [Neural Networks](neural-networks-and-deep-learning/01-Neural_Networks.md#2-1-the-artificial-neuron), [Linear Algebra](math-for-ml/01-Linear_Algebra.md#machine-learning-connection), [Calculus](math-for-ml/02-Calculus.md#step-6-differentiate-with-respect-to-the-bias)

### Linear Algebra

The additive vector in an affine transformation:

```math
f(\mathbf{x}) = W\mathbf{x} + \mathbf{b}
```

Unlike a purely linear transformation, an affine transformation with a nonzero bias need not map the origin to the origin.

**Reference:** [Linear Algebra](math-for-ml/01-Linear_Algebra.md#machine-learning-connection)

### Calculus and Optimization

The parameter $b$ in a model such as:

```math
\hat{y} = wx + b
```

It is differentiated and updated like any other trainable parameter. For a single neuron, $\partial z / \partial b = 1$.

**Reference:** [Calculus](math-for-ml/02-Calculus.md#step-6-differentiate-with-respect-to-the-bias)

### Classical Machine Learning

In the project’s linear-model notation, the same role is usually named the **intercept** rather than the bias. It is the predicted value when all features are zero, subject to the model’s feature representation.

**Reference:** [Linear Regression](classical-ml/01-Linear_Regression.md#mathematical-foundation)

### Statistics

For an estimator $\hat{\theta}$ of a parameter $\theta$, statistical bias is the systematic difference between the estimator’s expected value and the true parameter:

```math
\mathrm{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta
```

An unbiased estimator has bias equal to zero.

This is distinct from the trainable bias parameter in a neural network, although both ideas involve systematic displacement from a target.

**Reference:** [Statistics for Machine Learning](math-for-ml/04-Statistics_for_Machine_Learning.md#bias)

### Bias–Variance Trade-off

In model evaluation, bias describes error caused by a model family being too restrictive to represent the underlying relationship adequately. High bias is commonly associated with underfitting.

**Reference:** [Statistics for Machine Learning](math-for-ml/04-Statistics_for_Machine_Learning.md#bias-and-variance)

### Related Terms

`affine transformation` · `intercept` · `parameter` · `underfitting` · `variance`

---

## Classification

### Classical and Neural Machine Learning

A supervised-learning task in which the target is a category or class rather than a continuous numerical value.

Binary classification has two classes; multiclass classification has more than two; multilabel classification allows multiple labels per example.

**Reference:** [Logistic Regression](classical-ml/03-Logistic_Regression.md), [Neural Networks](neural-networks-and-deep-learning/01-Neural_Networks.md)

### Related Terms

`class` · `label` · `logistic regression` · `regression`

## Cost Function

### Classical Machine Learning

A scalar measure of the total error made by a model over a dataset. Training attempts to minimize it by changing model parameters.

The cost is often an average or sum of per-example losses.

**Reference:** [Gradient Descent](classical-ml/02-Gradient_Descent.md)

### Related Terms

`loss function` · `objective function` · `optimization`

## Covariance Matrix

### Statistics and Machine Learning

A square matrix whose diagonal contains variable variances and whose off-diagonal entries contain pairwise covariances.

It is symmetric and positive semidefinite.

**Reference:** [Types of Matrices](math-for-ml/Appendix-A_Types_of_Matrices.md#covariance-matrix)

### Related Terms

`correlation` · `variance` · `positive semidefinite`

## Decision Boundary

### Classification

The boundary in feature space that separates different predicted classes. For a linear classifier, it is described by an equation such as:

```math
\mathbf{w}^{T}\mathbf{x} + b = 0
```

The bias shifts the boundary, while the weights determine its orientation.

**Reference:** [Neural Networks](neural-networks-and-deep-learning/01-Neural_Networks.md#2-3-what-does-the-bias-do), [Logistic Regression](classical-ml/03-Logistic_Regression.md)

### Related Terms

`classification` · `hyperplane` · `perceptron`

## Derivative

### Calculus

The instantaneous rate at which a function changes with respect to one of its variables. Geometrically, it is the slope of the tangent at a point.

**Reference:** [Calculus](math-for-ml/02-Calculus.md)

### Related Terms

`gradient` · `partial derivative` · `chain rule`

## Determinant

### Linear Algebra

A scalar associated with a square matrix. It indicates, among other things, the signed scaling of area or volume under the transformation.

A square matrix is singular when its determinant is zero and invertible when its determinant is nonzero.

**Reference:** [Linear Algebra](math-for-ml/01-Linear_Algebra.md)

### Related Terms

`inverse` · `singular matrix` · `eigenvalue`

## Dot Product

### Linear Algebra

For vectors of the same dimension, the sum of pairwise products:

```math
\mathbf{a}^{T}\mathbf{b} = \sum_i a_i b_i
```

It measures alignment and is zero for perpendicular vectors.

**Reference:** [Linear Algebra](math-for-ml/01-Linear_Algebra.md)

### Related Terms

`inner product` · `norm` · `vector`

## Eigenvalue and Eigenvector

### Linear Algebra

For a square matrix $A$, a nonzero vector $\mathbf{v}$ is an eigenvector if:

```math
A\mathbf{v} = \lambda\mathbf{v}
```

The scalar $\lambda$ is its eigenvalue. The matrix changes the vector’s scale, but not its direction, apart from a possible sign reversal.

**Reference:** [Linear Algebra](math-for-ml/01-Linear_Algebra.md)

### Related Terms

`eigendecomposition` · `singular value decomposition` · `matrix`

## Epoch

### Neural Networks

One complete pass through all training examples. If the dataset has $N$ examples and the batch size is $B$, one epoch usually contains approximately $N/B$ parameter updates.

**Reference:** [Training a Neural Network](neural-networks-and-deep-learning/08-Training_a_Neural_Network.md)

### Related Terms

`batch` · `iteration` · `training`

## Feature

### Machine Learning

An input variable or measurable attribute supplied to a model. A dataset is commonly represented as rows of observations and columns of features.

**Reference:** [Linear Regression](classical-ml/01-Linear_Regression.md), [Types of Matrices](math-for-ml/Appendix-A_Types_of_Matrices.md#design-matrix)

### Related Terms

`input` · `target` · `design matrix` · `feature engineering`

## Gradient

### Calculus and Optimization

The vector of partial derivatives of a scalar-valued function with respect to its variables:

```math
\nabla f =
\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\vdots\\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
```

It points in the direction of steepest increase. Negative-gradient methods therefore move toward decreasing function values.

**Reference:** [Calculus](math-for-ml/02-Calculus.md), [Gradient Descent](classical-ml/02-Gradient_Descent.md)

### Related Terms

`derivative` · `gradient descent` · `Hessian`

## Gradient Descent

### Optimization

An iterative optimization method that updates parameters in the direction opposite the gradient:

```math
\theta_{\mathrm{new}} = \theta_{\mathrm{old}} - \eta\nabla J(\theta)
```

Here, $\eta$ is the learning rate and $J$ is the objective or cost function.

**Reference:** [Gradient Descent](classical-ml/02-Gradient_Descent.md), [Calculus](math-for-ml/02-Calculus.md)

### Related Terms

`gradient` · `learning rate` · `optimizer` · `parameter`

## Hessian

### Calculus and Optimization

The matrix of second-order partial derivatives of a scalar-valued function. It describes local curvature and helps distinguish locally convex, concave, and saddle-shaped behavior.

**Reference:** [Types of Matrices](math-for-ml/Appendix-A_Types_of_Matrices.md#hessian-matrix)

### Related Terms

`curvature` · `gradient` · `positive definite`

## Hyperparameter

### Machine Learning

A configuration value chosen before or outside parameter training. Examples include learning rate, batch size, tree depth, number of clusters, and regularization strength.

Hyperparameters are not normally learned directly by gradient descent.

**Reference:** [Hyperparameters and Hyperparameter Tuning](neural-networks-and-deep-learning/15-Hyperparameters_and_Hyperparameter_Tuning.md)

### Related Terms

`parameter` · `learning rate` · `model selection` · `regularization`

## Intercept

### Classical Machine Learning

The constant term in a regression equation. In a one-feature model,

```math
\hat{y} = mx + c
```

the intercept $c$ is the predicted value when $x=0$. In a multi-feature model, it is the prediction when every represented feature is zero.

**Reference:** [Linear Regression](classical-ml/01-Linear_Regression.md#mathematical-foundation)

### Neural Networks

For a neuron, the intercept-like role is played by the **bias** $b$:

```math
z = \mathbf{w}^{T}\mathbf{x} + b
```

The terms are closely related, but “bias” is the standard neural-network name and “intercept” is the standard regression name.

**Reference:** [Neural Networks](neural-networks-and-deep-learning/01-Neural_Networks.md#2-3-what-does-the-bias-do)

### Related Terms

`bias` · `coefficient` · `feature` · `linear regression` · `parameter`

## Learning Rate

### Optimization

A hyperparameter that controls the size of each parameter update during training. It is usually represented by $\eta$ and appears in the gradient-descent update rule:

```math
\theta \leftarrow \theta - \eta\nabla J(\theta)
```

A learning rate that is too small can make training unnecessarily slow, while one that is too large can cause the objective to overshoot or diverge.

**Reference:** [Gradient Descent](neural-networks-and-deep-learning/06-Gradient_Descent.md#19-the-learning-rate)

### Related Terms

`gradient descent` · `hyperparameter` · `optimizer` · `parameter`

---

## Loss Function

### Machine Learning

A function that measures the error for one training example or a small group of examples. A cost function commonly aggregates losses over a complete dataset.

**Reference:** [Loss Functions](neural-networks-and-deep-learning/05-Loss_Functions.md)

### Related Terms

`cost function` · `objective function` · `mean squared error` · `cross-entropy`

## Matrix

### Linear Algebra

A rectangular arrangement of numbers with a defined number of rows and columns. An $m \times n$ matrix has $m$ rows and $n$ columns.

Matrices represent datasets, linear transformations, parameter collections, and relationships between variables.

**Reference:** [Linear Algebra](math-for-ml/01-Linear_Algebra.md), [Types of Matrices](math-for-ml/Appendix-A_Types_of_Matrices.md)

### Related Terms

`vector` · `transpose` · `matrix multiplication` · `rank`

## Overfitting

### Machine Learning

A model behavior in which the model fits training data or its idiosyncrasies too closely and performs worse on unseen data.

Regularization, early stopping, dropout, and more representative data can reduce overfitting.

**Reference:** [Regularization](classical-ml/04-Regularization.md), [Deep Learning Workflow](neural-networks-and-deep-learning/19-Deep_Learning_Workflow.md)

### Related Terms

`underfitting` · `generalization` · `regularization` · `variance`

## Parameter

### Machine Learning

A value learned from data during model training. Weights and biases are parameters in neural networks; slopes and intercepts are parameters in linear regression.

**Reference:** [Gradient Descent](classical-ml/02-Gradient_Descent.md), [Neural Network Architecture](neural-networks-and-deep-learning/03-Neural_Network_Architecture.md)

### Related Terms

`hyperparameter` · `weight` · `bias` · `optimization`

## Penalty

### Machine Learning

An additional cost included in a model’s objective function to discourage undesirable behavior, such as excessive complexity or large parameter values. Regularization penalties such as L1 and L2 help reduce overfitting by making complex models more costly during training.

**Reference:** [Regularization](classical-ml/04-Regularization.md)

### Related Terms

`regularization` · `loss function` · `objective function` · `overfitting`

## Regression

### Classical Machine Learning

A supervised-learning task in which the target is a continuous numerical value. Linear Regression models the target as a weighted combination of features plus an intercept.

**Reference:** [Linear Regression](classical-ml/01-Linear_Regression.md)

### Related Terms

`classification` · `feature` · `intercept` · `residual`

## Regularization

### Machine Learning

A family of methods that discourages overly complex models to improve generalization. Common methods add a penalty to the objective, constrain parameters, or stop training before the model overfits.

**Reference:** [Regularization](classical-ml/04-Regularization.md), [Regularization in Neural Networks](neural-networks-and-deep-learning/12-Regularization_in_Neural_Networks.md)

### Related Terms

`overfitting` · `underfitting` · `L1 regularization` · `L2 regularization` · `dropout`

## Saddle Point

### Calculus and Optimization

A point on a multivariable function where the gradient may be zero, but the point is neither a local minimum nor a local maximum. The function curves upward in at least one direction and downward in another. For example, $(0,0)$ is a saddle point of $f(x,y)=x^2-y^2$.

**Reference:** [Calculus](math-for-ml/02-Calculus.md#saddle-points), [Gradient Descent](neural-networks-and-deep-learning/06-Gradient_Descent.md#125-saddle-points)

### Related Terms

`gradient` · `Hessian` · `local minimum` · `stationary point`

## Singular Value Decomposition

### Linear Algebra

A factorization of a matrix $A$ into:

```math
A = U\Sigma V^T
```

where $U$ and $V$ are orthogonal and $\Sigma$ contains singular values. SVD supports dimensionality reduction, least-squares solutions, and numerical analysis.

**Reference:** [Linear Algebra](math-for-ml/01-Linear_Algebra.md)

### Related Terms

`eigenvalue` · `matrix` · `principal component analysis`

## Target

### Machine Learning

The output value a supervised-learning model is trained to predict. It is also called the response, outcome, or label, depending on the task.

**Reference:** [Linear Regression](classical-ml/01-Linear_Regression.md), [Logistic Regression](classical-ml/03-Logistic_Regression.md)

### Related Terms

`feature` · `label` · `prediction` · `supervised learning`

## Underfitting

### Machine Learning

A model behavior in which the model is too simple or too constrained to capture important structure in the data. It performs poorly even on the training data and usually has high bias.

**Reference:** [Regularization](classical-ml/04-Regularization.md), [Deep Learning Workflow](neural-networks-and-deep-learning/19-Deep_Learning_Workflow.md)

### Related Terms

`overfitting` · `bias` · `variance` · `model capacity`

## Variance

### Statistics

A measure of how far values or an estimator tend to vary around their mean or expected value. For an estimator, high variance means that its result changes substantially across different samples.

### Machine Learning

In the bias–variance trade-off, variance describes a model’s sensitivity to peculiarities of the training sample. High variance is commonly associated with overfitting.

**Reference:** [Statistics for Machine Learning](math-for-ml/04-Statistics_for_Machine_Learning.md#sample-variance), [Model Evaluation](classical-ml/12-Model_Evaluation_and_Model_Selection.md)

### Related Terms

`bias` · `overfitting` · `standard deviation`

## Vector

### Linear Algebra

An ordered collection of numbers. A vector can represent a point, direction, feature row, parameter set, or activation in a model.

It may be written as a row or column, but its orientation matters when performing matrix operations.

**Reference:** [Linear Algebra](math-for-ml/01-Linear_Algebra.md)

### Related Terms

`scalar` · `matrix` · `norm` · `dot product`

## Weight

### Machine Learning

A trainable parameter that controls the contribution of an input or previous-layer activation to a model’s output. In a neuron, weights are multiplied by inputs before the bias is added.

**Reference:** [Neural Networks](neural-networks-and-deep-learning/01-Neural_Networks.md), [Neural Network Architecture](neural-networks-and-deep-learning/03-Neural_Network_Architecture.md)

### Related Terms

`bias` · `parameter` · `weight matrix` · `gradient`
