# 02. Calculus

## 1. Why Calculus Matters in Machine Learning

Machine learning is fundamentally about **change**.

A model begins with some set of parameters — such as weights and biases — and produces predictions. Those predictions are compared with the true values using a **loss function**. Learning then consists of changing the parameters so that the loss becomes smaller.

This immediately creates a mathematical question:

> **If a model parameter changes slightly, how does the loss change?**

Calculus provides the machinery for answering precisely this question.

---

### From a Model to a Loss

Consider the simplest linear model:

```math
\hat{y}
=
wx+b
```

where:

- $x$ is the input,
- $w$ is the weight,
- $b$ is the bias,
- $\hat{y}$ is the model's prediction.

Suppose the true value is $y$. One possible measure of prediction error is squared error:

```math
L
=
(y-\hat{y})^2
```

Substituting the model prediction:

```math
L
=
\left(y-(wx+b)\right)^2
```

Now notice something important.

For a particular training example, $x$ and $y$ are fixed. But $w$ and $b$ are parameters that the model is allowed to change.

Therefore, the loss itself can be regarded as a function of the model parameters:

```math
L
=
L(w,b)
```

Learning the model means finding values of $w$ and $b$ that make this function as small as possible.

That is an **optimization problem**.

---

### Calculus Measures Change

Suppose we change the weight slightly:

```math
w
\rightarrow
w+\Delta w
```

The prediction changes:

```math
\hat{y}
\rightarrow
\hat{y}+\Delta\hat{y}
```

and consequently the loss changes:

```math
L
\rightarrow
L+\Delta L
```

The important question is not merely whether the loss changed.

We want to know:

> How sensitive is the loss to a change in $w$?

That sensitivity is measured using a **derivative**:

```math
\frac{dL}{dw}
```

Conceptually, this quantity asks:

> If I change $w$ by a very small amount, how quickly does $L$ change?

This is one of the central ideas connecting calculus to machine learning.

---

### The Landscape View

It is useful to imagine loss as a landscape.

If the model has only one parameter $w$, the loss might look conceptually like a valley:

```text
Loss
 ^
 |       \         /
 |        \       /
 |         \     /
 |          \___/
 |
 +--------------------> w
```

Every possible value of $w$ corresponds to some loss.

Training asks:

> **Where is the bottom of the valley?**

Calculus tells us about the slope of this landscape.

At some point on the curve:

- positive derivative → the function rises as $w$ increases,
- negative derivative → the function falls as $w$ increases,
- derivative near zero → the curve is locally flat.

At a smooth minimum, we therefore commonly encounter:

```math
\frac{dL}{dw}
=
0
```

This does **not** mean that every point with derivative zero is necessarily a minimum. It could also be a maximum or another stationary point. We will distinguish these later using curvature.

---

### Why Machine Learning Needs More Than One Derivative

Real models rarely contain a single parameter.

Even a basic linear model can contain many weights:

```math
\hat{y}
=
w_1x_1+w_2x_2+\cdots+w_nx_n+b
```

The loss therefore depends on many parameters:

```math
L
=
L(w_1,w_2,\ldots,w_n,b)
```

Now we need to ask several questions simultaneously:

```math
\frac{\partial L}{\partial w_1},
\quad
\frac{\partial L}{\partial w_2},
\quad
\ldots,
\quad
\frac{\partial L}{\partial w_n},
\quad
\frac{\partial L}{\partial b}
```

These are **partial derivatives**.

Collecting them produces one of the most important objects in machine learning: the **gradient**.

```math
\nabla L
=
\begin{bmatrix}
\frac{\partial L}{\partial w_1} \\
\frac{\partial L}{\partial w_2} \\
\vdots \\
\frac{\partial L}{\partial w_n} \\
\frac{\partial L}{\partial b}
\end{bmatrix}
```

The gradient describes how the loss changes with respect to all the model parameters.

---

### From Calculus to Learning

Once the gradient is known, the model can change its parameters in a direction intended to reduce the loss.

A typical weight update has the form:

```math
w
\leftarrow
w-\eta\frac{\partial L}{\partial w}
```

where $\eta$ is the **learning rate**.

For many parameters, this becomes:

```math
\boldsymbol{\theta}
\leftarrow
\boldsymbol{\theta}
-
\eta\nabla L
```

where $\boldsymbol{\theta}$ represents the collection of trainable parameters.

This is the basic idea behind **gradient descent**.

An important distinction is worth making immediately:

> **The gradient is not inherently negative.**

The gradient points in the direction in which a function increases most rapidly. Gradient descent deliberately moves in the **opposite direction**, which is why the update contains the minus sign.

```math
\text{gradient}
\rightarrow
\text{steepest increase}
```

```math
-\text{gradient}
\rightarrow
\text{steepest decrease}
```

This distinction becomes extremely important when studying optimization and backpropagation.

---

## ML Assocication

Calculus appears throughout machine learning because model training repeatedly involves the same general structure:

```math
\text{Parameters}
\rightarrow
\text{Prediction}
\rightarrow
\text{Loss}
\rightarrow
\text{Derivative}
\rightarrow
\text{Parameter Update}
```

For **linear regression**, derivatives tell us how the regression coefficients should change to reduce prediction error.

For **logistic regression**, derivatives tell us how the weights should change to reduce classification loss.

For **neural networks**, the same principle operates across potentially millions or billions of parameters. The **chain rule** allows derivatives to propagate backward through layers of computations — the foundation of backpropagation.

Thus, calculus is not an isolated mathematical prerequisite for machine learning.

It is the mathematics that answers the central question of learning:

> **What should the model change, in which direction, and by how much, to become less wrong?**

---

## 2. Functions, Slopes, and Rate of Change

Before introducing derivatives formally, we need to understand the idea they measure: **how one quantity changes when another quantity changes**.

Consider a function:

```math
y
=
f(x)
```

The function describes a relationship between an input $x$ and an output $y$.

If $x$ changes, $y$ may also change.

For example:

```math
f(x)
=
x^2
```

Some values of the function are:

| $x$ | $f(x)=x^2$ |
|---:|---:|
| 0 | 0 |
| 1 | 1 |
| 2 | 4 |
| 3 | 9 |
| 4 | 16 |

Clearly, the output is changing as the input changes.

But it is **not changing at a constant rate**.

From $x=1$ to $x=2$, the output increases by $3$.

From $x=2$ to $x=3$, it increases by $5$.

From $x=3$ to $x=4$, it increases by $7$.

Calculus gives us a precise way of describing this changing rate of change.

---

### Slope as Rate of Change

For a straight line:

```math
y
=
mx+b
```

the slope $m$ is:

```math
m
=
\frac{\text{change in }y}
{\text{change in }x}
```

Using the Greek letter delta $\Delta$ to represent a finite change:

```math
m
=
\frac{\Delta y}
{\Delta x}
```

For example, consider:

```math
y
=
2x+1
```

If $x$ changes from $2$ to $3$:

```math
\Delta x
=
3-2
=
1
```

The corresponding values of $y$ are:

```math
y(2)
=
5
```

and

```math
y(3)
=
7
```

therefore:

```math
\Delta y
=
7-5
=
2
```

and the slope is:

```math
m
=
\frac{2}{1}
=
2
```

This agrees with the coefficient of $x$ in the original equation.

More importantly, because this is a straight line, the slope is **2 everywhere**.

---

### Curves Do Not Have One Constant Slope

Now consider:

```math
y
=
x^2
```

Unlike a straight line, this function curves.

Its slope therefore changes depending on where we are on the curve.

Consider two points:

```math
x_1
=
2
```

```math
x_2
=
3
```

Their function values are:

```math
f(2)
=
4
```

```math
f(3)
=
9
```

The average slope between these two points is:

```math
\frac{\Delta y}{\Delta x}
=
\frac{9-4}{3-2}
=
5
```

This tells us the **average rate of change** between $x=2$ and $x=3$.

But it does not tell us the exact slope at $x=2$.

That distinction leads directly to derivatives.

---

### Secant Lines and Tangent Lines

If we choose two points on a curve and draw a straight line through them, that line is called a **secant line**.

Suppose the first point occurs at $x$ and the second at $x+h$.

The two function values are:

```math
f(x)
```

and

```math
f(x+h)
```

The horizontal change is:

```math
\Delta x
=
h
```

and the vertical change is:

```math
\Delta y
=
f(x+h)-f(x)
```

Therefore the slope of the secant line is:

```math
\frac{f(x+h)-f(x)}
{h}
```

This still measures an **average** rate of change because the two points are separated by some distance $h$.

Now imagine moving the second point progressively closer to the first.

```math
h
\rightarrow
0
```

The secant line approaches a line that touches the curve at the point of interest.

That limiting line is the **tangent line**.

Its slope represents the **instantaneous rate of change** of the function at that point.

genui{"calculus_analysis_learning_block":{"type_id":"DERIVATIVE_AS_SECANT"}}

---

### The Derivative Emerges

The previous idea can be expressed mathematically using a **limit**.

The derivative of $f(x)$ is defined as:

```math
f'(x)
=
\lim_{h\to0}
\frac{f(x+h)-f(x)}
{h}
```

The expression:

```math
\frac{f(x+h)-f(x)}
{h}
```

is called the **difference quotient**.

It measures the average rate of change over an interval of width $h$.

The limit asks what happens to this rate as the interval becomes arbitrarily small.

Thus:

```math
\text{average rate of change}
\xrightarrow{h\to0}
\text{instantaneous rate of change}
```

and that instantaneous rate of change is the **derivative**.

---

### Deriving the Derivative of $x^2$

Consider:

```math
f(x)
=
x^2
```

Using the derivative definition:

```math
f'(x)
=
\lim_{h\to0}
\frac{f(x+h)-f(x)}
{h}
```

Substitute the function:

```math
f'(x)
=
\lim_{h\to0}
\frac{(x+h)^2-x^2}
{h}
```

Expand the square:

```math
f'(x)
=
\lim_{h\to0}
\frac{x^2+2xh+h^2-x^2}
{h}
```

Cancel $x^2$:

```math
f'(x)
=
\lim_{h\to0}
\frac{2xh+h^2}
{h}
```

Factor out $h$:

```math
f'(x)
=
\lim_{h\to0}
\frac{h(2x+h)}
{h}
```

For $h\neq0$, the $h$ terms cancel:

```math
f'(x)
=
\lim_{h\to0}
(2x+h)
```

Now let $h$ approach zero:

```math
f'(x)
=
2x
```

Therefore:

```math
\boxed{
\frac{d}{dx}x^2
=
2x
}
```

This tells us something much richer than simply saying that $x^2$ is increasing.

It tells us the slope at **every possible value of $x$**.

At $x=1$:

```math
f'(1)
=
2
```

At $x=2$:

```math
f'(2)
=
4
```

At $x=3$:

```math
f'(3)
=
6
```

The curve becomes progressively steeper as $x$ increases.

---

### Derivative as a Function

An important conceptual point is that a derivative is usually **another function**.

Starting with:

```math
f(x)
=
x^2
```

we obtain:

```math
f'(x)
=
2x
```

The original function tells us the value of $y$.

The derivative tells us the slope of the original function.

So we can think of differentiation as a transformation:

```math
f(x)
\longrightarrow
f'(x)
```

or conceptually:

```math
\text{function values}
\longrightarrow
\text{rate of change of those values}
```

This distinction becomes extremely important in machine learning because we often have one function representing **loss** and another mathematical expression — its derivative — telling us how that loss changes.

---

## ML Association

Suppose a model has one parameter $w$ and its loss is:

```math
L(w)
=
w^2
```

The derivative is:

```math
\frac{dL}{dw}
=
2w
```

If:

```math
w
=
3
```

then:

```math
\frac{dL}{dw}
=
6
```

The positive derivative tells us that increasing $w$ locally increases the loss.

If:

```math
w
=
-3
```

then:

```math
\frac{dL}{dw}
=
-6
```

The negative derivative tells us that increasing $w$ locally decreases the loss.

And at:

```math
w
=
0
```

we obtain:

```math
\frac{dL}{dw}
=
0
```

which corresponds to the bottom of this particular loss curve.

This is the first mathematical glimpse of how a model can **learn**.

The model does not need to blindly try every possible parameter value. The derivative provides local information about how the loss is changing.

Gradient descent will eventually turn that information into an update rule:

```math
w
\leftarrow
w
-
\eta\frac{dL}{dw}
```

The derivative therefore acts like a local directional signal:

```math
\text{current parameter}
\rightarrow
\text{measure local slope}
\rightarrow
\text{choose direction of update}
```

This same idea scales from a single parameter $w$ to models containing millions or billions of parameters.

---

## 3. Differentiation Rules

The formal definition of a derivative is:

```math
f'(x)
=
\lim_{h\to0}
\frac{f(x+h)-f(x)}
{h}
```

This definition is fundamental because it explains what a derivative actually means.

However, deriving every derivative from this limit would be tedious.

In practice, we use a collection of **differentiation rules** that allow derivatives to be computed directly.

These rules are especially important in machine learning, where loss functions may contain powers, sums, products, exponentials, logarithms, and nested functions.

---

### Constant Rule

If a function is constant:

```math
f(x)
=
c
```

then:

```math
\frac{d}{dx}c
=
0
```

A constant does not change as $x$ changes, so its rate of change is zero.

For example:

```math
\frac{d}{dx}7
=
0
```

---

### Power Rule

For:

```math
f(x)
=
x^n
```

the derivative is:

```math
\frac{d}{dx}x^n
=
nx^{n-1}
```

For example:

```math
\frac{d}{dx}x^2
=
2x
```

```math
\frac{d}{dx}x^3
=
3x^2
```

```math
\frac{d}{dx}x^5
=
5x^4
```

The rule also works for negative and fractional powers where the function is defined.

For example:

```math
\frac{d}{dx}x^{-1}
=
-x^{-2}
```

and:

```math
\frac{d}{dx}x^{1/2}
=
\frac{1}{2}x^{-1/2}
```

---

### Constant Multiple Rule

If a function is multiplied by a constant:

```math
f(x)
=
c\,g(x)
```

then:

```math
\frac{d}{dx}
\left[
c\,g(x)
\right]
=
c\,g'(x)
```

For example:

```math
f(x)
=
5x^3
```

then:

```math
f'(x)
=
15x^2
```

because:

```math
5
\cdot
3x^2
=
15x^2
```

---

### Sum and Difference Rules

If:

```math
f(x)
=
g(x)+h(x)
```

then:

```math
f'(x)
=
g'(x)+h'(x)
```

Similarly:

```math
\frac{d}{dx}
\left[
g(x)-h(x)
\right]
=
g'(x)-h'(x)
```

This means each term can be differentiated independently.

For example:

```math
f(x)
=
3x^3+2x^2-5x+7
```

Differentiate term by term:

```math
f'(x)
=
9x^2+4x-5
```

The constant $7$ disappears because its derivative is zero.

---

### Product Rule

When two functions are multiplied:

```math
f(x)
=
u(x)v(x)
```

the derivative is **not** simply the product of their derivatives.

Instead:

```math
\frac{d}{dx}
\left[
u(x)v(x)
\right]
=
u'(x)v(x)
+
u(x)v'(x)
```

or more compactly:

```math
(uv)'
=
u'v+uv'
```

Why?

Because when $x$ changes, **both** $u$ and $v$ may change.

The total change in their product must therefore account for both effects.

For example:

```math
f(x)
=
x^2(x+3)
```

Let:

```math
u
=
x^2
```

and:

```math
v
=
x+3
```

Then:

```math
u'
=
2x
```

and:

```math
v'
=
1
```

Using the product rule:

```math
f'(x)
=
2x(x+3)
+
x^2(1)
```

which simplifies to:

```math
f'(x)
=
3x^2+6x
```

---

### Quotient Rule

If one function is divided by another:

```math
f(x)
=
\frac{u(x)}
{v(x)}
```

then:

```math
f'(x)
=
\frac{
u'(x)v(x)-u(x)v'(x)
}
{
[v(x)]^2
}
```

provided:

```math
v(x)
\neq
0
```

A common mnemonic is:

```text
bottom × derivative of top
−
top × derivative of bottom
--------------------------------
bottom squared
```

For example:

```math
f(x)
=
\frac{x^2}
{x+1}
```

Let:

```math
u
=
x^2
```

and:

```math
v
=
x+1
```

Then:

```math
u'
=
2x
```

and:

```math
v'
=
1
```

Therefore:

```math
f'(x)
=
\frac{
2x(x+1)-x^2
}
{
(x+1)^2
}
```

which simplifies to:

```math
f'(x)
=
\frac{
x^2+2x
}
{
(x+1)^2
}
```

---

### Exponential Functions

One of the most important functions in machine learning is:

```math
f(x)
=
e^x
```

Its derivative has a remarkable property:

```math
\frac{d}{dx}e^x
=
e^x
```

The function is its own derivative.

More generally:

```math
\frac{d}{dx}e^{kx}
=
ke^{kx}
```

Exponential functions appear in many ML contexts, including:

- logistic regression,
- sigmoid functions,
- softmax,
- probability distributions,
- likelihood functions.

---

### Logarithmic Functions

For the natural logarithm:

```math
f(x)
=
\ln x
```

the derivative is:

```math
\frac{d}{dx}\ln x
=
\frac{1}{x}
```

for:

```math
x
>
0
```

Logarithms are especially important in machine learning because many loss functions involve logarithms.

Examples include:

- binary cross-entropy,
- categorical cross-entropy,
- log-likelihood,
- negative log-likelihood.

---

### Why These Rules Matter

Consider a function such as:

```math
f(x)
=
3x^2+4x+7
```

Without differentiation rules, we could derive the derivative using the limit definition every time.

Instead, we can immediately write:

```math
f'(x)
=
6x+4
```

The rules do not replace the meaning of the derivative.

They simply provide efficient ways to calculate it.

A useful distinction is:

```math
\text{limit definition}
\rightarrow
\text{explains what a derivative is}
```

while:

```math
\text{differentiation rules}
\rightarrow
\text{make derivatives practical to compute}
```

---

## ML Association

Machine-learning loss functions are usually built from combinations of simpler mathematical functions.

Consider a simple squared loss:

```math
L(w)
=
(y-wx)^2
```

This expression contains:

- subtraction,
- multiplication,
- a power,
- and a function nested inside another function.

To differentiate such expressions efficiently, we combine differentiation rules.

As models become more complex, this becomes even more important.

For example, logistic regression uses the sigmoid function:

```math
\sigma(z)
=
\frac{1}
{1+e^{-z}}
```

This combines:

- an exponential,
- addition,
- division,
- and nested functions.

Neural networks go even further. Their computations may contain thousands or millions of nested operations.

Differentiation rules allow the derivative of a complicated model to be constructed systematically from the derivatives of its individual parts.

The most important rule for doing this is the **chain rule**, which we will examine next.

---

## 4. The Chain Rule

Many functions in machine learning are not simple standalone expressions. They are **functions built inside other functions**.

For example:

```math
f(x)
=
(3x+2)^2
```

The outer operation is squaring, while the inner function is:

```math
u
=
3x+2
```

so the original function can be written as:

```math
f(x)
=
u^2
```

The **chain rule** tells us how to differentiate such nested functions.

> In modern machine learning, this small rule carries an enormous workload.

---

### The Core Idea

Suppose:

```math
y
=
f(u)
```

and:

```math
u
=
g(x)
```

Then $y$ depends on $x$ indirectly through $u$:

```math
x
\rightarrow
u
\rightarrow
y
```

The chain rule states:

```math
\frac{dy}{dx}
=
\frac{dy}{du}
\frac{du}{dx}
```

Conceptually:

```math
\text{overall rate of change}
=
\text{outer rate of change}
\times
\text{inner rate of change}
```

In words:

> Differentiate the outer function, keep the inner function intact, then multiply by the derivative of the inner function.

---

### A Simple Example

Consider:

```math
f(x)
=
(3x+2)^2
```

Let:

```math
u
=
3x+2
```

Then:

```math
f
=
u^2
```

Differentiate the outer function with respect to $u$:

```math
\frac{df}{du}
=
2u
```

Differentiate the inner function with respect to $x$:

```math
\frac{du}{dx}
=
3
```

Using the chain rule:

```math
\frac{df}{dx}
=
\frac{df}{du}
\frac{du}{dx}
```

Therefore:

```math
\frac{df}{dx}
=
2u\cdot3
```

Substitute back:

```math
\frac{df}{dx}
=
6(3x+2)
```

which can also be written as:

```math
\frac{df}{dx}
=
18x+12
```

The derivative of the outer function alone would have produced only $2(3x+2)$. The additional factor of $3$ appears because the inner function itself changes three times as fast as $x$.

---

### Why Multiplication Appears

The multiplication in the chain rule is not arbitrary.

Suppose a small change in $x$ causes a change in $u$:

```math
\Delta x
\rightarrow
\Delta u
```

and that change in $u$ causes a change in $y$:

```math
\Delta u
\rightarrow
\Delta y
```

Approximately:

```math
\Delta u
\approx
\frac{du}{dx}\Delta x
```

and:

```math
\Delta y
\approx
\frac{dy}{du}\Delta u
```

Substituting the first into the second:

```math
\Delta y
\approx
\frac{dy}{du}
\frac{du}{dx}
\Delta x
```

Therefore:

```math
\frac{dy}{dx}
=
\frac{dy}{du}
\frac{du}{dx}
```

The chain rule therefore tracks how a change propagates through a sequence of dependent quantities.

> This idea of **propagating influence through a chain of computations** is exactly what makes the rule so important in neural networks.

---

### Longer Chains

The idea extends naturally to several nested functions.

Suppose:

```math
x
\rightarrow
a
\rightarrow
b
\rightarrow
c
\rightarrow
y
```

where each quantity depends on the previous one.

Then:

```math
\frac{dy}{dx}
=
\frac{dy}{dc}
\frac{dc}{db}
\frac{db}{da}
\frac{da}{dx}
```

Each derivative describes one local relationship.

Multiplying them gives the effect of $x$ on the final output $y$.

This is a powerful general principle:

> **Complex global behaviour can be differentiated by combining simple local derivatives.**

That principle scales remarkably well.

---

### Chain Rule with Common Functions

Consider an exponential of another function:

```math
f(x)
=
e^{g(x)}
```

The outer derivative is:

```math
\frac{d}{du}e^u
=
e^u
```

Therefore:

```math
\frac{d}{dx}e^{g(x)}
=
e^{g(x)}g'(x)
```

Similarly, for a logarithm:

```math
f(x)
=
\ln(g(x))
```

we obtain:

```math
\frac{d}{dx}\ln(g(x))
=
\frac{g'(x)}{g(x)}
```

For a power:

```math
f(x)
=
[g(x)]^n
```

we obtain:

```math
\frac{d}{dx}[g(x)]^n
=
n[g(x)]^{n-1}g'(x)
```

These patterns occur constantly in machine-learning loss functions and activation functions.

---

### A Loss-Function Example

Consider the squared loss for a simple model:

```math
L(w)
=
(y-wx)^2
```

This is a nested function.

Define the inner function:

```math
e
=
y-wx
```

where $e$ represents the prediction error.

Then:

```math
L
=
e^2
```

Differentiate the outer function:

```math
\frac{dL}{de}
=
2e
```

Differentiate the inner function with respect to $w$:

```math
\frac{de}{dw}
=
-x
```

Using the chain rule:

```math
\frac{dL}{dw}
=
\frac{dL}{de}
\frac{de}{dw}
```

Therefore:

```math
\frac{dL}{dw}
=
2e(-x)
```

Substituting the error back:

```math
\frac{dL}{dw}
=
-2x(y-wx)
```

or equivalently:

```math
\frac{dL}{dw}
=
2x(wx-y)
```

This derivative tells us how the loss changes when the model weight $w$ changes.

The chain rule has therefore converted a nested prediction-error calculation into exactly the information an optimizer needs.

---

### Computational Graph View

Machine-learning systems often represent computations conceptually as a graph.

For the previous example:

```math
w
\rightarrow
wx
\rightarrow
y-wx
\rightarrow
(y-wx)^2
\rightarrow
L
```

The forward direction computes the prediction and loss.

The reverse direction computes how the loss depends on each earlier quantity.

```math
L
\rightarrow
\frac{dL}{de}
\rightarrow
\frac{de}{dw}
\rightarrow
\frac{dL}{dw}
```

This backward traversal repeatedly applies the chain rule.

> Forward propagation computes values. Backpropagation computes responsibility.

Here, *responsibility* is intuitive rather than a formal mathematical term: the derivatives quantify how strongly an earlier quantity influences the final loss.

---

## ML Association

The chain rule is one of the mathematical foundations of **backpropagation**.

A neural network may contain many successive transformations:

```math
\mathbf{x}
\rightarrow
\mathbf{z}_1
\rightarrow
\mathbf{a}_1
\rightarrow
\mathbf{z}_2
\rightarrow
\mathbf{a}_2
\rightarrow
\cdots
\rightarrow
L
```

Each layer depends on the output of the layer before it.

To determine how an early weight affects the final loss, the model repeatedly applies the chain rule through this sequence of dependencies.

Conceptually:

```math
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial a_n}
\frac{\partial a_n}{\partial z_n}
\cdots
\frac{\partial z_1}{\partial w}
```

This is the mathematical backbone of backpropagation.

The same principle appears beyond neural networks whenever an objective function is composed of nested transformations.

A useful way to remember the role of the chain rule in ML is:

> **If the model is a chain of computations, the chain rule is how learning travels backward through it.**

Modern deep-learning frameworks can perform these derivative calculations automatically, but the mathematics underneath remains the same.

> Autodiff automates the bookkeeping; the chain rule supplies the logic.

Once the chain rule is understood, backpropagation stops looking like a mysterious neural-network trick. It becomes what it really is: a systematic application of calculus across a computational graph.

---

## 5. Partial Derivatives

So far, we have mostly differentiated functions containing a single variable.

For example:

```math
f(x)
=
x^2
```

has the derivative:

```math
\frac{df}{dx}
=
2x
```

But machine-learning models usually depend on **many variables at once**.

Consider:

```math
f(x,y)
=
x^2+3xy+y^2
```

The output depends on both $x$ and $y$.

Now we can ask two different questions:

> How does $f$ change when $x$ changes?

and:

> How does $f$ change when $y$ changes?

These are answered using **partial derivatives**.

---

### The Core Idea

A partial derivative measures how a multivariable function changes with respect to **one variable while treating the others as constant**.

The partial derivative of $f$ with respect to $x$ is written:

```math
\frac{\partial f}{\partial x}
```

The symbol $\partial$ is used instead of $d$ to indicate that the function depends on multiple variables.

For:

```math
f(x,y)
=
x^2+3xy+y^2
```

differentiate with respect to $x$ while treating $y$ as constant.

The derivative of:

```math
x^2
```

is:

```math
2x
```

The derivative of:

```math
3xy
```

with respect to $x$ is:

```math
3y
```

because $y$ behaves like a constant.

The derivative of:

```math
y^2
```

with respect to $x$ is zero.

Therefore:

```math
\frac{\partial f}{\partial x}
=
2x+3y
```

---

### Differentiating with Respect to Another Variable

Now differentiate the same function with respect to $y$:

```math
f(x,y)
=
x^2+3xy+y^2
```

Treat $x$ as constant.

The derivative of:

```math
x^2
```

with respect to $y$ is zero.

The derivative of:

```math
3xy
```

with respect to $y$ is:

```math
3x
```

and:

```math
\frac{\partial}{\partial y}y^2
=
2y
```

Therefore:

```math
\frac{\partial f}{\partial y}
=
3x+2y
```

So the same function has different rates of change depending on which direction we examine.

---

### Geometric Interpretation

For a function of one variable:

```math
y
=
f(x)
```

we can imagine a curve.

For a function of two variables:

```math
z
=
f(x,y)
```

we can imagine a surface.

The value of $z$ changes as we move across the surface in different directions.

The partial derivative:

```math
\frac{\partial f}{\partial x}
```

measures the slope when moving in the $x$ direction while keeping $y$ fixed.

Similarly:

```math
\frac{\partial f}{\partial y}
```

measures the slope when moving in the $y$ direction while keeping $x$ fixed.

The same surface may be rising sharply in one direction and falling in another.

This is why a single derivative is no longer enough.

---

### Why Holding Other Variables Constant Works

Suppose:

```math
f(x,y)
=
x^2+xy
```

and we want:

```math
\frac{\partial f}{\partial x}
```

Because $y$ is held constant, the term:

```math
xy
```

behaves just like a constant multiple of $x$.

So:

```math
\frac{\partial}{\partial x}(xy)
=
y
```

Similarly:

```math
\frac{\partial}{\partial y}(xy)
=
x
```

The same term contributes differently depending on which variable is being differentiated.

This idea becomes extremely important when model parameters interact with one another.

---

### Partial Derivatives in a Linear Model

Consider a model with two inputs:

```math
\hat{y}
=
w_1x_1+w_2x_2+b
```

Suppose the squared loss is:

```math
L
=
(y-\hat{y})^2
```

Substituting the prediction:

```math
L
=
\left(
y-(w_1x_1+w_2x_2+b)
\right)^2
```

The loss depends on three trainable parameters:

```math
L
=
L(w_1,w_2,b)
```

To understand how each parameter affects the loss, we compute:

```math
\frac{\partial L}{\partial w_1}
```

```math
\frac{\partial L}{\partial w_2}
```

and:

```math
\frac{\partial L}{\partial b}
```

Each derivative answers a separate question.

For example:

```math
\frac{\partial L}{\partial w_1}
```

asks:

> If $w_1$ changes slightly while $w_2$ and $b$ are held fixed, how does the loss change?

---

### Deriving One Weight Gradient

Let:

```math
e
=
y-\hat{y}
```

so:

```math
L
=
e^2
```

For the first weight:

```math
\frac{\partial L}{\partial w_1}
=
\frac{\partial L}{\partial e}
\frac{\partial e}{\partial w_1}
```

Using the chain rule:

```math
\frac{\partial L}{\partial e}
=
2e
```

and:

```math
\frac{\partial e}{\partial w_1}
=
-x_1
```

Therefore:

```math
\frac{\partial L}{\partial w_1}
=
-2x_1e
```

Substituting the error:

```math
\frac{\partial L}{\partial w_1}
=
-2x_1
\left(
y-\hat{y}
\right)
```

Similarly:

```math
\frac{\partial L}{\partial w_2}
=
-2x_2
\left(
y-\hat{y}
\right)
```

and:

```math
\frac{\partial L}{\partial b}
=
-2
\left(
y-\hat{y}
\right)
```

The same loss function therefore produces a separate derivative for every trainable parameter.

---

### One Loss, Many Directions

This is an important shift in perspective.

With one parameter:

```math
L
=
L(w)
```

there is only one direction in which $w$ can move.

With many parameters:

```math
L
=
L(w_1,w_2,\ldots,w_n)
```

the loss exists over a multidimensional parameter space.

Each partial derivative describes the slope along one coordinate direction.

Conceptually:

```math
\frac{\partial L}{\partial w_1}
\rightarrow
\text{sensitivity to }w_1
```

```math
\frac{\partial L}{\partial w_2}
\rightarrow
\text{sensitivity to }w_2
```

```math
\vdots
```

```math
\frac{\partial L}{\partial w_n}
\rightarrow
\text{sensitivity to }w_n
```

But an optimizer needs to consider all of these directions together.

That leads directly to the **gradient**.

---

## ML Association

Machine-learning models typically contain multiple trainable parameters.

A linear regression model may contain dozens or thousands of coefficients.

A neural network may contain millions or billions of weights and biases.

The loss therefore depends on all of them:

```math
L
=
L(\theta_1,\theta_2,\ldots,\theta_n)
```

For every parameter, training needs to know:

```math
\frac{\partial L}{\partial \theta_i}
```

These partial derivatives quantify how sensitive the loss is to each parameter individually.

This gives us a useful mental model:

> **A partial derivative asks how much one parameter is responsible for changing the loss while the others are temporarily frozen.**

The optimizer does not update one parameter in isolation, however. It collects all these partial derivatives together and uses them simultaneously.

That collection is the **gradient**:

```math
\nabla L
=
\begin{bmatrix}
\frac{\partial L}{\partial \theta_1} \\
\frac{\partial L}{\partial \theta_2} \\
\vdots \\
\frac{\partial L}{\partial \theta_n}
\end{bmatrix}
```

So partial derivatives give us the individual pieces.

The gradient assembles them into a single object describing the local behaviour of the entire loss function.

> **Partial derivatives inspect one control knob at a time; the gradient reads the whole control panel.**

---

## 6. Gradients and Directional Change

Partial derivatives tell us how a multivariable function changes when we vary one input at a time.

But machine-learning models contain many parameters that can change together.

To reason about all of them simultaneously, we collect the partial derivatives into a vector called the **gradient**.

For a function:

```math
f(x_1,x_2,\ldots,x_n)
```

the gradient is:

```math
\nabla f
=
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
```

The symbol:

```math
\nabla
```

is called **nabla** or **del**.

The gradient therefore combines all first-order partial derivatives into a single vector.

---

### A Two-Variable Example

Consider:

```math
f(x,y)
=
x^2+y^2
```

The partial derivatives are:

```math
\frac{\partial f}{\partial x}
=
2x
```

and:

```math
\frac{\partial f}{\partial y}
=
2y
```

Therefore:

```math
\nabla f
=
\begin{bmatrix}
2x \\
2y
\end{bmatrix}
```

At the point:

```math
(x,y)
=
(1,2)
```

the gradient becomes:

```math
\nabla f(1,2)
=
\begin{bmatrix}
2 \\
4
\end{bmatrix}
```

This vector contains two pieces of information:

- how strongly the function changes in the $x$ direction,
- how strongly the function changes in the $y$ direction.

But the gradient tells us more than that.

It also has a geometric interpretation.

---

### The Gradient Points Uphill

For a multivariable function, the gradient points in the direction of **steepest local increase**.

For:

```math
f(x,y)
=
x^2+y^2
```

the surface resembles a bowl.

The minimum occurs at:

```math
(x,y)
=
(0,0)
```

At:

```math
(1,2)
```

the gradient is:

```math
\begin{bmatrix}
2 \\
4
\end{bmatrix}
```

This points away from the minimum and toward increasing values of the function.

That gives us one of the most important facts in optimization:

> **The gradient points uphill.**

Therefore:

```math
-\nabla f
```

points in the direction of steepest local decrease.

This is precisely why gradient descent uses the **negative gradient**.

---

### Gradient Magnitude

Because the gradient is a vector, it has both direction and magnitude.

For:

```math
\nabla f
=
\begin{bmatrix}
2 \\
4
\end{bmatrix}
```

its magnitude is:

```math
\left\|
\nabla f
\right\|
=
\sqrt{2^2+4^2}
```

so:

```math
\left\|
\nabla f
\right\|
=
\sqrt{20}
```

A large gradient magnitude indicates that the function is changing rapidly nearby.

A small gradient magnitude indicates that the surface is relatively flat.

Conceptually:

```math
\left\|\nabla f\right\|
\text{ large}
\rightarrow
\text{steep region}
```

```math
\left\|\nabla f\right\|
\text{ small}
\rightarrow
\text{flat region}
```

This becomes important during optimization because the gradient carries information about both **where to move** and **how strongly the function is changing**.

---

### Directional Change

The gradient gives the direction of steepest increase, but we may want to know how the function changes in some other direction.

Suppose:

```math
\mathbf{u}
```

is a unit vector representing a direction.

The rate of change of $f$ in that direction is given by the **directional derivative**:

```math
D_{\mathbf{u}}f
=
\nabla f
\cdot
\mathbf{u}
```

This is a dot product.

That means the directional derivative depends on the alignment between:

```math
\nabla f
```

and:

```math
\mathbf{u}
```

Using the dot-product identity:

```math
\nabla f
\cdot
\mathbf{u}
=
\left\|
\nabla f
\right\|
\left\|
\mathbf{u}
\right\|
\cos\theta
```

and because $\mathbf{u}$ is a unit vector:

```math
\left\|
\mathbf{u}
\right\|
=
1
```

we obtain:

```math
D_{\mathbf{u}}f
=
\left\|
\nabla f
\right\|
\cos\theta
```

where $\theta$ is the angle between the gradient and the chosen direction.

---

### Why the Gradient Is the Steepest Direction

The directional derivative is largest when:

```math
\cos\theta
=
1
```

which happens when:

```math
\theta
=
0
```

In other words, the greatest increase occurs when the direction $\mathbf{u}$ points exactly along the gradient.

Therefore:

```math
\mathbf{u}
\parallel
\nabla f
```

gives the steepest increase.

Similarly, the greatest decrease occurs when:

```math
\cos\theta
=
-1
```

which happens when:

```math
\theta
=
\pi
```

so the direction is opposite to the gradient:

```math
\mathbf{u}
\parallel
-\nabla f
```

This gives the mathematical justification for moving along the negative gradient during optimization.

---

### A Loss Surface Example

Suppose the loss depends on two model parameters:

```math
L(w_1,w_2)
=
w_1^2+w_2^2
```

The gradient is:

```math
\nabla L
=
\begin{bmatrix}
2w_1 \\
2w_2
\end{bmatrix}
```

At:

```math
(w_1,w_2)
=
(3,4)
```

we obtain:

```math
\nabla L
=
\begin{bmatrix}
6 \\
8
\end{bmatrix}
```

The negative gradient is:

```math
-\nabla L
=
\begin{bmatrix}
-6 \\
-8
\end{bmatrix}
```

The gradient points away from the minimum.

The negative gradient points toward it.

This is why the update:

```math
\boldsymbol{\theta}
\leftarrow
\boldsymbol{\theta}
-
\eta\nabla L
```

moves the model parameters downhill on the loss surface.

---

### Gradient as Local Information

The gradient is fundamentally **local**.

It tells us how the function behaves near the current point.

It does not directly tell us what the entire loss surface looks like.

For example, if:

```math
\nabla L
=
\mathbf{0}
```

then the loss is locally flat.

But this does not automatically mean we have found the global minimum.

The point could be:

- a local minimum,
- a local maximum,
- a saddle point,
- or part of a flat region.

The gradient tells us the local slope.

To understand curvature, we need second derivatives, which we will examine later.

---

### Gradient of a Model Loss

Suppose a model contains parameters:

```math
\boldsymbol{\theta}
=
\begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_n \\
b
\end{bmatrix}
```

and the loss is:

```math
L
=
L(\boldsymbol{\theta})
```

The gradient is:

```math
\nabla_{\boldsymbol{\theta}}L
=
\begin{bmatrix}
\frac{\partial L}{\partial w_1} \\
\frac{\partial L}{\partial w_2} \\
\vdots \\
\frac{\partial L}{\partial w_n} \\
\frac{\partial L}{\partial b}
\end{bmatrix}
```

Each element tells us how the loss changes with respect to one parameter.

Taken together, the vector describes the local slope of the loss across the entire parameter space.

This is the point where calculus and linear algebra meet directly:

> **Calculus computes the partial derivatives. Linear algebra packages them into a direction.**

---

## ML Association

In machine learning, the gradient is one of the central objects used during training.

A model may have parameter vector:

```math
\boldsymbol{\theta}
=
(\theta_1,\theta_2,\ldots,\theta_n)
```

and loss:

```math
L(\boldsymbol{\theta})
```

Training repeatedly performs the following conceptual sequence:

```math
\text{compute predictions}
\rightarrow
\text{compute loss}
\rightarrow
\text{compute gradient}
\rightarrow
\text{update parameters}
```

The gradient tells the optimizer how the loss responds to changes in every parameter.

The update direction is usually based on:

```math
-\nabla L
```

because:

```math
\nabla L
```

points toward steepest increase, while:

```math
-\nabla L
```

points toward steepest decrease.

This gives us a compact interpretation:

> **The gradient is the model's local map of the loss landscape.**

And the negative gradient gives the optimizer its immediate downhill direction.

In neural networks, backpropagation computes the partial derivatives required to form these gradients.

Gradient-based optimizers then use them to modify the weights.

So the relationship is:

```math
\text{chain rule}
\rightarrow
\text{backpropagation}
\rightarrow
\text{gradient}
\rightarrow
\text{parameter update}
```

This sequence forms a large part of the mathematical machinery behind modern deep learning.

> **Backpropagation tells us the gradient. Optimization decides what to do with it.**

---

## 7. Gradient Descent and Optimization

We now have the mathematical pieces needed to understand one of the most important algorithms in machine learning: **gradient descent**.

Suppose a model has parameters:

```math 
\boldsymbol{\theta}
=
(\theta_1,\theta_2,\ldots,\theta_n)
```

and a loss function:

```math 
L(\boldsymbol{\theta})
```

Training seeks parameter values that make the loss as small as possible:

```math 
\boldsymbol{\theta}^{*}
=
\arg\min_{\boldsymbol{\theta}}
L(\boldsymbol{\theta})
```

The symbol $\arg\min$ means:

> Find the **argument**, or parameter values, at which the function reaches its minimum.

Gradient descent approaches this problem iteratively.

Instead of trying every possible parameter combination, it repeatedly asks:

> **Which direction locally decreases the loss?**

---

### The Gradient Gives the Direction

From the previous section:

```math 
\nabla L
```

points in the direction of steepest local **increase** in loss.

Therefore:

```math
-\nabla L
```

points in the direction of steepest local **decrease**.

This immediately gives us the gradient descent update:

```math
\boldsymbol{\theta}
\leftarrow
\boldsymbol{\theta}
-
\eta\nabla L
```

where:

- $\boldsymbol{\theta}$ represents the model parameters,
- $\nabla L$ is the gradient of the loss,
- $\eta$ is the **learning rate**.

This single equation captures the central mechanism of gradient descent.

---

### Why the Minus Sign Matters

The minus sign does not appear because gradients are inherently negative.

The gradient itself may contain:

- positive components,
- negative components,
- zero components.

Consider a single parameter:

```math 
w
```

with update:

```math 
w
\leftarrow
w
-
\eta
\frac{dL}{dw}
```

Suppose:

```math
\frac{dL}{dw}
>
0
```

The loss increases as $w$ increases locally.

Therefore the update subtracts a positive quantity:

```math 
w
\leftarrow
w-\text{positive amount}
```

so $w$ decreases.

Now suppose:

```math 
\frac{dL}{dw}
<
0
```

Then:

```math 
w
\leftarrow
w-\text{negative amount}
```

which is equivalent to:

```math 
w
\leftarrow
w+\text{positive amount}
```

so $w$ increases.

In both cases, the update moves opposite to the local uphill direction.

Therefore:

> **The sign of the gradient tells us which way is uphill; the minus sign deliberately sends us the other way.**

---

### A Complete Numerical Example

Consider:

```math
L(w)
=
w^2
```

Its derivative is:

```math
\frac{dL}{dw}
=
2w
```

Suppose we begin with:

```math
w_0
=
4
```

and choose learning rate:

```math
\eta
=
0.1
```

At the initial point:

```math
\frac{dL}{dw}
=
2(4)
=
8
```

Apply the update:

```math 
w_1
=
4-(0.1)(8)
```

Therefore:

```math 
w_1
=
3.2
```

The loss has changed from:

```math
L(4)
=
16
```

to:

```math 
L(3.2)
=
10.24
```

The loss decreased.

---

### Another Iteration

At:

```math 
w_1
=
3.2
```

the gradient is:

```math 
\frac{dL}{dw}
=
6.4
```

Update again:

```math 
w_2
=
3.2-(0.1)(6.4)
```

so:

```math 
w_2
=
2.56
```

The loss becomes:

```math 
L(2.56)
=
6.5536
```

Repeating this process gives:

```text 
w = 4.000
w = 3.200
w = 2.560
w = 2.048
w = 1.638
...
```

The parameter progressively approaches:

```math 
w
=
0
```

which is the minimum of:

```math
L(w)
=
w^2
```

Gradient descent therefore reaches the solution through a sequence of increasingly better parameter values.

---

### Gradient Descent Is Iterative

Unlike solving an equation directly, gradient descent generally does not jump immediately to the optimum.

Instead:

```math
\boldsymbol{\theta}_0
\rightarrow
\boldsymbol{\theta}_1
\rightarrow
\boldsymbol{\theta}_2
\rightarrow
\cdots
\rightarrow
\boldsymbol{\theta}^{*}
```

Each step uses the gradient calculated at the **current** parameter values.

Conceptually:

```text 
1. Start with parameters
2. Make predictions
3. Calculate loss
4. Calculate gradient
5. Update parameters
6. Repeat
```

This loop is the heart of training for many machine-learning models.

---

### The Learning Rate

The learning rate $\eta$ controls how far the parameters move during each update.

```math
\boldsymbol{\theta}
\leftarrow
\boldsymbol{\theta}
-
\eta\nabla L
```

The gradient determines the direction and contributes information about the local slope.

The learning rate scales the size of the update.

A useful interpretation is:

```math 
\text{parameter update}
=
\text{direction and slope information}
\times
\text{step-size control}
```

---

### Learning Rate Too Small

If $\eta$ is very small:

```math id="l6mh71"
\eta
\ll
1
```

the optimizer takes tiny steps.

This may produce stable progress, but training can become unnecessarily slow.

Conceptually:

```text 
start  •
        ↓
       •
       ↓
      •
      ↓
     •
     ↓
minimum
```

The optimizer eventually reaches the valley but requires many updates.

---

### Learning Rate Too Large

If the learning rate is too large, an update may jump across the minimum.

Instead of approaching the bottom smoothly:

```text 
      \       /
       \     /
        \   /
         \_/
```

the optimizer may repeatedly overshoot:

```text 
      •\       /•
        \     /
         \   /
          \_/
```

With sufficiently large steps, the optimization process may even move farther away from the minimum and **diverge**.

Therefore the learning rate represents an important balance:

> **Too small wastes steps; too large can destroy convergence.**

---

### Why the Gradient Changes During Training

The gradient must be recalculated after every update because it describes the slope only at the current point.

Suppose:

```math 
\nabla L(\boldsymbol{\theta}_0)
```

is calculated at the initial parameters.

After an update:

```math
\boldsymbol{\theta}_1
=
\boldsymbol{\theta}_0
-
\eta
\nabla L(\boldsymbol{\theta}_0)
```

the model is now at a different location on the loss surface.

Therefore we need a new gradient:

```math
\nabla L(\boldsymbol{\theta}_1)
```

and then:

```math
\boldsymbol{\theta}_2
=
\boldsymbol{\theta}_1
-
\eta
\nabla L(\boldsymbol{\theta}_1)
```

The process continues.

> **Gradient descent does not follow a map drawn in advance. It repeatedly redraws its local map after every step.**

---

### Convergence

Gradient descent is said to **converge** when successive updates approach a stable solution.

Near a smooth minimum:

```math
\nabla L
\approx
\mathbf{0}
```

so the parameter updates become small:

```math
\eta\nabla L
\approx
\mathbf{0}
```

and therefore:

```math 
\boldsymbol{\theta}_{t+1}
\approx
\boldsymbol{\theta}_t
```

This indicates that the optimizer has reached, or is close to, a stationary point.

However:

```math
\nabla L
=
\mathbf{0}
```

does not by itself prove that the point is the global minimum.

The loss surface may contain local minima, maxima, saddle points, and flat regions.

---

### Batch, Stochastic, and Mini-Batch Gradient Descent

The gradient can be calculated using different amounts of training data.

**Batch gradient descent** computes the gradient using the entire training dataset before performing an update.

Conceptually:

```math 
\text{all training examples}
\rightarrow
\nabla L
\rightarrow
\text{one update}
```

This produces a stable estimate of the gradient but can be computationally expensive for large datasets.

**Stochastic gradient descent (SGD)** performs an update using one training example at a time:

```math 
\text{one example}
\rightarrow
\nabla L
\rightarrow
\text{update}
```

This produces frequent but noisy updates.

**Mini-batch gradient descent** uses a small subset of training examples:

```math 
\text{mini-batch}
\rightarrow
\nabla L
\rightarrow
\text{update}
```

It provides a practical compromise between computational efficiency and gradient stability.

Mini-batch training is widely used in modern deep learning.

---

### Gradient Descent and Epochs

Suppose a dataset contains:

```math 
N
```

training examples and the mini-batch size is:

```math
B
```

Approximately:

```math
\frac{N}{B}
```

parameter updates occur during one complete pass through the dataset.

One complete pass through all training examples is called an **epoch**.

Therefore:

```math 
1\text{ epoch}
\approx
\frac{N}{B}
\text{ gradient updates}
```

when mini-batch gradient descent is used.

This distinction is important:

> **An epoch is a pass through the data; an iteration is a parameter-update step.**

They are related, but they are not the same thing.

---

### Gradient Descent Is an Optimization Algorithm

Gradient descent itself is not a predictive model.

It does not define:

```math 
\hat{y}
```

It does not define the loss function.

Instead, it is an **optimization algorithm** used to find parameters that minimize an objective.

For example:

```math
\text{Linear Regression}
+
\text{MSE}
+
\text{Gradient Descent}
```

means:

- linear regression defines the prediction model,
- MSE defines what counts as error,
- gradient descent adjusts the model parameters to reduce that error.

The same optimization principle can be applied to many different differentiable models.

---

## ML Association

Gradient descent provides the mechanism through which many machine-learning models **learn from error**.

The complete training loop can now be expressed as:

```math 
\mathbf{x}
\rightarrow
\hat{y}
\rightarrow
L
\rightarrow
\nabla L
\rightarrow
\boldsymbol{\theta}_{\text{new}}
```

For neural networks, we can expand the middle of this process:

```math
\text{forward propagation}
\rightarrow
\text{loss}
\rightarrow
\text{backpropagation}
\rightarrow
\text{gradients}
\rightarrow
\text{optimizer}
```

These components perform different jobs:

- **Forward propagation** computes predictions.
- **The loss function** measures how wrong those predictions are.
- **Backpropagation** uses the chain rule to compute gradients.
- **The optimizer** uses those gradients to update the parameters.

Gradient descent supplies the fundamental update idea:

```math 
\boldsymbol{\theta}
\leftarrow
\boldsymbol{\theta}
-
\eta\nabla L
```

More advanced optimizers such as momentum-based methods, RMSProp, and Adam modify how gradient information is used, but they retain the same central objective: use derivative information to improve the parameters.

The conceptual chain is therefore:

```math 
\text{error}
\rightarrow
\text{calculus}
\rightarrow
\text{gradient}
\rightarrow
\text{optimization}
\rightarrow
\text{learning}
```

> **A model predicts. A loss function judges. Calculus assigns direction. The optimizer acts.**

That is the mathematical core of gradient-based learning.

---

## 8. Second Derivatives and Curvature

The first derivative tells us how a function changes.

The **second derivative** tells us how the first derivative itself changes.

If:

```math
f'(x)
=
\frac{df}{dx}
```

then the second derivative is:

```math
f''(x)
=
\frac{d^2f}{dx^2}
```

This gives us information about the **curvature** of a function.

For machine learning, curvature helps us understand the shape of loss functions and what kind of stationary point an optimizer may have encountered.

---

### From Slope to Curvature

Consider:

```math
f(x)
=
x^2
```

Its first derivative is:

```math
f'(x)
=
2x
```

The slope changes as $x$ changes.

Differentiate again:

```math
f''(x)
=
2
```

The positive second derivative tells us that the curve bends upward.

Conceptually:

```text
\       /
 \     /
  \   /
   \_/
```

This shape is often described as **convex** or bowl-shaped.

Now consider:

```math
f(x)
=
-x^2
```

Its first derivative is:

```math
f'(x)
=
-2x
```

and its second derivative is:

```math
f''(x)
=
-2
```

The curve bends downward:

```text
   /‾\
  /   \
 /     \
/       \
```

This is a concave shape.

---

### Stationary Points

A **stationary point** occurs where:

```math
f'(x)
=
0
```

At such a point, the function is locally flat.

But a zero first derivative does not tell us whether we have found a minimum or maximum.

Consider:

```math
f(x)
=
x^2
```

At:

```math
x
=
0
```

we have:

```math
f'(0)
=
0
```

and:

```math
f''(0)
=
2
>
0
```

The curve bends upward, so $x=0$ is a local minimum.

Now consider:

```math
f(x)
=
-x^2
```

Again:

```math
f'(0)
=
0
```

but:

```math
f''(0)
=
-2
<
0
```

The curve bends downward, so $x=0$ is a local maximum.

This gives the basic **second derivative test**.

---

### The Second Derivative Test

At a stationary point $x^{*}$ where:

```math
f'(x^{*})
=
0
```

if:

```math
f''(x^{*})
>
0
```

the point is locally bowl-shaped and is typically a **local minimum**.

If:

```math
f''(x^{*})
<
0
```

the point is locally hill-shaped and is typically a **local maximum**.

If:

```math
f''(x^{*})
=
0
```

the test is inconclusive.

So:

```text
first derivative  → Is the surface locally flat?
second derivative → How is the surface curved?
```

---

### When the Second Derivative Is Zero

Consider:

```math
f(x)
=
x^3
```

The first derivative is:

```math
f'(x)
=
3x^2
```

At:

```math
x
=
0
```

we obtain:

```math
f'(0)
=
0
```

The second derivative is:

```math
f''(x)
=
6x
```

so:

```math
f''(0)
=
0
```

But $x=0$ is neither a local minimum nor a local maximum.

The curve passes through a flat point and continues increasing.

This illustrates why:

```math
f'(x)
=
0
```

does not automatically mean:

```math
\text{minimum}
```

and why second-order information can help us understand the local geometry.

---

### Convexity

A differentiable function is convex over a region when it curves upward rather than downward.

For a one-dimensional twice-differentiable function, a useful condition is:

```math
f''(x)
\ge
0
```

throughout the region.

A strictly bowl-shaped function such as:

```math
f(x)
=
x^2
```

has:

```math
f''(x)
=
2
>
0
```

everywhere.

Convex functions are particularly attractive in optimization because any local minimum of a convex function is also a **global minimum**.

Conceptually:

```text
        \           /
         \         /
          \       /
           \_____/
              ^
        global minimum
```

There are no deceptive lower valleys elsewhere.

---

### Non-Convex Functions

A non-convex function may contain a much more complicated landscape:

```text
       /\          /\
      /  \__    __/  \
 ____/      \__/      \____
        ^        ^
      local    local
     minimum   minimum
```

Such a surface may contain:

- multiple local minima,
- local maxima,
- flat regions,
- saddle points.

Many modern neural-network loss landscapes are non-convex.

This makes their optimization considerably more complicated than minimizing a simple quadratic function.

---

### Local Minimum vs Global Minimum

A **local minimum** is lower than nearby points.

A **global minimum** is the lowest point over the entire domain.

Conceptually:

```text
       __
      /  \__
 ____/      \___
   ^            \____
 local               ^
 minimum         global minimum
```

Gradient-based optimization uses local derivative information.

Therefore, in a general non-convex landscape, finding a stationary point does not by itself guarantee that the globally smallest possible loss has been found.

---

### Saddle Points

Multivariable functions introduce another important possibility: the **saddle point**.

Consider:

```math
f(x,y)
=
x^2-y^2
```

Its partial derivatives are:

```math
\frac{\partial f}{\partial x}
=
2x
```

and:

```math
\frac{\partial f}{\partial y}
=
-2y
```

At:

```math
(x,y)
=
(0,0)
```

the gradient is:

```math
\nabla f
=
\begin{bmatrix}
0 \\
0
\end{bmatrix}
```

Yet this point is neither a minimum nor a maximum.

Along the $x$ direction:

```math
f(x,0)
=
x^2
```

the function curves upward.

Along the $y$ direction:

```math
f(0,y)
=
-y^2
```

the function curves downward.

The surface therefore resembles a saddle.

This is why:

> **Zero gradient means stationary, not necessarily optimal.**

---

### From Second Derivatives to the Hessian

For a function containing several variables, curvature can also vary in several directions.

Suppose:

```math
f
=
f(x,y)
```

We can compute second partial derivatives:

```math
\frac{\partial^2 f}{\partial x^2}
```

```math
\frac{\partial^2 f}{\partial y^2}
```

as well as mixed partial derivatives:

```math
\frac{\partial^2 f}{\partial x\partial y}
```

and:

```math
\frac{\partial^2 f}{\partial y\partial x}
```

These can be collected into a matrix called the **Hessian**:

```math
\mathbf{H}
=
\begin{bmatrix}
\frac{\partial^2 f}{\partial x^2}
&
\frac{\partial^2 f}{\partial x\partial y}
\\
\frac{\partial^2 f}{\partial y\partial x}
&
\frac{\partial^2 f}{\partial y^2}
\end{bmatrix}
```

For $n$ variables, the Hessian is an $n\times n$ matrix containing all second-order partial derivatives.

---

### Gradient vs Hessian

The gradient and Hessian describe different aspects of the local loss landscape.

The gradient:

```math
\nabla L
```

provides **first-order information**.

It tells us about local slope and direction.

The Hessian:

```math
\mathbf{H}
```

provides **second-order information**.

It tells us how those slopes themselves change.

A useful mental model is:

```text
gradient → which way is the terrain sloping?
Hessian  → how is the terrain bending?
```

---

### Why We Do Not Always Use the Hessian

Second-order information can improve optimization because it reveals curvature.

However, for a model containing millions or billions of parameters, the Hessian can become enormous.

For $n$ parameters:

```math
\nabla L
\in
\mathbb{R}^{n}
```

but:

```math
\mathbf{H}
\in
\mathbb{R}^{n\times n}
```

If a model has one million parameters, its full Hessian would conceptually contain:

```math
10^6
\times
10^6
=
10^{12}
```

entries.

Computing and storing such a matrix directly is usually impractical.

This is one reason first-order gradient-based methods are so important in large-scale machine learning.

They use far less information:

```math
\text{gradient}
\rightarrow
\text{first-order optimization}
```

while methods that explicitly exploit curvature belong broadly to:

```math
\text{Hessian information}
\rightarrow
\text{second-order optimization}
```

---

## ML Association

Curvature helps explain why optimization behaves differently across different loss landscapes.

For a simple convex loss:

```math
L(w)
=
w^2
```

the optimization problem is friendly.

There is one global minimum, and the surface consistently guides gradient descent toward it.

But complex models can produce high-dimensional, non-convex loss surfaces containing:

- different degrees of curvature,
- flat regions,
- saddle points,
- multiple valleys.

The gradient tells us:

```math
\text{where the loss increases fastest}
```

while curvature tells us:

```math
\text{how that direction itself is changing}
```

This distinction also explains why a very small gradient does not always mean:

```math
\text{training complete}
```

It may instead indicate:

```math
\text{minimum}
\quad\text{or}\quad
\text{saddle point}
\quad\text{or}\quad
\text{flat region}
```

Second-order information can distinguish some of these situations, although computing it explicitly becomes expensive for large models.

This gives us another useful hierarchy:

```math
\text{function}
\rightarrow
\text{derivative}
\rightarrow
\text{gradient}
\rightarrow
\text{curvature}
```

or, in optimization language:

```text
function   → where are we?
gradient   → which way is uphill?
curvature  → how is the terrain bending?
```

For large-scale AI, gradients usually do most of the day-to-day work.

But curvature explains much of the terrain they are forced to navigate.

> **The gradient gives the optimizer a compass; curvature tells us what kind of landscape the compass is navigating.**

---

## 9. Calculus of Important ML Functions

Machine-learning models repeatedly use a relatively small family of mathematical functions.

Among the most important are:

- polynomial functions,
- exponential functions,
- logarithms,
- sigmoid functions.

Understanding how these functions behave — and especially how their derivatives behave — makes many ML formulas much easier to interpret.

---

### Polynomial Functions

Polynomial functions appear frequently in regression, loss functions, and regularization.

A simple example is:

```math
f(x)
=
x^2
```

Its derivative is:

```math
f'(x)
=
2x
```

The function grows quadratically, while its derivative grows linearly.

For:

```math
f(x)
=
x^n
```

the power rule gives:

```math
\frac{d}{dx}x^n
=
nx^{n-1}
```

Squared terms are particularly common in machine learning.

For example, squared error uses:

```math
(y-\hat{y})^2
```

and L2 regularization uses terms such as:

```math
\lambda w^2
```

The derivative of the regularization term is:

```math
\frac{d}{dw}
\lambda w^2
=
2\lambda w
```

This means larger weights receive proportionally larger gradient contributions.

---

### Exponential Functions

The natural exponential function is:

```math
f(x)
=
e^x
```

Its derivative is:

```math
\frac{d}{dx}e^x
=
e^x
```

This property makes the exponential function particularly convenient in calculus.

If the exponent itself is a function:

```math
f(x)
=
e^{g(x)}
```

the chain rule gives:

```math
\frac{df}{dx}
=
e^{g(x)}g'(x)
```

For example:

```math
f(x)
=
e^{-x}
```

then:

```math
f'(x)
=
-e^{-x}
```

because:

```math
\frac{d}{dx}(-x)
=
-1
```

Exponential functions appear in:

- logistic regression,
- sigmoid functions,
- softmax,
- probability distributions,
- likelihood calculations.

---

### Logarithmic Functions

The natural logarithm is:

```math
f(x)
=
\ln x
```

Its derivative is:

```math
\frac{d}{dx}\ln x
=
\frac{1}{x}
```

for:

```math
x
>
0
```

If the logarithm contains another function:

```math
f(x)
=
\ln(g(x))
```

the chain rule gives:

```math
\frac{df}{dx}
=
\frac{g'(x)}
{g(x)}
```

For example:

```math
f(x)
=
\ln(x^2+1)
```

then:

```math
f'(x)
=
\frac{2x}
{x^2+1}
```

Logarithms are extremely important in machine learning because they convert products into sums and often make probability-based objectives easier to optimize.

---

### Why Logs Appear in Probability-Based Models

Suppose the likelihood of a dataset is expressed as a product:

```math
L
=
p_1p_2p_3\cdots p_n
```

Taking the logarithm gives:

```math
\ln L
=
\ln p_1
+
\ln p_2
+
\cdots
+
\ln p_n
```

This transformation is useful because sums are usually easier to manipulate and differentiate than long products.

Since the logarithm is a monotonically increasing function, maximizing:

```math
L
```

is equivalent to maximizing:

```math
\ln L
```

This is why many statistical and machine-learning models work with **log-likelihood** rather than likelihood directly.

---

### The Sigmoid Function

The sigmoid function is central to logistic regression and appears historically in neural networks.

It is defined as:

```math
\sigma(z)
=
\frac{1}
{1+e^{-z}}
```

The sigmoid maps any real number into the interval:

```math
0
<
\sigma(z)
<
1
```

This makes it convenient for representing probability-like outputs.

As:

```math
z
\rightarrow
+\infty
```

we obtain:

```math
\sigma(z)
\rightarrow
1
```

and as:

```math
z
\rightarrow
-\infty
```

we obtain:

```math
\sigma(z)
\rightarrow
0
```

At:

```math
z
=
0
```

we obtain:

```math
\sigma(0)
=
\frac{1}{2}
```

---

### Derivative of the Sigmoid

Start with:

```math
\sigma(z)
=
\frac{1}
{1+e^{-z}}
```

Rewrite it as:

```math
\sigma(z)
=
(1+e^{-z})^{-1}
```

Differentiate using the chain rule:

```math
\sigma'(z)
=
-(1+e^{-z})^{-2}
\left(
-e^{-z}
\right)
```

Therefore:

```math
\sigma'(z)
=
\frac{e^{-z}}
{(1+e^{-z})^2}
```

This expression can be rewritten in a much more elegant form:

```math
\sigma'(z)
=
\sigma(z)
\left(
1-\sigma(z)
\right)
```

This identity is extremely useful.

It means the derivative of the sigmoid can be computed directly from the sigmoid output itself.

---

### Interpreting the Sigmoid Derivative

The derivative is largest near:

```math
z
=
0
```

where:

```math
\sigma(z)
=
0.5
```

Then:

```math
\sigma'(z)
=
0.5(1-0.5)
=
0.25
```

As the sigmoid approaches either extreme:

```math
\sigma(z)
\rightarrow
0
```

or:

```math
\sigma(z)
\rightarrow
1
```

its derivative approaches zero.

So:

```math
\sigma'(z)
\rightarrow
0
```

in the saturated regions.

This means the function becomes increasingly flat near its extremes.

That observation later becomes important when studying the **vanishing gradient problem** in neural networks.

---

### Binary Cross-Entropy

For binary classification, a commonly used loss is binary cross-entropy:

```math
L
=
-
\left[
y\ln(\hat{y})
+
(1-y)\ln(1-\hat{y})
\right]
```

where:

- $y$ is the true label,
- $\hat{y}$ is the predicted probability.

If:

```math
y
=
1
```

the loss becomes:

```math
L
=
-\ln(\hat{y})
```

If the model predicts:

```math
\hat{y}
\rightarrow
1
```

then:

```math
L
\rightarrow
0
```

But if:

```math
\hat{y}
\rightarrow
0
```

then:

```math
L
\rightarrow
+\infty
```

So highly confident wrong predictions are penalized heavily.

---

### Why Logarithms Fit Classification Losses Well

Suppose the correct class is:

```math
y
=
1
```

Then:

```math
L
=
-\ln(\hat{y})
```

If:

```math
\hat{y}
=
0.9
```

the loss is relatively small.

If:

```math
\hat{y}
=
0.1
```

the loss becomes much larger.

This gives the optimizer a strong signal when the model is confidently wrong.

The logarithm therefore provides more than mathematical convenience.

Its shape creates a useful penalty structure for probabilistic predictions.

---

### Sigmoid and Cross-Entropy Together

In logistic regression:

```math
z
=
\mathbf{w}^{T}\mathbf{x}+b
```

The sigmoid converts this score into a predicted probability:

```math
\hat{y}
=
\sigma(z)
```

Then binary cross-entropy measures the prediction error:

```math
L
=
-
\left[
y\ln(\hat{y})
+
(1-y)\ln(1-\hat{y})
\right]
```

The overall chain is:

```math
\mathbf{x}
\rightarrow
z
\rightarrow
\sigma(z)
\rightarrow
\hat{y}
\rightarrow
L
```

When differentiating the loss with respect to model parameters, the chain rule operates through this entire sequence.

A remarkable simplification occurs.

For sigmoid combined with binary cross-entropy:

```math
\frac{\partial L}{\partial z}
=
\hat{y}-y
```

The complicated derivatives of the logarithm and sigmoid collapse into a very simple error term.

This is one reason the pairing of sigmoid and binary cross-entropy is mathematically natural.

---

### Softmax

For multiclass classification, the sigmoid is often replaced by the **softmax** function.

For class score $z_i$:

```math
P(y=i)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
```

Softmax converts a collection of scores into probabilities that sum to one.

The exponential function ensures positive values, while the denominator normalizes them:

```math
\sum_i P(y=i)
=
1
```

Softmax is commonly paired with categorical cross-entropy.

The derivatives are more involved than for sigmoid, but the same mathematical ingredients appear:

- exponentials,
- logarithms,
- partial derivatives,
- chain rule.

---

### Regularization Functions

Calculus also explains how regularization modifies model training.

For L2 regularization:

```math
R(w)
=
\lambda w^2
```

the derivative is:

```math
\frac{dR}{dw}
=
2\lambda w
```

This adds a gradient component proportional to the size of the weight.

Large weights therefore experience a stronger pull toward zero.

For L1 regularization:

```math
R(w)
=
\lambda |w|
```

the behaviour is different.

For:

```math
w
>
0
```

the derivative is:

```math
\frac{dR}{dw}
=
\lambda
```

and for:

```math
w
<
0
```

the derivative is:

```math
\frac{dR}{dw}
=
-\lambda
```

At:

```math
w
=
0
```

the ordinary derivative is not defined.

This sharp point helps explain why L1 regularization tends to drive some weights exactly to zero.

---

### Differentiability Matters

Gradient-based optimization requires useful derivative information.

A smooth function provides a well-defined slope across most of its domain.

But some functions are not differentiable everywhere.

For example:

```math
f(x)
=
|x|
```

has a sharp corner at:

```math
x
=
0
```

because the slope approaching from the left is:

```math
-1
```

while the slope approaching from the right is:

```math
1
```

So there is no unique ordinary derivative at zero.

Machine learning can still optimize functions with such points using related concepts such as **subgradients**, but the clean derivative picture becomes slightly more general.

---

## ML Association

Many apparently different machine-learning techniques repeatedly reuse the same calculus toolkit.

Linear regression with squared error relies heavily on polynomial derivatives.

Logistic regression combines:

```math
\text{linear score}
\rightarrow
\text{exponential}
\rightarrow
\text{sigmoid}
\rightarrow
\text{logarithmic loss}
```

Regularization adds additional functions to the objective:

```math
\text{model loss}
+
\text{penalty}
```

whose derivatives alter the gradient used during optimization.

Neural networks extend the same pattern across many layers:

```math
\text{linear transformation}
\rightarrow
\text{activation}
\rightarrow
\text{linear transformation}
\rightarrow
\cdots
\rightarrow
\text{loss}
```

The chain rule then connects all of those derivatives during backpropagation.

A useful summary is:

```text
polynomials  → squared loss and L2 penalties
exponentials → sigmoid, softmax, probabilities
logarithms   → likelihood and cross-entropy
chain rule   → connects them all
```

The important lesson is not to memorize isolated derivative formulas.

It is to recognize the small set of mathematical building blocks that repeatedly appear inside much larger ML systems.

> **Modern models may be enormous, but much of their calculus is built from a surprisingly small vocabulary of functions.**

---

## 10. Putting It Together: Calculus Behind Model Training

We have now encountered the major calculus ideas needed to understand gradient-based machine learning:

```text
rate of change
→ derivatives
→ chain rule
→ partial derivatives
→ gradients
→ gradient descent
→ curvature
```

It is useful to bring these ideas together in one complete example.

Consider the simplest possible linear regression model:

```math
\hat{y}
=
wx+b
```

The model receives an input $x$ and produces a prediction $\hat{y}$.

Suppose the true target is $y$.

Our goal is to find values of $w$ and $b$ that make the prediction as close as possible to the true value.

---

### Step 1: Define the Model

The prediction is:

```math
\hat{y}
=
wx+b
```

The trainable parameters are:

```math
w
\quad\text{and}\quad
b
```

The input $x$ is data.

The model does not change $x$.

Instead, learning changes $w$ and $b$.

We can therefore collect the parameters conceptually as:

```math
\boldsymbol{\theta}
=
\begin{bmatrix}
w \\
b
\end{bmatrix}
```

---

### Step 2: Make a Prediction

Suppose:

```math
x
=
2
```

and the true target is:

```math
y
=
5
```

Assume the model currently has:

```math
w
=
1
```

and:

```math
b
=
0
```

The prediction is:

```math
\hat{y}
=
(1)(2)+0
=
2
```

The model predicted $2$, while the correct value is $5$.

We now need a mathematical way to measure how wrong the prediction is.

---

### Step 3: Calculate the Loss

Use squared error:

```math
L
=
(y-\hat{y})^2
```

For this prediction:

```math
L
=
(5-2)^2
=
9
```

The model therefore has a loss of:

```math
L
=
9
```

But simply knowing that the loss is $9$ does not tell us how to improve the model.

We need to know:

> Should $w$ increase or decrease?

and:

> Should $b$ increase or decrease?

That requires derivatives.

---

### Step 4: Express the Full Computation

Substitute the prediction into the loss:

```math
L
=
\left(
y-(wx+b)
\right)^2
```

The dependency structure is:

```text
w, b
  ↓
wx + b
  ↓
prediction
  ↓
error
  ↓
squared error
  ↓
loss
```

More compactly:

```math
(w,b)
\rightarrow
\hat{y}
\rightarrow
L
```

The loss depends on the parameters indirectly through the prediction.

This is exactly the kind of nested dependency for which the chain rule was developed.

---

### Step 5: Differentiate with Respect to the Weight

We want:

```math
\frac{\partial L}{\partial w}
```

Start with:

```math
L
=
(y-\hat{y})^2
```

Using the chain rule:

```math
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial w}
```

First:

```math
\frac{\partial L}{\partial \hat{y}}
=
2(\hat{y}-y)
```

and because:

```math
\hat{y}
=
wx+b
```

we have:

```math
\frac{\partial \hat{y}}{\partial w}
=
x
```

Therefore:

```math
\frac{\partial L}{\partial w}
=
2(\hat{y}-y)x
```

For our example:

```math
\frac{\partial L}{\partial w}
=
2(2-5)(2)
```

so:

```math
\frac{\partial L}{\partial w}
=
-12
```

The weight gradient is negative.

This does **not** mean gradients are supposed to be negative.

It means that, at this particular point, increasing $w$ will locally reduce the loss.

---

### Step 6: Differentiate with Respect to the Bias

Now compute:

```math
\frac{\partial L}{\partial b}
```

Again:

```math
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial b}
```

We already know:

```math
\frac{\partial L}{\partial \hat{y}}
=
2(\hat{y}-y)
```

and:

```math
\frac{\partial \hat{y}}{\partial b}
=
1
```

Therefore:

```math
\frac{\partial L}{\partial b}
=
2(\hat{y}-y)
```

For our example:

```math
\frac{\partial L}{\partial b}
=
2(2-5)
=
-6
```

We now know how the loss responds to both trainable parameters.

---

### Step 7: Assemble the Gradient

The individual partial derivatives can be collected into the gradient:

```math
\nabla L
=
\begin{bmatrix}
\frac{\partial L}{\partial w}
\\
\frac{\partial L}{\partial b}
\end{bmatrix}
```

For the current model:

```math
\nabla L
=
\begin{bmatrix}
-12 \\
-6
\end{bmatrix}
```

This is the local slope information needed by the optimizer.

The gradient points toward increasing loss.

The optimizer therefore moves in the opposite direction.

---

### Step 8: Update the Parameters

Suppose the learning rate is:

```math
\eta
=
0.1
```

Gradient descent updates the weight:

```math
w_{\text{new}}
=
w
-
\eta
\frac{\partial L}{\partial w}
```

Substituting the values:

```math
w_{\text{new}}
=
1-(0.1)(-12)
```

Therefore:

```math
w_{\text{new}}
=
2.2
```

Similarly for the bias:

```math
b_{\text{new}}
=
b
-
\eta
\frac{\partial L}{\partial b}
```

so:

```math
b_{\text{new}}
=
0-(0.1)(-6)
```

and therefore:

```math
b_{\text{new}}
=
0.6
```

The model has learned new parameter values:

```math
w
=
2.2
```

```math
b
=
0.6
```

---

### Step 9: Make the Next Prediction

Using the updated model:

```math
\hat{y}_{\text{new}}
=
(2.2)(2)+0.6
```

Therefore:

```math
\hat{y}_{\text{new}}
=
5
```

The new loss is:

```math
L_{\text{new}}
=
(5-5)^2
=
0
```

For this deliberately simple single-example problem, one gradient step happened to land exactly on parameter values that predict the target.

That is not what normally happens in a real dataset.

With many training examples, changing parameters to improve one prediction may affect many others. Training therefore repeatedly calculates an aggregate loss and adjusts the parameters over many iterations.

But the underlying mechanism remains the same.

---

### From One Example to a Dataset

Suppose we have $N$ training examples.

For linear regression:

```math
\hat{y}_i
=
wx_i+b
```

A common loss is mean squared error:

```math
L
=
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
```

Now the gradient reflects the errors across the training examples.

For the weight:

```math
\frac{\partial L}{\partial w}
=
\frac{2}{N}
\sum_{i=1}^{N}
(\hat{y}_i-y_i)x_i
```

For the bias:

```math
\frac{\partial L}{\partial b}
=
\frac{2}{N}
\sum_{i=1}^{N}
(\hat{y}_i-y_i)
```

The optimizer then performs:

```math
w
\leftarrow
w
-
\eta
\frac{\partial L}{\partial w}
```

and:

```math
b
\leftarrow
b
-
\eta
\frac{\partial L}{\partial b}
```

This process repeats until a stopping condition is reached.

---

### The Complete Training Loop

We can now describe gradient-based model training from beginning to end:

```text
1. Initialize parameters
2. Feed training data into the model
3. Compute predictions
4. Compare predictions with targets
5. Compute loss
6. Differentiate the loss
7. Assemble the gradients
8. Update the parameters
9. Repeat
```

Mathematically:

```math
\mathbf{x}
\rightarrow
\hat{\mathbf{y}}
\rightarrow
L
\rightarrow
\nabla_{\boldsymbol{\theta}}L
\rightarrow
\boldsymbol{\theta}_{\text{new}}
```

and then the cycle begins again:

```math
\boldsymbol{\theta}_{\text{new}}
\rightarrow
\hat{\mathbf{y}}_{\text{new}}
\rightarrow
L_{\text{new}}
\rightarrow
\nabla L_{\text{new}}
\rightarrow
\cdots
```

This repeating loop is what we call **training**.

---

### Where Each Calculus Concept Fits

We can now place the concepts from this chapter into their roles.

**Derivatives** measure how one quantity changes with another:

```math
\frac{dL}{dw}
```

**Partial derivatives** allow us to examine one parameter of a multivariable loss at a time:

```math
\frac{\partial L}{\partial \theta_i}
```

**The chain rule** propagates derivatives through nested computations:

```math
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial w}
```

**The gradient** collects all parameter derivatives:

```math
\nabla_{\boldsymbol{\theta}}L
```

**Gradient descent** turns that information into parameter updates:

```math
\boldsymbol{\theta}
\leftarrow
\boldsymbol{\theta}
-
\eta
\nabla_{\boldsymbol{\theta}}L
```

**Second derivatives and curvature** help us understand the geometry of the loss surface on which this optimization occurs.

The concepts are therefore not independent pieces of mathematics.

They form one connected system.

---

### The Same Machinery in Logistic Regression

The model changes, but the learning principle does not.

Logistic regression begins with a linear score:

```math
z
=
\mathbf{w}^{T}\mathbf{x}+b
```

The score passes through the sigmoid:

```math
\hat{y}
=
\sigma(z)
```

Binary cross-entropy then measures the loss:

```math
L
=
-
\left[
y\ln(\hat{y})
+
(1-y)\ln(1-\hat{y})
\right]
```

So the forward computation is:

```math
\mathbf{x}
\rightarrow
z
\rightarrow
\sigma(z)
\rightarrow
\hat{y}
\rightarrow
L
```

Calculus travels backward through this chain.

For sigmoid with binary cross-entropy:

```math
\frac{\partial L}{\partial z}
=
\hat{y}-y
```

and therefore the weight gradient contains:

```math
\frac{\partial L}{\partial \mathbf{w}}
=
(\hat{y}-y)\mathbf{x}
```

The formulas differ from linear regression, but the pattern is identical:

```text
predict
→ measure error
→ differentiate
→ update parameters
```

---

### The Same Machinery in Neural Networks

A neural network introduces many more intermediate computations:

```math
\mathbf{x}
\rightarrow
\mathbf{z}_1
\rightarrow
\mathbf{a}_1
\rightarrow
\mathbf{z}_2
\rightarrow
\mathbf{a}_2
\rightarrow
\cdots
\rightarrow
\hat{\mathbf{y}}
\rightarrow
L
```

The forward pass calculates the prediction and loss.

Backpropagation works backward through the computational graph, repeatedly applying the chain rule to calculate:

```math
\frac{\partial L}{\partial w}
```

for every trainable weight.

The optimizer then updates those weights.

A neural network therefore does not abandon the mathematics used by simpler models.

It **scales it up**.

The same derivative that tells us how one weight affects a simple squared-error function becomes part of a massive collection of derivatives describing how millions or billions of parameters affect a neural-network loss.

---

## ML Association

The mathematical foundation of supervised gradient-based learning can now be compressed into one relationship:

```math
\boxed{
\text{Prediction}
\rightarrow
\text{Loss}
\rightarrow
\text{Gradient}
\rightarrow
\text{Parameter Update}
}
```

The model answers:

> **What do I currently predict?**

The loss function answers:

> **How wrong is that prediction?**

Calculus answers:

> **How does each parameter contribute to changing that error?**

The optimizer answers:

> **Given that information, how should the parameters change?**

And repeated parameter updates produce what we call **learning**.

This same architecture appears across models of dramatically different complexity:

```text
Linear Regression
        ↓
Logistic Regression
        ↓
Neural Networks
        ↓
Deep Neural Networks
```

The equations become larger.

The computational graphs become deeper.

The number of parameters may grow from a handful to billions.

But the mathematical idea remains recognizable:

```math
\text{parameters}
\rightarrow
\text{prediction}
\rightarrow
\text{loss}
\rightarrow
\frac{\partial L}{\partial\text{parameters}}
\rightarrow
\text{better parameters}
```

This is why calculus occupies such a central place in machine learning.

It provides the mathematical bridge between **being wrong** and **knowing how to change**.

> **Prediction produces an answer. Loss exposes the error. Calculus turns that error into a direction for learning.**

---

## 11. Chapter Summary

Calculus gives machine learning a way to reason about **change**.

A model contains parameters. Those parameters affect predictions. Predictions produce a loss. Calculus tells us how that loss changes when the parameters change.

The chapter can be compressed into the following chain:

```text
function
→ rate of change
→ derivative
→ partial derivative
→ gradient
→ optimization
→ learning
```

The derivative measures the rate of change of a function with respect to one variable:

```math
\frac{df}{dx}
```

For multivariable functions, partial derivatives measure sensitivity with respect to individual variables:

```math
\frac{\partial f}{\partial x_i}
```

These partial derivatives combine to form the gradient:

```math
\nabla f
=
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
```

The gradient points in the direction of steepest local increase.

Therefore:

```math
-\nabla f
```

points in the direction of steepest local decrease.

Gradient descent uses this fact:

```math
\boldsymbol{\theta}
\leftarrow
\boldsymbol{\theta}
-
\eta\nabla L
```

where:

- $\boldsymbol{\theta}$ represents model parameters,
- $L$ is the loss function,
- $\nabla L$ is the gradient,
- $\eta$ is the learning rate.

The chain rule allows derivatives to propagate through nested functions:

```math
\frac{dy}{dx}
=
\frac{dy}{du}
\frac{du}{dx}
```

This simple rule becomes enormously important in neural networks because models are built as chains of dependent computations.

Backpropagation repeatedly applies the chain rule to determine how each parameter influences the final loss.

Second derivatives extend the idea further.

The first derivative tells us about slope:

```math
f'(x)
```

while the second derivative tells us about curvature:

```math
f''(x)
```

For multivariable functions, second-order information is represented by the Hessian:

```math
\mathbf{H}
=
\left[
\frac{\partial^2 f}
{\partial x_i \partial x_j}
\right]
```

This allows us to distinguish different kinds of local geometry such as:

- minima,
- maxima,
- saddle points,
- flat regions.

Several common ML functions repeatedly use the same calculus rules.

Squared-error losses rely heavily on polynomial derivatives.

Sigmoid and softmax rely on exponentials.

Cross-entropy and likelihood methods rely on logarithms.

Regularization introduces additional terms whose derivatives alter the optimization path.

The overall picture is:

```text
model parameters
      ↓
prediction
      ↓
loss
      ↓
derivatives
      ↓
gradient
      ↓
optimizer
      ↓
updated parameters
      ↓
better prediction
```

This loop repeats during training.

The essential idea is therefore not merely that machine learning uses calculus.

It is that calculus provides the mechanism that converts **error into information about how the model should change**.

> **Prediction tells us where the model is. Loss tells us how wrong it is. Calculus tells us how to move.**

---

## 12. Calculus for ML — Quick Recall

### Derivative

Measures instantaneous rate of change.

```math
f'(x)
=
\frac{df}{dx}
```

Think:

```text
How fast is the function changing here?
```

---

### Partial Derivative

Measures change with respect to one variable while the others are held constant.

```math
\frac{\partial f}{\partial x_i}
```

Think:

```text
What happens if I turn only this parameter?
```

---

### Chain Rule

Differentiates nested functions.

```math
\frac{dy}{dx}
=
\frac{dy}{du}
\frac{du}{dx}
```

Think:

```text
How does change propagate through a chain of computations?
```

In ML:

```text
chain rule
→ backpropagation
```

---

### Gradient

Collects partial derivatives into a vector.

```math
\nabla L
=
\begin{bmatrix}
\frac{\partial L}{\partial \theta_1} \\
\frac{\partial L}{\partial \theta_2} \\
\vdots \\
\frac{\partial L}{\partial \theta_n}
\end{bmatrix}
```

Think:

```text
Which way is uphill?
```

---

### Negative Gradient

Points toward steepest local decrease.

```math
-\nabla L
```

Think:

```text
Which way is downhill?
```

---

### Gradient Descent

Uses the negative gradient to update model parameters.

```math
\boldsymbol{\theta}
\leftarrow
\boldsymbol{\theta}
-
\eta\nabla L
```

Think:

```text
gradient gives direction
learning rate gives step size
```

---

### Learning Rate

Controls the size of each optimization step.

```math
\eta
```

Too small:

```text
slow learning
```

Too large:

```text
overshooting or divergence
```

---

### Second Derivative

Measures how the slope itself changes.

```math
f''(x)
```

Think:

```text
How is the curve bending?
```

For a stationary point:

```math
f''(x)>0
\rightarrow
\text{local minimum}
```

```math
f''(x)<0
\rightarrow
\text{local maximum}
```

---

### Hessian

Multivariable equivalent of second-order curvature information.

```math
\mathbf{H}
=
\left[
\frac{\partial^2 f}
{\partial x_i\partial x_j}
\right]
```

Think:

```text
How is the loss surface bending in many directions?
```

---

### Important Derivatives

Power rule:

```math
\frac{d}{dx}x^n
=
nx^{n-1}
```

Exponential:

```math
\frac{d}{dx}e^x
=
e^x
```

Natural logarithm:

```math
\frac{d}{dx}\ln x
=
\frac{1}{x}
```

Sigmoid:

```math
\sigma(z)
=
\frac{1}{1+e^{-z}}
```

```math
\sigma'(z)
=
\sigma(z)
\left(
1-\sigma(z)
\right)
```

---

### The ML Training Mental Model

```text
Forward:
parameters
→ prediction
→ loss

Backward:
loss
→ derivatives
→ gradients

Update:
gradients
→ optimizer
→ new parameters
```

Or even more compactly:

```text
forward pass
→ loss
→ backpropagation
→ gradient
→ optimizer
→ update
```

The relationship between the major concepts is:

```text
derivatives
   ↓
partial derivatives
   ↓
gradient
   ↓
gradient descent
   ↓
parameter updates
   ↓
learning
```

and for neural networks:

```text
chain rule
   ↓
backpropagation
   ↓
gradients
   ↓
optimizer
   ↓
weight updates
```

The single most important takeaway from this chapter is:

> **Machine learning does not improve merely because it knows that it is wrong. It improves because calculus tells it how its parameters contributed to that wrongness and in which direction they should change.**

---

## 11. Integration and Its Role in Machine Learning

Most of this chapter has focused on **differentiation** because derivatives and gradients directly drive the optimization of many machine-learning models.

But calculus has another major branch: **integration**.

If differentiation asks:

> **How fast is something changing?**

integration asks:

> **How much has accumulated?**

These two ideas are deeply connected.

In machine learning, integration appears especially in **probability, statistics, continuous probability distributions, expected values, Bayesian methods, and probabilistic models**.

---

### Integration as Accumulation

Suppose we have a function:

```math
y
=
f(x)
```

The definite integral:

```math
\int_a^b f(x)\,dx
```

represents the accumulated quantity described by $f(x)$ between $x=a$ and $x=b$.

Geometrically, when $f(x)$ is non-negative, this can be interpreted as the **area under the curve** between those two points.

Conceptually:

```text
f(x)
 ^
 |              /\
 |            /    \
 |          /        \
 |________/____________\______> x
          a            b

       area under curve
             ↓
        definite integral
```

Instead of examining the function at one point, integration combines contributions across an entire interval.

---

### From Small Pieces to a Whole

Imagine dividing the interval from $a$ to $b$ into many tiny pieces.

Each piece has width:

```math
\Delta x
```

A small rectangle under the curve has approximately the area:

```math
f(x_i)\Delta x
```

Adding many such rectangles gives:

```math
\sum_i f(x_i)\Delta x
```

As the rectangles become arbitrarily narrow, the approximation approaches the exact integral:

```math
\int_a^b f(x)\,dx
```

So integration can be understood as:

```text
tiny contributions
→ add them together
→ make the pieces infinitesimally small
→ total accumulation
```

This idea becomes particularly important when dealing with **continuous quantities**.

---

### Indefinite Integrals and Antiderivatives

An **indefinite integral** asks for a function whose derivative produces the original function.

Suppose:

```math
f(x)
=
2x
```

We know:

```math
\frac{d}{dx}x^2
=
2x
```

Therefore:

```math
\int 2x\,dx
=
x^2+C
```

where $C$ is the **constant of integration**.

Why is $C$ necessary?

Because:

```math
\frac{d}{dx}(x^2)
=
2x
```

but also:

```math
\frac{d}{dx}(x^2+5)
=
2x
```

and:

```math
\frac{d}{dx}(x^2-100)
=
2x
```

Differentiation removes constants.

Integration therefore cannot determine which constant was originally present.

So:

```math
\int f(x)\,dx
=
F(x)+C
```

where:

```math
F'(x)
=
f(x)
```

The function $F(x)$ is called an **antiderivative** of $f(x)$.

---

### Definite Integrals

A definite integral has explicit boundaries:

```math
\int_a^b f(x)\,dx
```

If $F(x)$ is an antiderivative of $f(x)$, then:

```math
\int_a^b f(x)\,dx
=
F(b)-F(a)
```

For example:

```math
\int_0^2 2x\,dx
```

Since:

```math
\int 2x\,dx
=
x^2+C
```

we evaluate:

```math
\int_0^2 2x\,dx
=
\left[x^2\right]_0^2
```

Therefore:

```math
=
2^2-0^2
```

so:

```math
\int_0^2 2x\,dx
=
4
```

Unlike an indefinite integral, a definite integral produces a specific accumulated quantity over an interval.

---

### The Fundamental Theorem of Calculus

Differentiation and integration may initially appear to perform opposite kinds of tasks.

Differentiation breaks behaviour down into local rates of change.

Integration accumulates local contributions into a total.

The **Fundamental Theorem of Calculus** connects them.

If:

```math
F(x)
=
\int_a^x f(t)\,dt
```

then:

```math
F'(x)
=
f(x)
```

Conversely, if:

```math
F'(x)
=
f(x)
```

then:

```math
\int_a^b f(x)\,dx
=
F(b)-F(a)
```

So differentiation and integration are, in an important sense, inverse operations.

A useful mental model is:

```text
differentiation
whole behaviour → local rate of change

integration
local contributions → accumulated whole
```

---

### Why Integration Appears in Probability

Integration becomes especially important in machine learning because ML relies heavily on probability.

For a **discrete** random variable, probabilities can be added.

For example:

```math
P(X\in A)
=
\sum_{x\in A}P(X=x)
```

But a continuous random variable can take infinitely many possible values.

Instead of summing individual probabilities, we integrate a **probability density function**.

Suppose:

```math
p(x)
```

is the probability density of a continuous random variable $X$.

Then the probability that $X$ lies between $a$ and $b$ is:

```math
P(a\le X\le b)
=
\int_a^b p(x)\,dx
```

This is one of the most important applications of integration in statistics and machine learning.

The probability is literally represented by the **area under the probability-density curve** over the desired interval.

---

### Probability Density Is Not Probability

This distinction is important.

For a continuous random variable:

```math
p(x)
```

is a **density**, not the probability of observing exactly $x$.

In fact, for a continuous variable:

```math
P(X=x)
=
0
```

for any exact individual value $x$.

Probability arises from an interval:

```math
P(a\le X\le b)
=
\int_a^b p(x)\,dx
```

So:

```text
density at a point
≠
probability at that point
```

Instead:

```text
area under density over an interval
=
probability of that interval
```

---

### Normalization of Probability Distributions

A probability distribution must assign total probability equal to one.

For a continuous probability density:

```math
\int_{-\infty}^{\infty}
p(x)\,dx
=
1
```

This means the total area under the probability-density curve must equal one.

Conceptually:

```text
entire probability density
        ↓
integrate over all possible values
        ↓
        1
```

This requirement is called **normalization**.

Many probabilistic models involve constructing functions and ensuring that they integrate to one so they represent valid probability distributions.

---

### Cumulative Distribution Function

Integration also connects a probability density function to its **cumulative distribution function**, or CDF.

The CDF is:

```math
F(x)
=
P(X\le x)
```

For a continuous random variable:

```math
F(x)
=
\int_{-\infty}^{x}
p(t)\,dt
```

The CDF therefore accumulates probability from the far left of the distribution up to $x$.

And by the Fundamental Theorem of Calculus:

```math
\frac{dF(x)}{dx}
=
p(x)
```

This gives a beautiful connection:

```text
PDF
↓ integrate
CDF

CDF
↓ differentiate
PDF
```

Probability therefore uses both major branches of calculus together.

---

### Expected Value

Integration also allows us to calculate the expected value of a continuous random variable.

For a discrete variable:

```math
\mathbb{E}[X]
=
\sum_x xP(X=x)
```

For a continuous variable, the sum becomes an integral:

```math
\mathbb{E}[X]
=
\int_{-\infty}^{\infty}
x\,p(x)\,dx
```

More generally, for some function $g(X)$:

```math
\mathbb{E}[g(X)]
=
\int_{-\infty}^{\infty}
g(x)p(x)\,dx
```

This idea appears throughout machine learning.

Many objectives can be interpreted as minimizing or maximizing an **expected quantity** over some underlying data distribution.

---

### From Sums to Integrals

There is a useful pattern connecting discrete and continuous mathematics.

For discrete values:

```math
\sum_x
```

performs accumulation.

For continuous values:

```math
\int
```

plays the analogous role.

For example, discrete expectation is:

```math
\mathbb{E}[X]
=
\sum_x xP(X=x)
```

while continuous expectation is:

```math
\mathbb{E}[X]
=
\int x\,p(x)\,dx
```

This provides a useful mental shortcut:

> **When a probability problem moves from discrete outcomes to a continuous range, sums often become integrals.**

---

### Integration and Area Under a Curve

The phrase **area under the curve** appears frequently in machine learning.

One familiar example is the ROC curve used to evaluate binary classifiers.

The **Area Under the ROC Curve**, or ROC-AUC, summarizes the curve into a single value.

Conceptually:

```math
\text{AUC}
=
\int
\text{ROC curve}
```

In practice, an empirical ROC curve consists of discrete points rather than a perfectly smooth analytical function.

Its area is therefore commonly estimated numerically, for example using the **trapezoidal rule**.

The important connection here is conceptual:

```text
curve
+
integration
→
accumulated area
```

So the word **area** in *Area Under the Curve* is not merely descriptive terminology.

It comes directly from the integral interpretation of area.

---

### Numerical Integration

Not every integral has a convenient analytical solution.

Sometimes the function is too complicated.

Sometimes we only know the function at sampled data points.

In such situations, the integral can be approximated numerically.

Common numerical methods include:

- rectangle approximations,
- the trapezoidal rule,
- Simpson's rule,
- Monte Carlo integration.

For example, the trapezoidal rule approximates the area between neighbouring points using trapezoids rather than infinitesimally thin rectangles.

This is particularly relevant in computing because real systems frequently work with finite samples rather than perfect symbolic functions.

---

### Integration in Probabilistic Machine Learning

Integration appears whenever a probabilistic model needs to account for a continuous range of possible values.

For example, suppose a model contains an unobserved continuous variable $z$.

To obtain the probability of observed data $x$, we may need to account for every possible value of $z$:

```math
p(x)
=
\int
p(x,z)\,dz
```

This operation is called **marginalization**.

Conceptually:

```text
many possible hidden values of z
             ↓
account for all of them
             ↓
          integrate
             ↓
           p(x)
```

This idea becomes important in probabilistic graphical models, Bayesian machine learning, latent-variable models, and generative modelling.

Some of these integrals become extremely difficult or impossible to calculate exactly, which leads to approximation techniques such as sampling and variational inference.

Those topics belong to more advanced ML, but their mathematical foundation begins here.

---

## ML Association

Differentiation and integration play different but complementary roles in machine learning.

Differentiation is most visible during **optimization**:

```math
L
\rightarrow
\nabla L
\rightarrow
\text{parameter update}
```

Integration is especially visible in **probability**:

```math
\text{probability density}
\rightarrow
\text{integrate}
\rightarrow
\text{probability}
```

Integration gives us continuous probabilities:

```math
P(a\le X\le b)
=
\int_a^b p(x)\,dx
```

normalizes probability distributions:

```math
\int_{-\infty}^{\infty}
p(x)\,dx
=
1
```

calculates expectations:

```math
\mathbb{E}[X]
=
\int_{-\infty}^{\infty}
x\,p(x)\,dx
```

and marginalizes continuous variables:

```math
p(x)
=
\int p(x,z)\,dz
```

This creates a useful division of labour:

```text
Differentiation
→ change
→ slopes
→ gradients
→ optimization

Integration
→ accumulation
→ areas
→ probabilities
→ expectations
```

The two eventually meet throughout probabilistic machine learning.

A model may define a probability distribution using integrals and then learn its parameters using derivatives.

So although gradient-based training makes differentiation the more visible half of calculus in everyday ML, integration is quietly doing essential work underneath probability and statistics.

> **Differentiation tells a model how to change. Integration lets it reason over everything that could happen.**