# 01. Linear Algebra

> *"Linear Algebra is the mathematics of representing and transforming information."*

---

# Why Should I Care?

If Machine Learning had a native language, it would be Linear Algebra.

Every machine learning model eventually performs operations on vectors and matrices.

A regression model fits a line using vectors.

A decision boundary is described using vectors.

A neural network repeatedly multiplies matrices.

A recommendation system compares vectors.

Large Language Models represent the meaning of words, sentences and even entire documents as vectors.

Understanding Linear Algebra means understanding the mathematical language spoken by almost every AI system.

---

# Five-Minute Story

Imagine you're standing at the entrance of a large city.

Your friend calls and says,

> "Walk 3 km east, then 2 km north."

Without seeing a map, you know exactly where to go.

That instruction describes a **movement**.

Now imagine another friend gives you this instruction instead:

> "Walk 4 km northeast."

Different instruction.

Same destination.

Interesting.

A movement can be described in many different ways.

Mathematics represents every movement as an **arrow**.

That arrow has:

- a direction
- a length

This arrow is called a **vector**.

Now imagine plotting hundreds of such arrows.

Then thousands.

Eventually, every location in the city can be represented mathematically.

Machine Learning does exactly the same thing.

A customer.

A photograph.

A song.

A medical scan.

A sentence.

Everything eventually becomes vectors.

Learning Linear Algebra is learning how machines "see" information.

---

# Learning Objectives

After completing this chapter, you should be able to:

- Explain what Linear Algebra studies.
- Distinguish between scalars, vectors and matrices.
- Understand vector magnitude and direction.
- Perform basic vector operations.
- Explain the geometric meaning of the dot product.
- Understand matrix multiplication as a transformation.
- Develop intuition for linear transformations.
- Explain why eigenvalues and eigenvectors are important.
- Recognize Linear Algebra inside Machine Learning algorithms.

---

# What Is Linear Algebra?

Linear Algebra is the branch of mathematics concerned with representing information and transforming it.

Unlike ordinary arithmetic, which mostly deals with individual numbers, Linear Algebra studies collections of numbers that work together.

Those collections allow us to describe:

- locations
- movements
- relationships
- datasets
- images
- sounds
- documents
- machine learning models

Almost everything in AI is eventually expressed using these mathematical objects.

---

# The Three Building Blocks

Everything in Linear Algebra begins with three simple objects.

| Object | Intuition | Example |
|----------|-----------|---------|
| Scalar | One number | 5 |
| Vector | One ordered list of numbers | [3, 8, 2] |
| Matrix | A table of numbers | 3 × 3 matrix |

Nearly every concept in this chapter is built by combining these three objects in increasingly sophisticated ways.

---

# Scalar

A scalar is a single numerical quantity.

Examples include:

- Temperature
- Salary
- Age
- Height
- Learning rate
- Probability

Mathematically,

5

−3

2.718

π

are all scalars.

A scalar contains only **magnitude**.

It has no direction.

---

# Vector

A vector is an ordered collection of numbers.

Unlike a scalar, each position inside a vector has meaning.

For example,

x =

[170
 68
 25]

could represent

Height = 170 cm

Weight = 68 kg

Age = 25 years

Notice something subtle.

The vector does **not** represent three different people.

It represents **one observation** described using multiple features.

That idea appears everywhere in Machine Learning.

---

# Matrix

A matrix is a rectangular arrangement of numbers.

One useful way to think about a matrix is:

> **A matrix is simply a collection of vectors placed together.**

For example,

X =

170  68  25

180  82  31

165  60  22

Each row represents one observation.

Each column represents one feature.

Every dataset used in Machine Learning can be represented as a matrix.

---

# Machine Learning's View of Data

Suppose you open a spreadsheet.

| Height | Weight | Age |
|---------|---------|------|
|170|68|25|
|180|82|31|
|165|60|22|

Humans see a table.

Excel sees cells.

A Machine Learning algorithm sees

X

= matrix

Nothing more.

Nothing less.

The names of the columns disappear.

Formatting disappears.

Colours disappear.

Only numbers remain.

This matrix becomes the input to every learning algorithm.

---

# Rows and Columns

Rows and columns have different meanings.

Rows usually represent **observations**.

Columns usually represent **features**.

For example,

Rows:

- Person 1
- Person 2
- Person 3

Columns:

- Height
- Weight
- Age

This convention appears in almost every ML library, including NumPy, pandas and scikit-learn.

---

# Dimension

The **dimension** tells us how many values are required to describe something.

A person's location on a straight road requires one number.

One-dimensional.

A location on a map requires two numbers.

Two-dimensional.

A point in space requires three numbers.

Three-dimensional.

A customer described by 25 features lives in a **25-dimensional feature space**.

Machine Learning routinely works with hundreds or thousands of dimensions.

Large Language Models often work with embeddings containing several thousand dimensions.

Although impossible to visualize directly, the mathematics remains exactly the same.

---

# Why This Matters

From this point onward, almost every chapter in this handbook will reuse these ideas.

When we study:

- Linear Regression
- Logistic Regression
- PCA
- Gradient Descent
- Neural Networks
- Transformers
- Embeddings

the data will always be represented as vectors and matrices.

Learning Linear Algebra is therefore not learning "another subject."

It is learning the common language spoken by all of them.

---

# Looking Ahead

Now that we know what vectors and matrices are, the next question naturally arises:

> **How do we perform useful operations on them?**

That is where vector addition, scalar multiplication and the dot product begin.

---

# Vector Operations

So far, we have learned what vectors are.

The next natural question is:

> **What can we do with them?**

Just as ordinary arithmetic allows us to add, subtract and multiply numbers, Linear Algebra defines operations on vectors.

These operations allow us to describe movement, measure similarity and transform data.

Many Machine Learning algorithms rely on nothing more than these simple operations applied millions or billions of times.

---

# Vector Addition

Suppose two people push a car.

One pushes towards the east.

Another pushes towards the north.

The car moves according to the **combined effect** of both pushes.

Vector addition works in exactly the same way.

Two vectors are added by adding their corresponding components.

For example,

$$
\mathbf{a} =
\begin{bmatrix}
2\\
3
\end{bmatrix}\,
\qquad
\mathbf{b} =
\begin{bmatrix}
4\\
1
\end{bmatrix}
$$

Then,

$$
\mathbf{a}+\mathbf{b}
=
\begin{bmatrix}
2+4\\
3+1
\end{bmatrix}
=
\begin{bmatrix}
6\\
4
\end{bmatrix}
$$

Notice that addition happens element by element.

---

## Geometric Interpretation

Imagine placing the tail of the second vector at the head of the first.

The vector from the starting point to the final point is their sum.

This is known as the **head-to-tail rule**.

Vector addition therefore combines movements.

---

## Machine Learning Connection

Suppose a customer's profile is represented by a vector.

A recommendation algorithm may slightly modify that vector after observing new behaviour.

Conceptually,

```
Updated Profile

=

Old Profile

+

Behaviour Change
```

Many learning algorithms repeatedly update vectors in exactly this way.

---

# Vector Subtraction

Subtraction measures the difference between two vectors.

For example,

$$
\begin{bmatrix}
7\\
5
\end{bmatrix}
-
\begin{bmatrix}
2\\
1
\end{bmatrix}
=
\begin{bmatrix}
5\\
4
\end{bmatrix}
$$

Instead of asking

> "Where are we?"

subtraction asks

> **"How far apart are we?"**

---

## Machine Learning Connection

Prediction errors are differences between vectors.

Gradient Descent computes how much parameters should change.

Nearest-neighbour algorithms compare vectors by measuring differences.

Vector subtraction appears almost everywhere.

---

# Scalar Multiplication

A scalar can stretch or shrink a vector.

Suppose

$$
\mathbf{x}
=
\begin{bmatrix}
2\\
4
\end{bmatrix}
$$

Multiplying by 3 gives

$$
3\mathbf{x}
=
\begin{bmatrix}
6\\
12
\end{bmatrix}
$$

Every component is multiplied by the same scalar.

---

## What Happens Geometrically?

Multiplying by

- 2 doubles the length.
- 0.5 halves the length.
- -1 reverses the direction.
- 0 collapses the vector to the origin.

The direction remains unchanged unless the scalar is negative.

---

## Machine Learning Connection

Learning rates in Gradient Descent scale update vectors.

Feature scaling also involves multiplying vectors by constants.

Weights inside neural networks are repeatedly scaled during training.

---

# Vector Magnitude (Length)

Sometimes we need to know not where a vector points, but **how long it is**.

The length of a vector is called its **magnitude** or **norm**.

For

$$
\mathbf{x}
=
\begin{bmatrix}
3\\
4
\end{bmatrix}
$$

its magnitude is

$$
\|\|\mathbf{x}\|\|
=
\sqrt{3\^2+4\^2}
=
5
$$

This is simply Pythagoras' theorem.

---

## Why Is Magnitude Important?

Magnitude measures the "size" of a vector.

Examples include:

- Distance travelled
- Strength of a force
- Length of a movement
- Size of an embedding

Many Machine Learning algorithms compare magnitudes before making decisions.

---

# Unit Vectors

A **unit vector** has magnitude equal to exactly one.

Instead of representing size, it represents only direction.

For example,

$$
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

points upward with unit length.

Unit vectors become extremely important in optimization and geometry.

---

# Summary

We have now learned four fundamental vector operations.

| Operation | Purpose |
|-----------|---------|
| Addition | Combine vectors |
| Subtraction | Find differences |
| Scalar Multiplication | Scale vectors |
| Magnitude | Measure length |

These simple operations form the vocabulary upon which the rest of Linear Algebra is built.

---

# Looking Ahead

So far, every operation has treated vectors independently.

The next question is much more interesting.

> **How similar are two vectors?**

Answering that question leads us to one of the most important ideas in all of Machine Learning:

**The Dot Product.**

---

# Dot Product

So far, we have learned how to add, subtract and scale vectors.

Now we arrive at one of the most important operations in Linear Algebra:

> **The Dot Product**

If there is one mathematical operation you should remember from this chapter, it is this one.

The dot product appears repeatedly throughout Machine Learning and Deep Learning.

---

# What Problem Does It Solve?

Suppose we have two vectors.

Rather than asking

> "How long are they?"

or

> "Where do they point?"

we ask a different question:

> **"How similar are they?"**

The dot product gives us a numerical measure of how much two vectors point in the same direction.

---

# Mathematical Definition

For two vectors

$$
\mathbf{a}=
\begin{bmatrix}
a\_1\\
a\_2\\
\vdots\\
a\_n
\end{bmatrix}\,
\qquad
\mathbf{b}=
\begin{bmatrix}
b\_1\\
b\_2\\
\vdots\\
b\_n
\end{bmatrix}
$$

their dot product is

$$
\mathbf{a}\cdot\mathbf{b}
=
a\_1b\_1+a\_2b\_2+\cdots+a\_nb\_n
$$

The process is simple.

1. Multiply corresponding elements.
2. Add the results.

The answer is **a scalar**, not another vector.

---

# Worked Example

Suppose

$$
\mathbf{a}=
\begin{bmatrix}
2\\
3
\end{bmatrix}\,
\qquad
\mathbf{b}=
\begin{bmatrix}
4\\
5
\end{bmatrix}
$$

Then

$$
\mathbf{a}\cdot\mathbf{b}
=
2\times4
+
3\times5
=
8+15
=
23
$$

Notice that two vectors produced a single number.

---

# Geometric Interpretation

The dot product is more than just multiplication.

It measures how well two vectors are aligned.

Imagine two people walking.

If both walk in exactly the same direction,

the dot product is **large and positive**.

If they walk at right angles,

the dot product is **zero**.

If they walk in exactly opposite directions,

the dot product becomes **negative**.

This gives the dot product an intuitive geometric meaning.

It measures directional agreement.

---

# Relationship with the Angle

The dot product can also be written as

$$
\mathbf{a}\cdot\mathbf{b}
=
\|\|\mathbf{a}\|\|
\\,
\|\|\mathbf{b}\|\|
\cos\theta
$$

where

- $begin:math:text$\|\|\mathbf{a}\|\|$end:math:text$ is the magnitude of vector **a**
- $begin:math:text$\|\|\mathbf{b}\|\|$end:math:text$ is the magnitude of vector **b**
- $begin:math:text$\theta$end:math:text$ is the angle between them

This equation reveals something profound.

The dot product is largest when the vectors point in the same direction.

It becomes zero when they are perpendicular.

It becomes negative when they point in opposite directions.

---

# Why Perpendicular Means Zero

Suppose two movements are completely independent.

Walking north does not move you east.

Walking east does not move you north.

The two movements contribute nothing to each other.

Their dot product is zero.

Vectors with a dot product of zero are called **orthogonal**.

Orthogonality is one of the most important concepts in Linear Algebra.

---

# Machine Learning Connection

Suppose a model has learned weights

$$
\mathbf{w}
=
\begin{bmatrix}
0\.5\\
1\.2\\
-0\.8
\end{bmatrix}
$$

A new observation arrives

$$
\mathbf{x}
=
\begin{bmatrix}
170\\
68\\
25
\end{bmatrix}
$$

The model computes

$$
z=\mathbf{w}\^T\mathbf{x}
$$

This is nothing more than a dot product.

Every neuron inside a neural network performs this calculation.

Every perceptron performs this calculation.

Linear Regression performs this calculation.

Logistic Regression performs this calculation.

The first step of modern AI is often just one very sophisticated dot product.

---

# A Mental Model

Think of the dot product as asking

> "How strongly do these two vectors agree?"

The larger the agreement,

the larger the dot product.

The more independent they are,

the closer it moves toward zero.

The more opposite they become,

the more negative the answer becomes.

---

# Summary

The dot product:

- multiplies corresponding components
- produces a scalar
- measures directional similarity
- becomes zero for perpendicular vectors
- forms the mathematical heart of many Machine Learning algorithms

If vectors are the language of Machine Learning,

the dot product is one of its most frequently spoken words.

---

# Matrix Multiplication

The dot product lets us compare two vectors.

Matrix multiplication takes the next step.

> **It lets us transform one collection of numbers into another.**

This is one of the most fundamental operations in Machine Learning and Deep Learning.

At first glance, matrix multiplication often appears to be nothing more than a strange "row by column" recipe.

While that recipe is mathematically correct, it hides the real beauty of the operation.

A much better way to think about matrix multiplication is this:

> **A matrix is a mathematical function that transforms vectors.**

This single idea will eventually explain Linear Regression, Neural Networks, Principal Component Analysis (PCA), Computer Graphics, and even Transformers.

---

# Matrix × Vector Multiplication

Suppose we have the vector

$$
\mathbf{x}=
\begin{bmatrix}
2\
3
\end{bmatrix}
$$

and the matrix

$$
\mathbf{A}=
\begin{bmatrix}
1 & 0\
0 & 2
\end{bmatrix}
$$

Multiplying them gives

$$
\mathbf{A}\mathbf{x}
=
\begin{bmatrix}
1 & 0\
0 & 2
\end{bmatrix}
\begin{bmatrix}
2\
3
\end{bmatrix}
=
\begin{bmatrix}
2\
6
\end{bmatrix}
$$

Notice what happened.

The original vector

$$
\begin{bmatrix}
2\
3
\end{bmatrix}
$$

became

$$
\begin{bmatrix}
2\
6
\end{bmatrix}
$$

The matrix stretched the second coordinate while leaving the first coordinate unchanged.

The vector has been **transformed**.

This is the real purpose of matrix multiplication.

---

# Where Does the Row-by-Column Rule Come From?

Suppose

$$
\mathbf{A}=
\begin{bmatrix}
a & b\
c & d
\end{bmatrix}
$$

and

$$
\mathbf{x}=
\begin{bmatrix}
x_1\
x_2
\end{bmatrix}
$$

Then

$$
\mathbf{A}\mathbf{x}
=
\begin{bmatrix}
ax_1+bx_2\
cx_1+dx_2
\end{bmatrix}
$$

Look carefully at the first output.

$$
ax_1+bx_2
$$

That is simply the **dot product** between

- the first row of the matrix
- the input vector

The second output is another dot product.

So matrix-vector multiplication is really nothing more than **many dot products performed together**.

Once you understand the dot product, matrix multiplication becomes much less mysterious.

---

# Worked Example

Let

$$
\mathbf{A}=
\begin{bmatrix}
2 & 1\
3 & 4
\end{bmatrix}
$$

and

$$
\mathbf{x}=
\begin{bmatrix}
5\
2
\end{bmatrix}
$$

The first element of the output is

$$
2\times5+1\times2=12
$$

The second element is

$$
3\times5+4\times2=23
$$

Therefore

$$
\mathbf{A}\mathbf{x}
=
\begin{bmatrix}
12\
23
\end{bmatrix}
$$

Notice that one vector has been transformed into another vector.

---

# Matrix × Matrix Multiplication

Matrices can also transform other matrices.

Suppose

$$
\mathbf{A}=
\begin{bmatrix}
1 & 2\
3 & 4
\end{bmatrix}
$$

and

$$
\mathbf{B}=
\begin{bmatrix}
5 & 6\
7 & 8
\end{bmatrix}
$$

Then

$$
\mathbf{A}\mathbf{B}
=
\begin{bmatrix}
1\times5+2\times7 & 1\times6+2\times8\
3\times5+4\times7 & 3\times6+4\times8
\end{bmatrix}
=
\begin{bmatrix}
19 & 22\
43 & 50
\end{bmatrix}
$$

Every element is computed by taking the dot product of

- one row from the first matrix
- one column from the second matrix

---

# Shape Compatibility

Unlike ordinary multiplication, matrices cannot always be multiplied.

The number of columns in the first matrix **must equal** the number of rows in the second matrix.

If

$$
\mathbf{A}
\text{ has shape }
(m\times n)
$$

and

$$
\mathbf{B}
\text{ has shape }
(n\times p)
$$

then

$$
\mathbf{A}\mathbf{B}
$$

is valid and has shape

$$
(m\times p)
$$

A useful memory trick is

```text
(m × n) · (n × p) = (m × p)
       ↑     ↑
   Must Match
```

The **inner dimensions must match**.

The **outer dimensions become the shape of the answer**.

This simple rule will save you from countless mistakes.

---

# Why Order Matters

With ordinary numbers,

$$
ab=ba
$$

This property is called the **commutative property**.

Matrix multiplication is different.

In general,

$$
\mathbf{A}\mathbf{B}
\neq
\mathbf{B}\mathbf{A}
$$

Why?

Because transformations have an order.

Imagine editing a photograph.

Suppose you perform two operations:

1. Rotate the image by 90°.
2. Stretch it horizontally.

Now reverse the order.

1. Stretch it horizontally.
2. Rotate it by 90°.

The final images are different.

The same idea applies to matrices.

Changing the order changes the transformation.

---

# Machine Learning Connection

Suppose a neural network receives an input vector

$$
\mathbf{x}
$$

It also stores a matrix of learned weights

$$
\mathbf{W}
$$

The first computation performed by the neuron is

$$
\mathbf{z}
=
\mathbf{W}\mathbf{x}
+
\mathbf{b}
$$

where

- **W** is the weight matrix
- **x** is the input vector
- **b** is the bias vector

This is nothing more than

- matrix multiplication
- followed by vector addition

Every hidden layer inside a neural network repeats this exact computation.

A deep neural network is therefore a sequence of transformations:

```text
Input Vector
      │
      ▼
Matrix Transformation
      │
      ▼
Activation Function
      │
      ▼
Matrix Transformation
      │
      ▼
Activation Function
      │
      ▼
Prediction
```

Whether the network contains one hundred parameters or one hundred billion parameters, the underlying mathematics is still the same.

---

# A Better Mental Model

Many students memorize:

> "Multiply rows by columns."

While correct, that description does not explain *why* matrix multiplication exists.

Instead, remember this:

> **A matrix is a machine that accepts a vector, transforms it according to a rule, and produces a new vector.**

Once you begin thinking this way, Linear Algebra becomes much more intuitive.

---

# Summary

Matrix multiplication:

- transforms vectors into new vectors
- combines many dot products into one operation
- requires compatible dimensions
- is generally **not commutative**
- forms the computational backbone of Machine Learning and Deep Learning

Although the arithmetic may seem mechanical at first, its true meaning is geometric: **matrix multiplication transforms information.**

---

# Looking Ahead

We now know that matrices transform vectors.

The next natural question is:

> **What kinds of transformations are possible?**

To answer that, we move from arithmetic to geometry and study **Linear Transformations**.

---

