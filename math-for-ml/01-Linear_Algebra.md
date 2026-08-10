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

# Linear Transformations

We have already seen that a matrix can transform a vector.

Now we make that idea more precise.

A **linear transformation** is a transformation that preserves the basic structure of vector space.

That sounds abstract, but the intuition is simple.

A linear transformation may:

- stretch
- shrink
- rotate
- reflect
- shear

vectors.

But it does so in a structured way.

---

# What Makes a Transformation Linear?

A transformation $begin:math:text$T$end:math:text$ is linear if it satisfies two rules.

For vectors $begin:math:text$\\mathbf\{u\}$end:math:text$ and $begin:math:text$\\mathbf\{v\}$end:math:text$,

$$
T(\mathbf{u}+\mathbf{v})
=
T(\mathbf{u})
+
T(\mathbf{v})
$$

and for any scalar $begin:math:text$c$end:math:text$,

$$
T(c\mathbf{u})
=
cT(\mathbf{u})
$$

These two properties are called:

- **Additivity**
- **Homogeneity**

Together they mean that the transformation respects vector addition and scalar multiplication.

---

# Why This Matters

Suppose

$$
\mathbf{v}
=
\mathbf{a}
+
\mathbf{b}
$$

If the transformation is linear, we can transform the parts separately:

$$
T(\mathbf{v})
=
T(\mathbf{a})
+
T(\mathbf{b})
$$

This gives linear transformations a predictable structure.

They do not arbitrarily warp space.

They preserve straight lines and relative structure.

---

# A Matrix Represents a Linear Transformation

Consider

$$
\mathbf{A}
=
\begin{bmatrix}
2 & 0\\
0 & 1
\end{bmatrix}
$$

and

$$
\mathbf{x}
=
\begin{bmatrix}
x\\
y
\end{bmatrix}
$$

Then

$$
\mathbf{A}\mathbf{x}
=
\begin{bmatrix}
2x\\
y
\end{bmatrix}
$$

This transformation doubles the horizontal component while leaving the vertical component unchanged.

Geometrically, the space has been stretched horizontally.

Every point changes according to the same rule.

---

# Scaling

A simple scaling matrix is

$$
\mathbf{S}
=
\begin{bmatrix}
2 & 0\\
0 & 3
\end{bmatrix}
$$

Applying it to

$$
\mathbf{x}
=
\begin{bmatrix}
x\\
y
\end{bmatrix}
$$

gives

$$
\mathbf{S}\mathbf{x}
=
\begin{bmatrix}
2x\\
3y
\end{bmatrix}
$$

The x-direction is stretched by a factor of 2.

The y-direction is stretched by a factor of 3.

---

# Reflection

Consider

$$
\mathbf{R}
=
\begin{bmatrix}
-1 & 0\\
0 & 1
\end{bmatrix}
$$

Then

$$
\mathbf{R}
\begin{bmatrix}
x\\
y
\end{bmatrix}
=
\begin{bmatrix}
-x\\
y
\end{bmatrix}
$$

This reflects every point across the y-axis.

The x-coordinate changes sign.

The y-coordinate remains unchanged.

---

# Rotation

A two-dimensional rotation can also be represented using a matrix.

For an angle $begin:math:text$\\theta$end:math:text$,

$$
\mathbf{R}
=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}
$$

Multiplying a vector by this matrix rotates it by $begin:math:text$\\theta$end:math:text$.

This is a beautiful example of how geometry becomes matrix multiplication.

---

# Shearing

A shear transformation shifts one coordinate according to another.

For example,

$$
\mathbf{H}
=
\begin{bmatrix}
1 & k\\
0 & 1
\end{bmatrix}
$$

gives

$$
\mathbf{H}
\begin{bmatrix}
x\\
y
\end{bmatrix}
=
\begin{bmatrix}
x+ky\\
y
\end{bmatrix}
$$

The y-coordinate remains unchanged, while the x-coordinate shifts according to $begin:math:text$y$end:math:text$.

A square can become a slanted parallelogram.

---

# The Origin Stays Fixed

A true linear transformation always maps the zero vector to the zero vector.

That is,

$$
T(\mathbf{0})
=
\mathbf{0}
$$

This property is important.

If a transformation shifts every point by some constant amount, it is not strictly linear.

For example,

$$
\mathbf{y}
=
\mathbf{A}\mathbf{x}
+
\mathbf{b}
$$

contains a translation due to $begin:math:text$\\mathbf\{b\}$end:math:text$.

Mathematically, this is called an **affine transformation**, not a purely linear transformation.

---

# Machine Learning Connection

This distinction appears constantly in Machine Learning.

A neuron computes

$$
\mathbf{z}
=
\mathbf{W}\mathbf{x}
+
\mathbf{b}
$$

The multiplication

$$
\mathbf{W}\mathbf{x}
$$

is a linear transformation.

The addition of

$$
\mathbf{b}
$$

shifts the result.

So the complete operation is technically **affine**.

In Machine Learning, however, people often casually refer to this entire step as a "linear layer."

That terminology is convenient, but mathematically the presence of bias makes it affine.

---

# Why Activation Functions Are Necessary

Suppose a neural network contains several layers but no nonlinear activation functions.

Then we may have

$$
\mathbf{W}_3
\mathbf{W}_2
\mathbf{W}_1
\mathbf{x}
$$

Because multiplying matrices together produces another matrix, this becomes

$$
\mathbf{W}
\mathbf{x}
$$

for some combined matrix $begin:math:text$\\mathbf\{W\}$end:math:text$.

So stacking many purely linear transformations still produces only another linear transformation.

The network would gain depth but not greater expressive power.

This is why activation functions are essential.

They introduce **nonlinearity**.

That nonlinear step prevents the entire network from collapsing mathematically into one equivalent matrix multiplication.

---

# A Powerful Mental Model

Think of a matrix as changing the coordinate space itself.

A vector enters.

The matrix:

- stretches some directions
- shrinks others
- rotates some directions
- mixes coordinates

and produces a new representation.

This idea becomes especially important later when studying:

- PCA
- Neural Networks
- Embeddings
- Computer Vision
- Transformers

In all of these cases, learning often means discovering useful transformations of the input space.

---

# Summary

A linear transformation:

- maps vectors to vectors
- preserves vector addition
- preserves scalar multiplication
- can be represented using a matrix
- can scale, rotate, reflect or shear space
- always maps the origin to the origin

When bias is added,

$$
\mathbf{W}\mathbf{x}+\mathbf{b}
$$

the transformation becomes affine.

Neural Networks rely on repeated affine transformations combined with nonlinear activation functions.

---

# Looking Ahead

Matrices transform vectors.

But one of the oldest uses of matrices is solving equations.

Suppose several unknown quantities must satisfy several relationships simultaneously.

That problem leads directly to **Systems of Linear Equations**.

---

# Systems of Linear Equations

A system of linear equations contains several equations that must all be satisfied at the same time.

For example,

$$
x+y=5
$$

and

$$
2x-y=1
$$

We are looking for values of $begin:math:text$x$end:math:text$ and $begin:math:text$y$end:math:text$ that satisfy both equations simultaneously.

---

# Solving the Simple Way

From

$$
x+y=5
$$

we get

$$
y=5-x
$$

Substitute this into

$$
2x-y=1
$$

to obtain

$$
2x-(5-x)=1
$$

Therefore,

$$
3x=6
$$

so

$$
x=2
$$

and therefore,

$$
y=3
$$

The solution is

$$
\begin{bmatrix}
2\\
3
\end{bmatrix}
$$

---

# Matrix Form

The same system can be written compactly as

$$
\begin{bmatrix}
1 & 1\\
2 & -1
\end{bmatrix}
\begin{bmatrix}
x\\
y
\end{bmatrix}
=
\begin{bmatrix}
5\\
1
\end{bmatrix}
$$

or more generally,

$$
\mathbf{A}\mathbf{x}
=
\mathbf{b}
$$

where

- $begin:math:text$\\mathbf\{A\}$end:math:text$ contains the coefficients
- $begin:math:text$\\mathbf\{x\}$end:math:text$ contains the unknowns
- $begin:math:text$\\mathbf\{b\}$end:math:text$ contains the results

This compact equation is one of the most important forms in Linear Algebra.

---

# Geometric Interpretation

Each linear equation describes a line.

For example,

$$
x+y=5
$$

describes one line.

And

$$
2x-y=1
$$

describes another.

The solution occurs where the two lines intersect.

So solving a system of equations means finding the point that satisfies all constraints at once.

---

# Three Possible Outcomes

Two lines may behave in three different ways.

## 1. One Unique Solution

The lines intersect at exactly one point.

The system has one solution.

---

## 2. No Solution

The lines are parallel.

They never meet.

The system is inconsistent.

---

## 3. Infinitely Many Solutions

The two equations describe the same line.

Every point on that line satisfies both equations.

---

# Why This Matters in Machine Learning

Many Machine Learning problems eventually produce equations of the form

$$
\mathbf{A}\mathbf{x}
=
\mathbf{b}
$$

For example, Linear Regression can be expressed using matrix equations.

In its closed-form solution,

$$
\hat{\boldsymbol{\beta}}
=
(\mathbf{X}^T\mathbf{X})^{-1}
\mathbf{X}^T\mathbf{y}
$$

the model is solving for the coefficient vector

$$
\hat{\boldsymbol{\beta}}
$$

using matrix operations.

The familiar regression problem is therefore deeply connected to solving systems of linear equations.

---

# Exact Solutions vs Approximate Solutions

In elementary mathematics, we often expect systems to have exact solutions.

Machine Learning is different.

Real-world datasets usually contain:

- noise
- measurement errors
- imperfect relationships
- redundant information

So there may be no vector of parameters that satisfies every equation exactly.

Instead, Machine Learning often asks:

> **Which solution gets us as close as possible?**

That idea leads directly to **least squares**, which forms the basis of Linear Regression.

---

# Overdetermined Systems

Suppose we have many equations but only a few unknowns.

For example,

100 observations but only 3 model parameters.

This creates an **overdetermined system**.

There are more constraints than unknowns.

Usually no exact solution exists.

Linear Regression therefore finds the parameter values that minimize the overall error.

---

# Underdetermined Systems

The opposite can also happen.

Suppose we have fewer equations than unknowns.

Then multiple solutions may satisfy the system.

This is called an **underdetermined system**.

Modern Machine Learning frequently operates in very high-dimensional spaces, so understanding this situation becomes increasingly important.

---

# Rank — A First Intuition

The idea of **rank** tells us how much independent information a matrix contains.

If several rows or columns carry redundant information, the effective dimensionality of the matrix is smaller than its apparent size.

For example,

$$
\begin{bmatrix}
1 & 2\\
2 & 4
\end{bmatrix}
$$

looks like a $begin:math:text$2\\times2$end:math:text$ matrix.

But the second row is simply twice the first.

So it does not contain completely new information.

This matrix has lower rank.

We will revisit this idea when discussing:

- linear independence
- basis
- eigenvalues
- dimensionality reduction

---

# Machine Learning Connection

Systems of equations appear behind many algorithms.

Examples include:

- Linear Regression
- Least Squares
- PCA
- Optimization
- Matrix Factorization
- Recommendation Systems

The notation

$$
\mathbf{A}\mathbf{x}
=
\mathbf{b}
$$

is therefore not merely a classroom exercise.

It is one of the fundamental patterns of applied Machine Learning.

---

# Summary

A system of linear equations:

- contains multiple relationships that must hold simultaneously
- can be written compactly as

$$
\mathbf{A}\mathbf{x}
=
\mathbf{b}
$$

- may have one solution, no solution or infinitely many solutions
- can be interpreted geometrically as intersections
- forms the mathematical basis of least squares and Linear Regression

Real-world Machine Learning frequently cannot solve systems exactly.

Instead, it searches for the **best approximate solution**.

---

# Looking Ahead

We have now seen vectors, matrices, transformations and systems of equations.

The next question is more structural:

> **How can a small collection of vectors describe an entire space?**

That leads us to three closely related ideas:

- Span
- Basis
- Linear Independence

---

# Span

We now know that vectors can be added and scaled.

That gives us a powerful idea:

> **A small collection of vectors can be combined to generate many other vectors.**

The set of all vectors that can be produced from such combinations is called the **span**.

---

# Linear Combinations

Suppose we have two vectors

$$
\mathbf{v}_1=
\begin{bmatrix}
1\\
0
\end{bmatrix}
$$

and

$$
\mathbf{v}_2=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

We may multiply each vector by any scalar and add the results:

$$
a\mathbf{v}_1+b\mathbf{v}_2
$$

where $$a$$ and $$b$$ are scalars.

This is called a **linear combination**.

For example,

$$
3\mathbf{v}_1+2\mathbf{v}_2
=
3
\begin{bmatrix}
1\\
0
\end{bmatrix}
+
2
\begin{bmatrix}
0\\
1
\end{bmatrix}
=
\begin{bmatrix}
3\\
2
\end{bmatrix}
$$

By choosing different values of $begin:math:text$a$end:math:text$ and $begin:math:text$b$end:math:text$, we can generate every point in the two-dimensional plane.

Therefore these two vectors **span** the plane.

---

# Geometric Intuition for Span

Consider just one non-zero vector:

$$
\mathbf{v}=
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

Multiplying it by different scalars gives

$$
\ldots,-2\mathbf{v},-\mathbf{v},0,\mathbf{v},2\mathbf{v},3\mathbf{v},\ldots
$$

Every resulting vector lies on the same straight line through the origin.

So one non-zero vector spans a **line**.

Now take two vectors pointing in genuinely different directions.

By scaling and adding them, we can move throughout a **plane**.

With three independent vectors in three-dimensional space, we can span the entire 3D space.

This gives us an important intuition:

```text
1 independent direction  → line
2 independent directions → plane
3 independent directions → 3D space
...
```

The number of genuinely independent directions determines the dimensionality of the space that can be generated.

---

# When Vectors Fail to Add a New Direction

Suppose

$$
\mathbf{v}_1=
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

and

$$
\mathbf{v}_2=
\begin{bmatrix}
2\\
4
\end{bmatrix}
$$

The second vector is simply

$$
\mathbf{v}_2=2\mathbf{v}_1
$$

Although we have two vectors, they point along the same direction.

Their linear combinations still generate only a line.

The second vector contributes no new direction.

This leads naturally to **linear independence**.

---

# Linear Independence

A collection of vectors is **linearly independent** if none of the vectors can be constructed from the others.

Each vector contributes genuinely new information or a new direction.

For example,

$$
\mathbf{v}_1=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
\mathbf{v}_2=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

are linearly independent.

Neither can be produced by scaling the other.

Together they introduce two distinct directions.

---

# Linear Dependence

Now consider

$$
\mathbf{v}_1=
\begin{bmatrix}
1\\
2
\end{bmatrix},
\qquad
\mathbf{v}_2=
\begin{bmatrix}
2\\
4
\end{bmatrix}
$$

Since

$$
\mathbf{v}_2=2\mathbf{v}_1
$$

the second vector does not introduce any new direction.

The vectors are therefore **linearly dependent**.

One contains redundant information.

---

# The Formal Test

Vectors

$$
\mathbf{v}_1,\mathbf{v}_2,\ldots,\mathbf{v}_n
$$

are linearly independent if the equation

$$
c_1\mathbf{v}_1+c_2\mathbf{v}_2+\cdots+c_n\mathbf{v}_n
=
\mathbf{0}
$$

has only the trivial solution

$$
c_1=c_2=\cdots=c_n=0
$$

If there is some non-zero combination of coefficients that produces the zero vector, the vectors are linearly dependent.

The formal definition may look abstract, but the intuition remains simple:

> **Independent vectors each contribute something new. Dependent vectors contain redundancy.**

---

# Machine Learning Connection — Redundant Features

Suppose a dataset contains these features:

- Height in centimetres
- Height in metres
- Weight

The first two features contain the same information because

$$
\text{height in metres}
=
0.01\times\text{height in centimetres}
$$

One feature is an exact linear combination of another.

This creates **linear dependence**.

In regression models, strong dependence between predictor variables is related to **multicollinearity**.

Redundant features can make model coefficients unstable and matrix calculations difficult.

So linear independence is not merely a geometric curiosity—it has direct practical importance in Machine Learning.

---

# Basis

We are now ready for one of the central ideas in Linear Algebra.

A **basis** is a minimal set of linearly independent vectors that spans a space.

Two conditions must therefore hold:

1. The vectors must span the entire space.
2. The vectors must be linearly independent.

A basis contains exactly enough directions to describe every vector in the space—no fewer and no redundant extras.

---

# The Standard Basis

In two dimensions, the most familiar basis is

$$
\mathbf{e}_1=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
\mathbf{e}_2=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

These are called the **standard basis vectors**.

Any vector

$$
\mathbf{x}=
\begin{bmatrix}
x\\
y
\end{bmatrix}
$$

can be written as

$$
\mathbf{x}=x\mathbf{e}_1+y\mathbf{e}_2
$$

For example,

$$
\begin{bmatrix}
3\\
2
\end{bmatrix}
=
3
\begin{bmatrix}
1\\
0
\end{bmatrix}
+
2
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

The numbers 3 and 2 are the coordinates of the vector relative to this basis.

---

# A Basis Is Not Unique

The standard basis is convenient, but it is not the only possible basis.

For example,

$$
\mathbf{u}_1=
\begin{bmatrix}
1\\
1
\end{bmatrix},
\qquad
\mathbf{u}_2=
\begin{bmatrix}
1\\
-1
\end{bmatrix}
$$

also form a basis for the two-dimensional plane.

They are linearly independent and together span the entire plane.

So the same vector can be described using different coordinate systems depending on the chosen basis.

This idea becomes extremely important later.

---

# Changing the Basis Changes the Description, Not the Vector

Imagine describing the location of a building.

One person may describe it using north-south and east-west directions.

Another may use roads running diagonally across the city.

The building itself has not moved.

Only the coordinate system used to describe it has changed.

A change of basis works in the same way.

> **The underlying information stays the same; only its representation changes.**

This is a powerful idea in Machine Learning because useful representations can make difficult problems much easier.

---

# Machine Learning Connection — PCA

Principal Component Analysis (PCA) can be understood partly as finding a more useful coordinate system for the data.

Instead of describing data using the original feature directions, PCA finds new directions that capture the greatest variation.

The data is then represented relative to those new directions.

In other words, PCA effectively changes the basis to one that is better aligned with the structure of the dataset.

We will revisit this when discussing eigenvectors.

---

# Dimension

The **dimension** of a vector space is the number of vectors required in any basis for that space.

For example:

- A line has dimension 1.
- A plane has dimension 2.
- Ordinary 3D space has dimension 3.

A dataset containing 100 features is naturally represented in a 100-dimensional feature space.

But its true information content may occupy fewer than 100 independent directions.

This distinction becomes very important in dimensionality reduction.

---

# Rank Revisited

Earlier, we introduced rank as the amount of independent information contained in a matrix.

We can now state this more precisely.

The **rank of a matrix** is the number of linearly independent directions represented by its rows or columns.

Consider

$$
\mathbf{A}=
\begin{bmatrix}
1 & 2\\
2 & 4
\end{bmatrix}
$$

The second row is twice the first.

Therefore only one row contributes genuinely new information.

The matrix has rank 1.

Now consider

$$
\mathbf{B}=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
$$

Its rows point in independent directions.

The matrix has rank 2.

---

# Full Rank

A matrix is called **full rank** when it contains the maximum possible number of independent rows or columns.

For a square $begin:math:text$n\\times n$end:math:text$ matrix, full rank means

$$
\operatorname{rank}(\mathbf{A})=n
$$

Full-rank matrices behave particularly well.

For example, a square full-rank matrix has an inverse.

A rank-deficient matrix does not.

This fact becomes important when solving systems of equations and when computing regression coefficients.

---

# Rank and Redundancy

A useful mental model is:

```text
Matrix size  → how much space is available
Matrix rank  → how much independent information is actually present
```

A matrix may have hundreds of columns but still have much lower rank if many features are redundant or correlated.

This is one reason dimensionality reduction can work.

The data may live in a large feature space while occupying a much smaller effective subspace.

---

# Connecting the Ideas

Span, linear independence, basis and rank are not separate topics.

They describe the same structure from different perspectives.

| Concept | Main Question |
|---------|---------------|
| Span | What space can these vectors generate? |
| Linear Independence | Does each vector add a genuinely new direction? |
| Basis | What is the smallest independent set that spans the space? |
| Dimension | How many basis vectors are required? |
| Rank | How many independent directions does this matrix actually contain? |

Together, these ideas tell us how much information is really present in a vector space or dataset.

---

# Machine Learning Connection

These concepts appear repeatedly in Machine Learning.

They help explain:

- redundant features
- multicollinearity
- matrix invertibility
- dimensionality reduction
- PCA
- low-rank approximations
- embeddings
- matrix factorization

A dataset may contain thousands of measured features while its meaningful structure occupies far fewer independent directions.

Finding those directions is one of the recurring themes of Machine Learning.

---

# Summary

A **linear combination** is formed by scaling vectors and adding them.

The **span** is the set of all vectors obtainable from those combinations.

Vectors are **linearly independent** when none can be constructed from the others.

A **basis** is a minimal independent set that spans the entire space.

The **dimension** tells us how many vectors a basis contains.

The **rank** of a matrix tells us how many independent directions or pieces of information the matrix actually contains.

These concepts give us the structural language required to understand high-dimensional data.

---

# Looking Ahead

So far, we have asked whether vectors point in independent directions.

Now we ask a more precise geometric question:

> **What happens when two directions are exactly perpendicular, and how can one vector be decomposed along another?**

That leads to:

- Orthogonality
- Projections
- Norms
- Distance

These ideas will connect Linear Algebra directly to similarity measures, least squares, K-Nearest Neighbours and PCA.

---

