# 01. Linear Algebra

> *"Linear Algebra is the mathematics of representing and transforming information."*

---

## 1. Orientation and Motivation

### Why Should I Care?

If Machine Learning had a native language, it would be Linear Algebra.

Every machine learning model eventually performs operations on vectors and matrices.

A regression model fits a line using vectors.

A decision boundary is described using vectors.

A neural network repeatedly multiplies matrices.

A recommendation system compares vectors.

Large Language Models represent the meaning of words, sentences and even entire documents as vectors.

Understanding Linear Algebra means understanding the mathematical language spoken by almost every AI system.

---

### Five-Minute Story

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

### Learning Objectives

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

## 2. Foundations: Scalars, Vectors, and Matrices

### What Is Linear Algebra?

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

### The Three Building Blocks

Everything in Linear Algebra begins with three simple objects.

| Object | Intuition | Example |
|----------|-----------|---------|
| Scalar | One number | 5 |
| Vector | One ordered list of numbers | $\begin{bmatrix}3 & 8 & 2\end{bmatrix}$ |
| Matrix | A table of numbers | $3 \times 3$ matrix |

Nearly every concept in this chapter is built by combining these three objects in increasingly sophisticated ways.

---

#### Scalar

A scalar is a single numerical quantity.

Examples include:

- Temperature
- Salary
- Age
- Height
- Learning rate
- Probability

Mathematically,

- 5
- −3
- 2.718
- π

are all scalars.

A scalar contains only **magnitude**.

It has no direction.

---

#### Vector

A vector is an ordered collection of numbers.

Unlike a scalar, each position inside a vector has meaning.

For example,

```math
\mathbf{x} =
\begin{bmatrix}
170\\
68\\
25
\end{bmatrix}
```

could represent

- Height = 170 cm
- Weight = 68 kg
- Age = 25 years

Notice something subtle.

The vector does **not** represent three different people.

It represents **one observation** described using multiple features.

That idea appears everywhere in Machine Learning.

---

#### Matrix

A matrix is a rectangular arrangement of numbers.

One useful way to think about a matrix is:

> **A matrix is simply a collection of vectors placed together.**

For example,

```math
\mathbf{X} =
\begin{bmatrix}
170 & 68 & 25\\
180 & 82 & 31\\
165 & 60 & 22
\end{bmatrix}
```

Each row represents one observation.

Each column represents one feature.

Every dataset used in Machine Learning can be represented as a matrix.

---

### Machine Learning's View of Data

Suppose you open a spreadsheet.

| Height | Weight | Age |
|---------|---------|------|
|170|68|25|
|180|82|31|
|165|60|22|

Humans see a table.

Excel sees cells.

A Machine Learning algorithm sees $\mathbf{X}$: a matrix.

Nothing more.

Nothing less.

The names of the columns disappear.

Formatting disappears.

Colours disappear.

Only numbers remain.

This matrix becomes the input to every learning algorithm.

---

#### Rows and Columns

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

#### Dimension

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

### Why This Matters

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

### Looking Ahead

Now that we know what vectors and matrices are, the next question naturally arises:

> **How do we perform useful operations on them?**

That is where vector addition, scalar multiplication and the dot product begin.

---

## 3. Working with Vectors

### Vector Operations

So far, we have learned what vectors are.

The next natural question is:

> **What can we do with them?**

Just as ordinary arithmetic allows us to add, subtract and multiply numbers, Linear Algebra defines operations on vectors.

These operations allow us to describe movement, measure similarity and transform data.

Many Machine Learning algorithms rely on nothing more than these simple operations applied millions or billions of times.

---

#### Vector Addition

Suppose two people push a car.

One pushes towards the east.

Another pushes towards the north.

The car moves according to the **combined effect** of both pushes.

Vector addition works in exactly the same way.

Two vectors are added by adding their corresponding components.

For example,

```math
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
```

Then,

```math
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
```

Notice that addition happens element by element.

---

##### Geometric Interpretation

Imagine placing the tail of the second vector at the head of the first.

The vector from the starting point to the final point is their sum.

This is known as the **head-to-tail rule**.

Vector addition therefore combines movements.

---

##### Machine Learning Connection

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

#### Vector Subtraction

Subtraction measures the difference between two vectors.

For example,

```math
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
```

Instead of asking

> "Where are we?"

subtraction asks

> **"How far apart are we?"**

---

##### Machine Learning Connection

Prediction errors are differences between vectors.

Gradient Descent computes how much parameters should change.

Nearest-neighbour algorithms compare vectors by measuring differences.

Vector subtraction appears almost everywhere.

---

#### Scalar Multiplication

A scalar can stretch or shrink a vector.

Suppose

```math
\mathbf{x}
=
\begin{bmatrix}
2\\
4
\end{bmatrix}
```

Multiplying by 3 gives

```math
3\mathbf{x}
=
\begin{bmatrix}
6\\
12
\end{bmatrix}
```

Every component is multiplied by the same scalar.

---

##### What Happens Geometrically?

Multiplying by

- 2 doubles the length.
- 0.5 halves the length.
- -1 reverses the direction.
- 0 collapses the vector to the origin.

The direction remains unchanged unless the scalar is negative.

---

##### Machine Learning Connection

Learning rates in Gradient Descent scale update vectors.

Feature scaling also involves multiplying vectors by constants.

Weights inside neural networks are repeatedly scaled during training.

---

#### Vector Magnitude (Length)

Sometimes we need to know not where a vector points, but **how long it is**.

The length of a vector is called its **magnitude** or **norm**.

For

```math
\mathbf{x}
=
\begin{bmatrix}
3\\
4
\end{bmatrix}
```

its magnitude is

```math
\lVert\mathbf{x}\rVert
=
\sqrt{3^2+4^2}
=
5
```

This is simply Pythagoras' theorem.

---

##### Why Is Magnitude Important?

Magnitude measures the "size" of a vector.

Examples include:

- Distance travelled
- Strength of a force
- Length of a movement
- Size of an embedding

Many Machine Learning algorithms compare magnitudes before making decisions.

---

#### Unit Vectors

A **unit vector** has magnitude equal to exactly one.

Instead of representing size, it represents only direction.

For example,

```math
\begin{bmatrix}
0\\
1
\end{bmatrix}
```

points upward with unit length.

Unit vectors become extremely important in optimization and geometry.

---

### Vector Operations Summary

We have now learned four fundamental vector operations.

| Operation | Purpose |
|-----------|---------|
| Addition | Combine vectors |
| Subtraction | Find differences |
| Scalar Multiplication | Scale vectors |
| Magnitude | Measure length |

These simple operations form the vocabulary upon which the rest of Linear Algebra is built.

---

### Looking Ahead

So far, every operation has treated vectors independently.

The next question is much more interesting.

> **How similar are two vectors?**

Answering that question leads us to one of the most important ideas in all of Machine Learning:

**The Dot Product.**

---

### Dot Product

So far, we have learned how to add, subtract and scale vectors.

Now we arrive at one of the most important operations in Linear Algebra:

> **The Dot Product**

If there is one mathematical operation you should remember from this chapter, it is this one.

The dot product appears repeatedly throughout Machine Learning and Deep Learning.

---

#### What Problem Does It Solve?

Suppose we have two vectors.

Rather than asking

> "How long are they?"

or

> "Where do they point?"

we ask a different question:

> **"How similar are they?"**

The dot product gives us a numerical measure of how much two vectors point in the same direction.

---

#### Mathematical Definition

For two vectors

```math
\mathbf{a}=
\begin{bmatrix}
a_1\\
a_2\\
\vdots\\
a_n
\end{bmatrix}\,
\qquad
\mathbf{b}=
\begin{bmatrix}
b_1\\
b_2\\
\vdots\\
b_n
\end{bmatrix}
```

their dot product is

```math
\mathbf{a}\cdot\mathbf{b}
=
a_1b_1+a_2b_2+\cdots+a_nb_n
```

The process is simple.

1. Multiply corresponding elements.
2. Add the results.

The answer is **a scalar**, not another vector.

---

#### Worked Example

Suppose

```math
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
```

Then

```math
\mathbf{a}\cdot\mathbf{b}
=
2\times4
+
3\times5
=
8+15
=
23
```

Notice that two vectors produced a single number.

---

#### Geometric Interpretation

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

#### Relationship with the Angle

The dot product can also be written as

```math
\mathbf{a}\cdot\mathbf{b}
=
\lVert\mathbf{a}\rVert
\\,
\lVert\mathbf{b}\rVert
\cos\theta
```

where

- $\lVert\mathbf{a}\rVert$ is the magnitude of vector **a**
- $\lVert\mathbf{b}\rVert$ is the magnitude of vector **b**
- $\theta$ is the angle between them

This equation reveals something profound.

The dot product is largest when the vectors point in the same direction.

It becomes zero when they are perpendicular.

It becomes negative when they point in opposite directions.

---

#### Why Perpendicular Means Zero

Suppose two movements are completely independent.

Walking north does not move you east.

Walking east does not move you north.

The two movements contribute nothing to each other.

Their dot product is zero.

Vectors with a dot product of zero are called **orthogonal**.

Orthogonality is one of the most important concepts in Linear Algebra.

---

#### Machine Learning Connection

Suppose a model has learned weights

```math
\mathbf{w}
=
\begin{bmatrix}
0.5\\
1.2\\
-0.8
\end{bmatrix}
```

A new observation arrives

```math
\mathbf{x}
=
\begin{bmatrix}
170\\
68\\
25
\end{bmatrix}
```

The model computes

```math
z=\mathbf{w}^T\mathbf{x}
```

This is nothing more than a dot product.

Every neuron inside a neural network performs this calculation.

Every perceptron performs this calculation.

Linear Regression performs this calculation.

Logistic Regression performs this calculation.

The first step of modern AI is often just one very sophisticated dot product.

---

#### A Mental Model

Think of the dot product as asking

> "How strongly do these two vectors agree?"

The larger the agreement,

the larger the dot product.

The more independent they are,

the closer it moves toward zero.

The more opposite they become,

the more negative the answer becomes.

---

#### Dot Product Summary

The dot product:

- multiplies corresponding components
- produces a scalar
- measures directional similarity
- becomes zero for perpendicular vectors
- forms the mathematical heart of many Machine Learning algorithms

If vectors are the language of Machine Learning,

the dot product is one of its most frequently spoken words.

---

## 4. Matrix Operations and Transformations

### Matrix Multiplication

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

#### Matrix × Vector Multiplication

Suppose we have the vector

```math
\mathbf{x}=
\begin{bmatrix}
2\\
3
\end{bmatrix}
```

and the matrix

```math
\mathbf{A}=
\begin{bmatrix}
1 & 0\\
0 & 2
\end{bmatrix}
```

Multiplying them gives

```math
\mathbf{A}\mathbf{x}
=
\begin{bmatrix}
1 & 0\\
0 & 2
\end{bmatrix}
\begin{bmatrix}
2\\
3
\end{bmatrix}
=
\begin{bmatrix}
2\\
6
\end{bmatrix}
```

Notice what happened.

The original vector

```math
\begin{bmatrix}
2\\
3
\end{bmatrix}
```

became

```math
\begin{bmatrix}
2\\
6
\end{bmatrix}
```

The matrix stretched the second coordinate while leaving the first coordinate unchanged.

The vector has been **transformed**.

This is the real purpose of matrix multiplication.

---

#### Where Does the Row-by-Column Rule Come From?

Suppose

```math
\mathbf{A}=
\begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
```

and

```math
\mathbf{x}=
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
```

Then

```math
\mathbf{A}\mathbf{x}
=
\begin{bmatrix}
ax_1+bx_2\\
cx_1+dx_2
\end{bmatrix}
```

Look carefully at the first output.

```math
ax_1+bx_2
```

That is simply the **dot product** between

- the first row of the matrix
- the input vector

The second output is another dot product.

So matrix-vector multiplication is really nothing more than **many dot products performed together**.

Once you understand the dot product, matrix multiplication becomes much less mysterious.

---

#### Worked Example

Let

```math
\mathbf{A}=
\begin{bmatrix}
2 & 1\\
3 & 4
\end{bmatrix}
```

and

```math
\mathbf{x}=
\begin{bmatrix}
5\\
2
\end{bmatrix}
```

The first element of the output is

```math
2\times5+1\times2=12
```

The second element is

```math
3\times5+4\times2=23
```

Therefore

```math
\mathbf{A}\mathbf{x}
=
\begin{bmatrix}
12\\
23
\end{bmatrix}
```

Notice that one vector has been transformed into another vector.

---

#### Matrix × Matrix Multiplication

Matrices can also transform other matrices.

Suppose

```math
\mathbf{A}=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
```

and

```math
\mathbf{B}=
\begin{bmatrix}
5 & 6\\
7 & 8
\end{bmatrix}
```

Then

```math
\mathbf{A}\mathbf{B}
=
\begin{bmatrix}
1\times5+2\times7 & 1\times6+2\times8\\
3\times5+4\times7 & 3\times6+4\times8
\end{bmatrix}
=
\begin{bmatrix}
19 & 22\\
43 & 50
\end{bmatrix}
```

Every element is computed by taking the dot product of

- one row from the first matrix
- one column from the second matrix

---

#### Shape Compatibility

Unlike ordinary multiplication, matrices cannot always be multiplied.

The number of columns in the first matrix **must equal** the number of rows in the second matrix.

If

```math
\mathbf{A}
\text{ has shape }
(m\times n)
```

and

```math
\mathbf{B}
\text{ has shape }
(n\times p)
```

then

```math
\mathbf{A}\mathbf{B}
```

is valid and has shape

```math
(m\times p)
```

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

#### Why Order Matters

With ordinary numbers,

```math
ab=ba
```

This property is called the **commutative property**.

Matrix multiplication is different.

In general,

```math
\mathbf{A}\mathbf{B}
\neq
\mathbf{B}\mathbf{A}
```

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

#### Machine Learning Connection

Suppose a neural network receives an input vector

```math
\mathbf{x}
```

It also stores a matrix of learned weights

```math
\mathbf{W}
```

The first computation performed by the neuron is

```math
\mathbf{z}
=
\mathbf{W}\mathbf{x}
+
\mathbf{b}
```

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

#### A Better Mental Model

Many students memorize:

> "Multiply rows by columns."

While correct, that description does not explain *why* matrix multiplication exists.

Instead, remember this:

> **A matrix is a machine that accepts a vector, transforms it according to a rule, and produces a new vector.**

Once you begin thinking this way, Linear Algebra becomes much more intuitive.

---

#### Matrix Multiplication Summary

Matrix multiplication:

- transforms vectors into new vectors
- combines many dot products into one operation
- requires compatible dimensions
- is generally **not commutative**
- forms the computational backbone of Machine Learning and Deep Learning

Although the arithmetic may seem mechanical at first, its true meaning is geometric: **matrix multiplication transforms information.**

---

#### Looking Ahead

We now know that matrices transform vectors.

The next natural question is:

> **What kinds of transformations are possible?**

To answer that, we move from arithmetic to geometry and study **Linear Transformations**.

---

### Linear Transformations

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

#### What Makes a Transformation Linear?

A transformation $T$ is linear if it satisfies two rules.

For vectors $\mathbf{u}$ and $\mathbf{v}$,

```math
T(\mathbf{u}+\mathbf{v})
=
T(\mathbf{u})
+
T(\mathbf{v})
```

and for any scalar $c$,

```math
T(c\mathbf{u})
=
cT(\mathbf{u})
```

These two properties are called:

- **Additivity**
- **Homogeneity**

Together they mean that the transformation respects vector addition and scalar multiplication.

---

#### Why This Matters

Suppose

```math
\mathbf{v}
=
\mathbf{a}
+
\mathbf{b}
```

If the transformation is linear, we can transform the parts separately:

```math
T(\mathbf{v})
=
T(\mathbf{a})
+
T(\mathbf{b})
```

This gives linear transformations a predictable structure.

They do not arbitrarily warp space.

They preserve straight lines and relative structure.

---

#### A Matrix Represents a Linear Transformation

Consider

```math
\mathbf{A}
=
\begin{bmatrix}
2 & 0\\
0 & 1
\end{bmatrix}
```

and

```math
\mathbf{x}
=
\begin{bmatrix}
x\\
y
\end{bmatrix}
```

Then

```math
\mathbf{A}\mathbf{x}
=
\begin{bmatrix}
2x\\
y
\end{bmatrix}
```

This transformation doubles the horizontal component while leaving the vertical component unchanged.

Geometrically, the space has been stretched horizontally.

Every point changes according to the same rule.

---

#### Scaling

A simple scaling matrix is

```math
\mathbf{S}
=
\begin{bmatrix}
2 & 0\\
0 & 3
\end{bmatrix}
```

Applying it to

```math
\mathbf{x}
=
\begin{bmatrix}
x\\
y
\end{bmatrix}
```

gives

```math
\mathbf{S}\mathbf{x}
=
\begin{bmatrix}
2x\\
3y
\end{bmatrix}
```

The x-direction is stretched by a factor of 2.

The y-direction is stretched by a factor of 3.

---

#### Reflection

Consider

```math
\mathbf{R}
=
\begin{bmatrix}
-1 & 0\\
0 & 1
\end{bmatrix}
```

Then

```math
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
```

This reflects every point across the y-axis.

The x-coordinate changes sign.

The y-coordinate remains unchanged.

---

#### Rotation

A two-dimensional rotation can also be represented using a matrix.

For an angle $\theta$,

```math
\mathbf{R}
=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}
```

Multiplying a vector by this matrix rotates it by $\theta$.

This is a beautiful example of how geometry becomes matrix multiplication.

---

#### Shearing

A shear transformation shifts one coordinate according to another.

For example,

```math
\mathbf{H}
=
\begin{bmatrix}
1 & k\\
0 & 1
\end{bmatrix}
```

gives

```math
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
```

The y-coordinate remains unchanged, while the x-coordinate shifts according to $y$.

A square can become a slanted parallelogram.

---

#### The Origin Stays Fixed

A true linear transformation always maps the zero vector to the zero vector.

That is,

```math
T(\mathbf{0})
=
\mathbf{0}
```

This property is important.

If a transformation shifts every point by some constant amount, it is not strictly linear.

For example,

```math
\mathbf{y}
=
\mathbf{A}\mathbf{x}
+
\mathbf{b}
```

contains a translation due to $\mathbf{b}$.

Mathematically, this is called an **affine transformation**, not a purely linear transformation.

---

#### Machine Learning Connection

This distinction appears constantly in Machine Learning.

A neuron computes

```math
\mathbf{z}
=
\mathbf{W}\mathbf{x}
+
\mathbf{b}
```

The multiplication

```math
\mathbf{W}\mathbf{x}
```

is a linear transformation.

The addition of

```math
\mathbf{b}
```

shifts the result.

So the complete operation is technically **affine**.

In Machine Learning, however, people often casually refer to this entire step as a "linear layer."

That terminology is convenient, but mathematically the presence of bias makes it affine.

---

#### Why Activation Functions Are Necessary

Suppose a neural network contains several layers but no nonlinear activation functions.

Then we may have

```math
\mathbf{W}_3
\mathbf{W}_2
\mathbf{W}_1
\mathbf{x}
```

Because multiplying matrices together produces another matrix, this becomes

```math
\mathbf{W}
\mathbf{x}
```

for some combined matrix $\mathbf{W}$.

So stacking many purely linear transformations still produces only another linear transformation.

The network would gain depth but not greater expressive power.

This is why activation functions are essential.

They introduce **nonlinearity**.

That nonlinear step prevents the entire network from collapsing mathematically into one equivalent matrix multiplication.

---

#### A Powerful Mental Model

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

#### Linear Transformations Summary

A linear transformation:

- maps vectors to vectors
- preserves vector addition
- preserves scalar multiplication
- can be represented using a matrix
- can scale, rotate, reflect or shear space
- always maps the origin to the origin

When bias is added,

```math
\mathbf{W}\mathbf{x}+\mathbf{b}
```

the transformation becomes affine.

Neural Networks rely on repeated affine transformations combined with nonlinear activation functions.

---

#### Looking Ahead

Matrices transform vectors.

But one of the oldest uses of matrices is solving equations.

Suppose several unknown quantities must satisfy several relationships simultaneously.

That problem leads directly to **Systems of Linear Equations**.

---

## 5. Solving Linear Systems

### Systems of Linear Equations

A system of linear equations contains several equations that must all be satisfied at the same time.

For example,

```math
x+y=5
```

and

```math
2x-y=1
```

We are looking for values of $x$ and $y$ that satisfy both equations simultaneously.

---

#### Solving the Simple Way

From

```math
x+y=5
```

we get

```math
y=5-x
```

Substitute this into

```math
2x-y=1
```

to obtain

```math
2x-(5-x)=1
```

Therefore,

```math
3x=6
```

so

```math
x=2
```

and therefore,

```math
y=3
```

The solution is

```math
\begin{bmatrix}
2\\
3
\end{bmatrix}
```

---

#### Matrix Form

The same system can be written compactly as

```math
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
```

or more generally,

```math
\mathbf{A}\mathbf{x}
=
\mathbf{b}
```

where

- $\mathbf{A}$ contains the coefficients
- $\mathbf{x}$ contains the unknowns
- $\mathbf{b}$ contains the results

This compact equation is one of the most important forms in Linear Algebra.

---

#### Geometric Interpretation

Each linear equation describes a line.

For example,

```math
x+y=5
```

describes one line.

And

```math
2x-y=1
```

describes another.

The solution occurs where the two lines intersect.

So solving a system of equations means finding the point that satisfies all constraints at once.

---

#### Three Possible Outcomes

Two lines may behave in three different ways.

##### 1. One Unique Solution

The lines intersect at exactly one point.

The system has one solution.

---

##### 2. No Solution

The lines are parallel.

They never meet.

The system is inconsistent.

---

##### 3. Infinitely Many Solutions

The two equations describe the same line.

Every point on that line satisfies both equations.

---

#### Why This Matters in Machine Learning

Many Machine Learning problems eventually produce equations of the form

```math
\mathbf{A}\mathbf{x}
=
\mathbf{b}
```

For example, Linear Regression can be expressed using matrix equations.

In its closed-form solution,

```math
\hat{\boldsymbol{\beta}}
=
(\mathbf{X}^T\mathbf{X})^{-1}
\mathbf{X}^T\mathbf{y}
```

the model is solving for the coefficient vector

```math
\hat{\boldsymbol{\beta}}
```

using matrix operations.

The familiar regression problem is therefore deeply connected to solving systems of linear equations.

---

#### Exact Solutions vs Approximate Solutions

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

#### Overdetermined Systems

Suppose we have many equations but only a few unknowns.

For example,

100 observations but only 3 model parameters.

This creates an **overdetermined system**.

There are more constraints than unknowns.

Usually no exact solution exists.

Linear Regression therefore finds the parameter values that minimize the overall error.

---

#### Underdetermined Systems

The opposite can also happen.

Suppose we have fewer equations than unknowns.

Then multiple solutions may satisfy the system.

This is called an **underdetermined system**.

Modern Machine Learning frequently operates in very high-dimensional spaces, so understanding this situation becomes increasingly important.

---

#### Rank — A First Intuition

The idea of **rank** tells us how much independent information a matrix contains.

If several rows or columns carry redundant information, the effective dimensionality of the matrix is smaller than its apparent size.

For example,

```math
\begin{bmatrix}
1 & 2\\
2 & 4
\end{bmatrix}
```

looks like a $2\times2$ matrix.

But the second row is simply twice the first.

So it does not contain completely new information.

This matrix has lower rank.

We will revisit this idea when discussing:

- linear independence
- basis
- eigenvalues
- dimensionality reduction

---

#### Machine Learning Connection

Systems of equations appear behind many algorithms.

Examples include:

- Linear Regression
- Least Squares
- PCA
- Optimization
- Matrix Factorization
- Recommendation Systems

The notation

```math
\mathbf{A}\mathbf{x}
=
\mathbf{b}
```

is therefore not merely a classroom exercise.

It is one of the fundamental patterns of applied Machine Learning.

---

#### Linear Systems Summary

A system of linear equations:

- contains multiple relationships that must hold simultaneously
- can be written compactly as

```math
\mathbf{A}\mathbf{x}
=
\mathbf{b}
```

- may have one solution, no solution or infinitely many solutions
- can be interpreted geometrically as intersections
- forms the mathematical basis of least squares and Linear Regression

Real-world Machine Learning frequently cannot solve systems exactly.

Instead, it searches for the **best approximate solution**.

---

#### Looking Ahead

We have now seen vectors, matrices, transformations and systems of equations.

The next question is more structural:

> **How can a small collection of vectors describe an entire space?**

That leads us to three closely related ideas:

- Span
- Basis
- Linear Independence

---

## 6. Vector Spaces and Their Structure

### Span and Linear Combinations

We now know that vectors can be added and scaled.

That gives us a powerful idea:

> **A small collection of vectors can be combined to generate many other vectors.**

The set of all vectors that can be produced from such combinations is called the **span**.

---

#### Linear Combinations

Suppose we have two vectors

```math
\mathbf{v}_1=
\begin{bmatrix}
1\\
0
\end{bmatrix}
```

and

```math
\mathbf{v}_2=
\begin{bmatrix}
0\\
1
\end{bmatrix}
```

We may multiply each vector by any scalar and add the results:

```math
a\mathbf{v}_1+b\mathbf{v}_2
```

where $a$ and $b$ are scalars.

This is called a **linear combination**.

For example,

```math
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
```

By choosing different values of $a$ and $b$, we can generate every point in the two-dimensional plane.

Therefore these two vectors **span** the plane.

---

#### Geometric Intuition for Span

Consider just one non-zero vector:

```math
\mathbf{v}=
\begin{bmatrix}
1\\
2
\end{bmatrix}
```

Multiplying it by different scalars gives

```math
\ldots,-2\mathbf{v},-\mathbf{v},0,\mathbf{v},2\mathbf{v},3\mathbf{v},\ldots
```

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

#### When Vectors Fail to Add a New Direction

Suppose

```math
\mathbf{v}_1=
\begin{bmatrix}
1\\
2
\end{bmatrix}
```

and

```math
\mathbf{v}_2=
\begin{bmatrix}
2\\
4
\end{bmatrix}
```

The second vector is simply

```math
\mathbf{v}_2=2\mathbf{v}_1
```

Although we have two vectors, they point along the same direction.

Their linear combinations still generate only a line.

The second vector contributes no new direction.

This leads naturally to **linear independence**.

---

### Linear Independence

A collection of vectors is **linearly independent** if none of the vectors can be constructed from the others.

Each vector contributes genuinely new information or a new direction.

For example,

```math
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
```

are linearly independent.

Neither can be produced by scaling the other.

Together they introduce two distinct directions.

---

#### Linear Dependence

Now consider

```math
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
```

Since

```math
\mathbf{v}_2=2\mathbf{v}_1
```

the second vector does not introduce any new direction.

The vectors are therefore **linearly dependent**.

One contains redundant information.

---

#### The Formal Test

Vectors

```math
\mathbf{v}_1,\mathbf{v}_2,\ldots,\mathbf{v}_n
```

are linearly independent if the equation

```math
c_1\mathbf{v}_1+c_2\mathbf{v}_2+\cdots+c_n\mathbf{v}_n
=
\mathbf{0}
```

has only the trivial solution

```math
c_1=c_2=\cdots=c_n=0
```

If there is some non-zero combination of coefficients that produces the zero vector, the vectors are linearly dependent.

The formal definition may look abstract, but the intuition remains simple:

> **Independent vectors each contribute something new. Dependent vectors contain redundancy.**

---

#### Machine Learning Connection — Redundant Features

Suppose a dataset contains these features:

- Height in centimetres
- Height in metres
- Weight

The first two features contain the same information because

```math
\text{height in metres}
=
0.01\times\text{height in centimetres}
```

One feature is an exact linear combination of another.

This creates **linear dependence**.

In regression models, strong dependence between predictor variables is related to **multicollinearity**.

Redundant features can make model coefficients unstable and matrix calculations difficult.

So linear independence is not merely a geometric curiosity—it has direct practical importance in Machine Learning.

---

### Basis and Dimension

We are now ready for one of the central ideas in Linear Algebra.

A **basis** is a minimal set of linearly independent vectors that spans a space.

Two conditions must therefore hold:

1. The vectors must span the entire space.
2. The vectors must be linearly independent.

A basis contains exactly enough directions to describe every vector in the space—no fewer and no redundant extras.

---

#### The Standard Basis

In two dimensions, the most familiar basis is

```math
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
```

These are called the **standard basis vectors**.

Any vector

```math
\mathbf{x}=
\begin{bmatrix}
x\\
y
\end{bmatrix}
```

can be written as

```math
\mathbf{x}=x\mathbf{e}_1+y\mathbf{e}_2
```

For example,

```math
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
```

The numbers 3 and 2 are the coordinates of the vector relative to this basis.

---

#### A Basis Is Not Unique

The standard basis is convenient, but it is not the only possible basis.

For example,

```math
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
```

also form a basis for the two-dimensional plane.

They are linearly independent and together span the entire plane.

So the same vector can be described using different coordinate systems depending on the chosen basis.

This idea becomes extremely important later.

---

#### Changing the Basis Changes the Description, Not the Vector

Imagine describing the location of a building.

One person may describe it using north-south and east-west directions.

Another may use roads running diagonally across the city.

The building itself has not moved.

Only the coordinate system used to describe it has changed.

A change of basis works in the same way.

> **The underlying information stays the same; only its representation changes.**

This is a powerful idea in Machine Learning because useful representations can make difficult problems much easier.

---

#### Machine Learning Connection — PCA

Principal Component Analysis (PCA) can be understood partly as finding a more useful coordinate system for the data.

Instead of describing data using the original feature directions, PCA finds new directions that capture the greatest variation.

The data is then represented relative to those new directions.

In other words, PCA effectively changes the basis to one that is better aligned with the structure of the dataset.

We will revisit this when discussing eigenvectors.

---

#### Dimension

The **dimension** of a vector space is the number of vectors required in any basis for that space.

For example:

- A line has dimension 1.
- A plane has dimension 2.
- Ordinary 3D space has dimension 3.

A dataset containing 100 features is naturally represented in a 100-dimensional feature space.

But its true information content may occupy fewer than 100 independent directions.

This distinction becomes very important in dimensionality reduction.

---

### Rank

Earlier, we introduced rank as the amount of independent information contained in a matrix.

We can now state this more precisely.

The **rank of a matrix** is the number of linearly independent directions represented by its rows or columns.

Consider

```math
\mathbf{A}=
\begin{bmatrix}
1 & 2\\
2 & 4
\end{bmatrix}
```

The second row is twice the first.

Therefore only one row contributes genuinely new information.

The matrix has rank 1.

Now consider

```math
\mathbf{B}=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
```

Its rows point in independent directions.

The matrix has rank 2.

---

#### Full Rank

A matrix is called **full rank** when it contains the maximum possible number of independent rows or columns.

For a square $n\times n$ matrix, full rank means

```math
\mathrm{rank}(\mathbf{A})=n
```

Full-rank matrices behave particularly well.

For example, a square full-rank matrix has an inverse.

A rank-deficient matrix does not.

This fact becomes important when solving systems of equations and when computing regression coefficients.

---

#### Rank and Redundancy

A useful mental model is:

```text
Matrix size  → how much space is available
Matrix rank  → how much independent information is actually present
```

A matrix may have hundreds of columns but still have much lower rank if many features are redundant or correlated.

This is one reason dimensionality reduction can work.

The data may live in a large feature space while occupying a much smaller effective subspace.

---

### Connecting the Ideas

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

#### Machine Learning Connection

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

### Vector Spaces Summary

A **linear combination** is formed by scaling vectors and adding them.

The **span** is the set of all vectors obtainable from those combinations.

Vectors are **linearly independent** when none can be constructed from the others.

A **basis** is a minimal independent set that spans the entire space.

The **dimension** tells us how many vectors a basis contains.

The **rank** of a matrix tells us how many independent directions or pieces of information the matrix actually contains.

These concepts give us the structural language required to understand high-dimensional data.

---

### Looking Ahead

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

## 7. Orthogonality, Projections, and Distance

### Orthogonality

One of the most useful geometric relationships between vectors is **orthogonality**.

Two vectors are orthogonal if they meet at a right angle (90°).

Mathematically,

```math
\mathbf{a}\cdot\mathbf{b}=0
```

provided neither vector is the zero vector.

This follows directly from the dot product equation.

Since

```math
\mathbf{a}\cdot\mathbf{b}
=
\|\mathbf{a}\|
\,
\|\mathbf{b}\|
\cos\theta
```

when

```math
\theta=90^\circ
```

we have

```math
\cos90^\circ=0
```

Therefore,

```math
\mathbf{a}\cdot\mathbf{b}=0
```

---

#### Why Orthogonality Matters

Orthogonal vectors are completely independent directions.

Imagine walking north.

Now imagine walking east.

Walking north contributes nothing to your eastward movement.

Likewise, walking east contributes nothing to your northward movement.

The two directions do not interfere with each other.

Orthogonal vectors behave exactly this way.

---

#### Example

Consider

```math
\mathbf{a}
=
\begin{bmatrix}
1\\
0
\end{bmatrix}
```

and

```math
\mathbf{b}
=
\begin{bmatrix}
0\\
5
\end{bmatrix}
```

Their dot product is

```math
1\times0+0\times5=0
```

Therefore the vectors are orthogonal.

Notice that their lengths are irrelevant.

Only the angle matters.

---

#### Orthonormal Vectors

Sometimes vectors are not only perpendicular but also have unit length.

Such vectors are called **orthonormal**.

An orthonormal set satisfies two conditions.

1. Every vector has length 1.
2. Every pair of vectors is orthogonal.

The standard basis vectors

```math
\mathbf{e}_1
=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
\mathbf{e}_2
=
\begin{bmatrix}
0\\
1
\end{bmatrix}
```

form an orthonormal basis.

Orthonormal bases greatly simplify many calculations.

---

### Projection

Suppose someone shines a flashlight onto the floor.

The shadow represents only the part of an object lying in a particular direction.

Projection works in much the same way.

A **projection** measures how much of one vector lies along another.

---

#### Intuition

Suppose

- one vector represents the direction you are walking
- another represents the direction of a strong wind

The wind only helps you if part of it blows in your direction.

Projection measures exactly that useful component.

---

#### Scalar Projection

The scalar projection of

```math
\mathbf{a}
```

onto

```math
\mathbf{b}
```

is

```math
\frac{\mathbf{a}\cdot\mathbf{b}}
{\|\mathbf{b}\|}
```

It answers the question

> **"How much of vector a lies along vector b?"**

The result is a scalar.

---

#### Vector Projection

Sometimes we want the projected vector itself.

The vector projection is

```math
\mathrm{proj}_{\mathbf{b}}(\mathbf{a})
=
\frac{\mathbf{a}\cdot\mathbf{b}}
{\mathbf{b}\cdot\mathbf{b}}
\mathbf{b}
```

The result is another vector.

It points in the direction of

```math
\mathbf{b}
```

and has exactly the correct length.

---

#### Why Projection Matters

Projection appears everywhere.

Examples include:

- Least Squares
- Linear Regression
- PCA
- Computer Graphics
- Signal Processing

Many optimization algorithms repeatedly project vectors into different subspaces.

---

### Vector Norms and Distance

Earlier we introduced magnitude.

In Linear Algebra, magnitude is usually called a **norm**.

A norm measures the size of a vector.

Different applications use different norms.

---

#### L2 Norm (Euclidean Norm)

The most familiar norm is the Euclidean norm.

For

```math
\mathbf{x}
=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_n
\end{bmatrix}
```

the L2 norm is

```math
\|\mathbf{x}\|_2
=
\sqrt{x_1^2+x_2^2+\cdots+x_n^2}
```

This is simply the ordinary geometric length.

---

#### L1 Norm

Another important norm is the L1 norm.

```math
\|\mathbf{x}\|_1
=
|x_1|
+
|x_2|
+
\cdots
+
|x_n|
```

Instead of measuring straight-line distance, it sums the absolute values.

It is often called the **Manhattan distance** because it resembles travelling through city streets arranged in a grid.

---

#### L∞ Norm

The infinity norm measures only the largest absolute component.

```math
\|\mathbf{x}\|_\infty
=
\max_i |x_i|
```

It ignores every other coordinate.

---

#### Why Different Norms Exist

Different notions of "distance" make sense in different problems.

Suppose two people live across a park.

A bird flies directly.

A pedestrian follows the roads.

Both travel between the same locations.

But they measure distance differently.

Norms formalize these different notions of distance.

---

#### Distance Between Vectors

Distance is simply the norm of their difference.

For vectors

```math
\mathbf{a}
```

and

```math
\mathbf{b}
```

the Euclidean distance is

```math
d(\mathbf{a},\mathbf{b})
=
\|\mathbf{a}-\mathbf{b}\|
```

Notice how vector subtraction and norms combine naturally.

---

#### Machine Learning Connection — K-Nearest Neighbours

K-Nearest Neighbours (KNN) predicts by finding observations closest to a query point.

But what does "closest" mean?

Usually,

it means the smallest distance according to a chosen norm.

Most commonly,

```math
L_2
```

distance is used.

Sometimes

```math
L_1
```

distance performs better.

The choice of norm can therefore change the model's predictions.

---

#### Machine Learning Connection — Regularization

Norms also appear inside loss functions.

For example,

Lasso Regression minimizes an L1 penalty.

Ridge Regression minimizes an L2 penalty.

We already studied these algorithms in Part I.

Now we can finally understand why those penalties are called

> **L1 Regularization**

and

> **L2 Regularization**

They literally penalize the corresponding vector norms of the parameter vector.

---

#### Machine Learning Connection — Cosine Similarity

Earlier we saw that the dot product measures alignment.

Sometimes magnitude is not important.

Suppose two documents use identical words, but one document is much longer.

We want to compare their direction, not their length.

Cosine similarity normalizes the dot product.

```math
\cos\theta
=
\frac{\mathbf{a}\cdot\mathbf{b}}
{\|\mathbf{a}\|
\,
\|\mathbf{b}\|}
```

Notice that this is simply the dot product divided by both magnitudes.

Cosine similarity therefore measures orientation instead of size.

This metric is heavily used in:

- Information Retrieval
- Search Engines
- NLP
- Embeddings
- Large Language Models

---

### Bringing Everything Together

Notice how several ideas now connect.

```text
Vector Subtraction
        │
        ▼
Distance
        │
        ▼
Nearest Neighbour Algorithms

Dot Product
        │
        ▼
Projection
        │
        ▼
Cosine Similarity
        │
        ▼
Embeddings

Norms
        │
        ▼
Regularization
```

What originally appeared to be separate topics are actually pieces of one larger picture.

---

### Geometry and Distance Summary

Orthogonality describes perpendicular directions.

Projection extracts the component of one vector along another.

Norms measure vector size.

Distance measures how far vectors are from each other.

These ideas underpin similarity measures, optimization, dimensionality reduction and many Machine Learning algorithms.

---

### Looking Ahead

There is one final question.

Some directions inside a transformation are special.

Instead of changing direction,

they merely stretch or shrink.

Those remarkable directions lead us to one of the most beautiful concepts in Linear Algebra:

> **Eigenvalues and Eigenvectors**

---
## 8. Spectral Methods and Matrix Decomposition

### Eigenvalues and Eigenvectors

We have seen that matrices transform vectors.

Usually, a transformation changes both

- the length of a vector
- its direction

Imagine grabbing a rubber sheet with a grid drawn on it.

Now stretch and twist the sheet.

Almost every arrow drawn on the sheet changes both its length and its direction.

But something remarkable happens.

A few very special arrows do **not rotate at all**.

They may become longer.

They may become shorter.

They may even reverse direction.

But they continue pointing along exactly the same line.

Those special vectors are called **eigenvectors**.

---

#### A Visual Intuition

Imagine a square piece of rubber.

You stretch it twice as much horizontally as vertically.

Every arrow drawn on the square changes.

Most arrows rotate while stretching.

However,

an arrow pointing exactly along the horizontal axis does not rotate.

It simply becomes longer.

Likewise,

an arrow pointing exactly along the vertical axis keeps its direction.

Only its length changes.

These arrows are eigenvectors.

They are the "preferred directions" of the transformation.

---

#### The Big Idea

An eigenvector is a vector whose direction remains unchanged after a linear transformation.

Only its magnitude changes.

The amount of stretching or shrinking is called the **eigenvalue**.

Therefore,

> **Eigenvectors tell us the important directions.**

> **Eigenvalues tell us how strongly those directions are stretched or compressed.**

---

#### The Mathematical Definition

Suppose

```math
\mathbf{A}
```

is a square matrix.

A vector

```math
\mathbf{v}
```

is called an eigenvector if

```math
\mathbf{A}\mathbf{v}
=
\lambda\mathbf{v}
```

where

```math
\lambda
```

is called the **eigenvalue**.

This equation says something surprisingly simple.

After applying the matrix,

the vector still points in exactly the same direction.

Only its length changes by a factor of

```math
\lambda
```

---

#### Understanding the Equation

Suppose

```math
\lambda=3
```

Then

```math
\mathbf{A}\mathbf{v}
=
3\mathbf{v}
```

The vector becomes three times longer.

Its direction stays the same.

---

Suppose

```math
\lambda=\frac12
```

Then

```math
\mathbf{A}\mathbf{v}
=
\frac12\mathbf{v}
```

The vector shrinks to half its length.

Again,

its direction is unchanged.

---

Suppose

```math
\lambda=-2
```

Now the vector becomes twice as long,

but also reverses direction.

Negative eigenvalues flip vectors.

---

Suppose

```math
\lambda=1
```

The vector remains exactly the same.

Neither its direction nor its length changes.

---

Suppose

```math
\lambda=0
```

The transformation completely collapses that direction.

The vector disappears to the origin.

---

#### A Simple Example

Consider

```math
\mathbf{A}
=
\begin{bmatrix}
2 & 0\\
0 & 1
\end{bmatrix}
```

Take the vector

```math
\mathbf{v}
=
\begin{bmatrix}
1\\
0
\end{bmatrix}
```

Then

```math
\mathbf{A}\mathbf{v}
=
\begin{bmatrix}
2\\
0
\end{bmatrix}
=
2
\begin{bmatrix}
1\\
0
\end{bmatrix}
```

So

```math
\lambda=2
```

The direction did not change.

Only the length doubled.

Now try

```math
\mathbf{v}
=
\begin{bmatrix}
0\\
1
\end{bmatrix}
```

Then

```math
\mathbf{A}\mathbf{v}
=
\begin{bmatrix}
0\\
1
\end{bmatrix}
=
1
\begin{bmatrix}
0\\
1
\end{bmatrix}
```

So this is another eigenvector.

Its eigenvalue is

```math
1
```

---

#### Why Only Certain Directions?

Most vectors are mixtures of several directions.

When transformed,

their components stretch by different amounts.

As a result,

their overall direction changes.

Eigenvectors are special because they already lie along the natural stretching directions of the transformation.

They require no "mixing."

---

#### Machine Learning Connection — Principal Component Analysis

Principal Component Analysis (PCA) searches for the directions in which data varies the most.

Those directions are precisely the eigenvectors of the covariance matrix.

The corresponding eigenvalues tell us how much variance exists along each direction.

Large eigenvalue

↓

Important direction

Small eigenvalue

↓

Less important direction

PCA keeps the important directions and often discards the rest.

This allows high-dimensional data to be represented using far fewer dimensions.

---

#### Machine Learning Connection — Google PageRank

Google's original PageRank algorithm is fundamentally an eigenvector problem.

Each web page receives importance from other important pages.

After repeated updates,

the importance scores settle into a stable pattern.

That stable pattern is an eigenvector.

The associated eigenvalue is

```math
1
```

Although modern search engines are much more sophisticated, the original PageRank algorithm introduced millions of engineers to eigenvectors.

---

#### Machine Learning Connection — Stability

Eigenvalues often tell us whether a system is stable.

If repeated transformations continually increase vector lengths,

the system may become unstable.

If vectors shrink toward zero,

the system becomes stable.

Eigenvalues therefore appear in:

- Optimization
- Control Systems
- Reinforcement Learning
- Dynamical Systems

---

#### A Mental Model

Imagine every matrix asking the question:

> "Which directions do I naturally like to stretch?"

The answers are its eigenvectors.

Then ask:

> "By how much do I stretch each of those directions?"

The answers are the corresponding eigenvalues.

Once you think this way,

the equation

```math
\mathbf{A}\mathbf{v}
=
\lambda\mathbf{v}
```

becomes almost obvious.

It simply formalizes that idea.

---

#### Eigenvalues and Eigenvectors Summary

An eigenvector is a direction that remains unchanged by a transformation.

An eigenvalue tells us how much that direction is stretched or compressed.

Together they reveal the hidden geometric structure of a transformation.

They are fundamental to:

- PCA
- Spectral Clustering
- PageRank
- Stability Analysis
- Computer Vision
- Quantum Mechanics

Among many other fields.

---

#### Looking Ahead

Eigenvalues reveal the important directions of a transformation.

But many real-world datasets are noisy, incomplete or rectangular rather than square.

To analyse those situations, we need an even more powerful mathematical tool:

> **Singular Value Decomposition (SVD)**

---

### Singular Value Decomposition (SVD)

Throughout this chapter we have repeatedly asked the same question:

> **How does a matrix transform information?**

Eigenvalues answered part of that question.

But they have one important limitation.

They are defined only for **square matrices**.

Real Machine Learning datasets are rarely square.

A dataset with

- 10,000 observations
- 200 features

is represented by a

```math
10000\times200
```

matrix.

Eigenvectors alone are no longer sufficient.

This is where **Singular Value Decomposition (SVD)** becomes one of the most powerful tools in Linear Algebra.

---

#### The Big Idea

Every matrix, regardless of its shape, can be decomposed into three simpler matrices.

Mathematically,

```math
\mathbf{A}
=
\mathbf{U}
\mathbf{\Sigma}
\mathbf{V}^T
```

This equation may look intimidating.

Its interpretation is surprisingly intuitive.

Imagine transforming a rubber sheet.

Instead of performing one complicated transformation,

SVD says we can think of it as three simpler steps.

1. Rotate the space.
2. Stretch or compress along special directions.
3. Rotate again.

Complex transformations become combinations of simple ones.

---

#### Understanding the Three Matrices

The decomposition

```math
\mathbf{A}
=
\mathbf{U}
\mathbf{\Sigma}
\mathbf{V}^T
```

contains three parts.

---

##### 1. Vᵀ — Rotate into a Convenient Coordinate System

The first transformation changes our viewpoint.

Instead of working in the original coordinate system,

we rotate into one where the important directions become easier to describe.

Nothing has been stretched yet.

Only the perspective changes.

---

##### 2. Σ — Stretch Along Independent Directions

The middle matrix

```math
\mathbf{\Sigma}
```

contains non-negative numbers called **singular values**.

These numbers tell us how much stretching or shrinking occurs along each important direction.

Large singular values correspond to important directions.

Small singular values correspond to directions containing little information.

---

##### 3. U — Rotate into the Final Orientation

Finally,

the transformed space is rotated into its final orientation.

So the entire transformation becomes

```text
Rotate
   ↓
Stretch
   ↓
Rotate
```

A remarkably complicated matrix can therefore be understood using only rotations and scaling.

---

#### Singular Values

The diagonal entries of

```math
\mathbf{\Sigma}
```

are called **singular values**.

They play a role similar to eigenvalues.

Large singular values indicate important directions.

Very small singular values often correspond to:

- noise
- redundancy
- weak patterns

This observation is the basis of many dimensionality reduction techniques.

---

#### Why SVD Matters

Suppose we have a huge dataset.

Much of its information may actually lie in only a handful of important directions.

Instead of storing everything,

we can keep only the largest singular values and discard the rest.

Surprisingly,

very little useful information is lost.

This idea appears repeatedly throughout Machine Learning.

---

#### A Picture Worth Remembering

Imagine listening to an orchestra.

Hundreds of instruments play simultaneously.

Suppose you could somehow identify the few instruments carrying almost all the melody.

You could remove many background sounds while preserving the music.

SVD performs something similar for data.

It separates important structure from less important detail.

---

#### Machine Learning Connection — Dimensionality Reduction

Many datasets contain hundreds or thousands of features.

Often,

those features are highly correlated.

SVD discovers the underlying independent directions.

This allows us to represent the same dataset using far fewer dimensions.

Smaller representations are:

- faster
- cheaper
- easier to visualize
- often less noisy

---

#### Machine Learning Connection — PCA

Earlier we learned that PCA uses eigenvectors of the covariance matrix.

There is another way to compute PCA.

Modern software libraries usually perform PCA using **SVD**.

Why?

Because SVD is:

- numerically stable
- computationally efficient
- applicable to rectangular datasets

This is why many implementations of PCA never explicitly compute eigenvectors.

Instead,

they perform SVD under the hood.

---

#### Machine Learning Connection — Recommendation Systems

Suppose millions of users rate millions of movies.

The ratings matrix is enormous.

Yet many users have similar preferences.

SVD helps discover hidden patterns.

Instead of memorizing every rating,

the system learns a much smaller set of latent factors.

These factors capture concepts such as:

- action preference
- romance preference
- comedy preference

even though those labels were never explicitly provided.

This idea became famous through the Netflix Prize competition.

---

#### Machine Learning Connection — Natural Language Processing

Modern NLP often represents documents using huge matrices.

SVD can discover hidden semantic structure.

Classical techniques such as **Latent Semantic Analysis (LSA)** rely directly on SVD.

Even though today's Large Language Models use Transformers,

many foundational ideas about embeddings and low-dimensional representations originate from techniques like SVD.

---

#### Low-Rank Approximation

One of SVD's greatest strengths is its ability to approximate a matrix.

Instead of keeping every singular value,

we retain only the largest ones.

This produces a **low-rank approximation**.

The approximation requires:

- less memory
- fewer computations

while preserving most of the useful information.

This is an important technique in data compression and model acceleration.

---

#### Bringing Everything Together

Notice the progression throughout this chapter.

```text
Scalars
      ↓
Vectors
      ↓
Matrices
      ↓
Transformations
      ↓
Eigenvectors
      ↓
SVD
```

Each concept built naturally upon the previous one.

SVD is not an isolated algorithm.

It is the culmination of everything we have learned about matrices and transformations.

---

#### SVD Summary

Singular Value Decomposition factors any matrix into three simpler matrices.

Conceptually,

it performs:

```text
Rotate
   ↓
Stretch
   ↓
Rotate
```

The singular values reveal the importance of different directions within the data.

SVD underpins:

- PCA
- LSA
- Recommendation Systems
- Low-rank Approximation
- Data Compression
- Dimensionality Reduction

It is one of the most important tools in modern applied Linear Algebra.

---

### Looking Ahead

We have now completed the mathematical foundations of Linear Algebra.

Before moving on to Calculus,

let us step back and ask one final question:

> **Where exactly did every concept from this chapter appear inside Machine Learning?**

That connection is the perfect way to conclude this chapter.

---

## 9. Linear Algebra in Machine Learning

### An ML Engineering Perspective

At the beginning of this chapter, we said:

> *"Linear Algebra is the mathematics of representing and transforming information."*

By now, that statement should feel much more concrete.

Almost every Machine Learning algorithm can be viewed as repeatedly representing data as vectors and transforming those vectors in useful ways.

What once appeared to be a collection of unrelated mathematical topics is actually one connected story.

---

### The Journey So Far

The chapter followed a natural progression.

```text
Scalars
      ↓
Vectors
      ↓
Matrices
      ↓
Vector Operations
      ↓
Dot Product
      ↓
Matrix Multiplication
      ↓
Linear Transformations
      ↓
Systems of Equations
      ↓
Span & Basis
      ↓
Orthogonality
      ↓
Norms & Distance
      ↓
Eigenvectors
      ↓
Singular Value Decomposition
```

Each idea depends on the ones before it.

Nothing was introduced in isolation.

---

### Where Does Each Concept Appear?

The following table summarizes where these ideas appear in Machine Learning.

| Linear Algebra Concept | Machine Learning Applications |
|-------------------------|-------------------------------|
| Scalars | Learning rate, regularization strength, probabilities, loss values |
| Vectors | Feature vectors, embeddings, model parameters |
| Matrices | Datasets, weight matrices, covariance matrices |
| Vector Addition | Parameter updates, feature aggregation |
| Scalar Multiplication | Learning rates, feature scaling |
| Dot Product | Linear Regression, Logistic Regression, Perceptrons, Neural Networks |
| Matrix Multiplication | Neural Networks, Transformers, PCA |
| Linear Transformations | Feature engineering, hidden layers, embeddings |
| Systems of Equations | Least Squares, Linear Regression |
| Span & Basis | Feature spaces, coordinate systems |
| Linear Independence | Multicollinearity, redundant features |
| Rank | Matrix invertibility, dimensionality reduction |
| Orthogonality | PCA, optimization, QR decomposition |
| Projection | Least Squares, PCA, optimization |
| Norms | Distance metrics, regularization |
| Eigenvalues & Eigenvectors | PCA, PageRank, spectral methods |
| SVD | PCA, recommendation systems, LSA, compression |

This table is worth revisiting throughout your Machine Learning journey.

---

### Classical Machine Learning

Think back to the models studied in Part I.

#### Linear Regression

Linear Regression computes

```math
\hat{y}
=
\mathbf{w}^T\mathbf{x}+b
```

Everything here is Linear Algebra.

- vectors
- dot products
- matrices
- least squares
- projections

---

#### Logistic Regression

Logistic Regression begins exactly the same way.

It first computes

```math
z
=
\mathbf{w}^T\mathbf{x}+b
```

Only after that does it apply the sigmoid function.

The underlying computation is still Linear Algebra.

---

#### K-Nearest Neighbours

KNN asks only one question.

> Which observations are closest?

To answer that question it computes distances between vectors.

Distance is simply

```math
\|\mathbf{x}-\mathbf{y}\|
```

---

#### Decision Trees

Decision Trees depend much less on Linear Algebra.

They split data using decision rules rather than vector transformations.

This explains why tree-based models often require less feature scaling than linear models.

---

#### Support Vector Machines

Support Vector Machines rely heavily on:

- vectors
- dot products
- projections
- distances

Kernel methods also build upon these same ideas.

---

#### Principal Component Analysis

PCA combines nearly everything from this chapter.

It uses:

- covariance matrices
- eigenvectors
- eigenvalues
- orthogonal directions
- projections

PCA is almost a celebration of Linear Algebra.

---

### Deep Learning

Deep Learning looks intimidating because of its enormous scale.

The mathematics, however, remains surprisingly familiar.

Each layer performs

```math
\mathbf{z}
=
\mathbf{W}\mathbf{x}
+
\mathbf{b}
```

followed by a nonlinear activation.

Every hidden layer repeats the same pattern.

```text
Input Vector
        ↓
Matrix Multiplication
        ↓
Bias Addition
        ↓
Activation Function
        ↓
Repeat
```

Deep Learning is therefore not a different branch of mathematics.

It is Linear Algebra repeated many times.

---

### Transformers

Transformers appear enormously complex.

Yet at their core they repeatedly perform:

- matrix multiplication
- dot products
- projections
- vector normalization
- similarity calculations

Even attention begins with a dot product.

The famous attention equation

```math
QK^T
```

is simply a large collection of dot products.

The mathematics you learned in this chapter therefore scales all the way to today's largest language models.

---

### One Unifying Mental Model

Whenever you encounter a new Machine Learning algorithm, ask yourself four questions.

1. **How is the data represented?**
   (Usually as vectors or matrices.)

2. **How is the data transformed?**
   (Usually using matrix multiplication.)

3. **How is similarity measured?**
   (Usually using dot products or distances.)

4. **How is the best representation discovered?**
   (Usually using optimization together with Linear Algebra.)

Remarkably, these four questions explain a large fraction of modern Machine Learning.

---

### Key Takeaways

By completing this chapter, you have learned to think about data mathematically.

You now understand:

- how information becomes vectors
- how vectors become matrices
- how matrices transform information
- how similarity is measured
- how dimensions are reduced
- why certain directions are more important than others

These ideas form the mathematical foundation upon which the rest of Machine Learning is built.

---

### Looking Ahead: Calculus

Linear Algebra taught us **how to represent and transform information**.

The next question is equally important.

> **How do we measure change?**

Machine Learning learns by continuously adjusting its parameters.

To understand how that learning happens, we need the mathematics of change itself.

That mathematics is **Calculus**.

In the next chapter, we will build the mathematical foundation that leads directly to:

- derivatives
- gradients
- optimization
- gradient descent
- backpropagation

Without Calculus, models cannot learn.

Without Linear Algebra, they cannot represent what they have learned.

Together, these two branches of mathematics form the backbone of modern Machine Learning.

---
