# Appendix A: Types of Matrices

This appendix is a quick reference to common types of matrices used in mathematics, statistics, and Machine Learning.

---

## 1. Matrix Size and Notation

A matrix is a rectangular arrangement of numbers organized into **rows** and **columns**.

An $m \times n$ matrix has:

- $m$ rows
- $n$ columns
- $mn$ entries

For example,

```math
A =
\begin{bmatrix}
1 & 2\\
3 & 4\\
5 & 6
\end{bmatrix}
```

is a $3 \times 2$ matrix because it has three rows and two columns.

The entry in row $i$ and column $j$ is usually written as $a_{ij}$.

For the matrix above,

```math
a_{21} = 3
```

because 3 is in the second row and first column.

---

## 2. Matrices Classified by Shape

### 2.1. Row Matrix

A **row matrix**, also called a **row vector**, has exactly one row.

Its shape is $1 \times n$.

```math
A =
\begin{bmatrix}
2 & 5 & 8
\end{bmatrix}
```

This is a $1 \times 3$ row matrix.

### 2.2. Column Matrix

A **column matrix**, also called a **column vector**, has exactly one column.

Its shape is $m \times 1$.

```math
B =
\begin{bmatrix}
2\\
5\\
8
\end{bmatrix}
```

This is a $3 \times 1$ column matrix.

### 2.3. Rectangular Matrix

A **rectangular matrix** has a different number of rows and columns, so $m \ne n$.

A $3 \times 2$ example is:

```math
C =
\begin{bmatrix}
1 & 2\\
3 & 4\\
5 & 6
\end{bmatrix}
```

A $2 \times 3$ example is:

```math
D =
\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6
\end{bmatrix}
```

Rectangular matrices frequently represent datasets. Rows may represent observations, while columns represent features.

### 2.4. Tall Matrix

A **tall matrix** has more rows than columns, so $m > n$.

```math
A =
\begin{bmatrix}
1 & 2\\
3 & 4\\
5 & 6
\end{bmatrix}
```

This $3 \times 2$ matrix is tall. In Machine Learning, a design matrix is tall when there are more observations than features.

### 2.5. Wide Matrix

A **wide matrix** has more columns than rows, so $m < n$.

```math
A =
\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6
\end{bmatrix}
```

This $2 \times 3$ matrix is wide. This shape can occur when a dataset has more features than observations.

### 2.6. Square Matrix

A **square matrix** has the same number of rows and columns, so $m = n$.

A $2 \times 2$ example is:

```math
A =
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
```

A $3 \times 3$ example is:

```math
B =
\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6\\
7 & 8 & 9
\end{bmatrix}
```

Concepts such as determinants, eigenvalues, and matrix inverses are defined for square matrices.

---

## 3. Matrices Classified by Their Entries

### 3.1. Zero Matrix

A **zero matrix**, also called a **null matrix**, contains only zeros.

```math
O_{2 \times 3} =
\begin{bmatrix}
0 & 0 & 0\\
0 & 0 & 0
\end{bmatrix}
```

The zero matrix is the additive identity:

```math
A + O = A
```

### 3.2. Constant Matrix

A **constant matrix** has the same value in every position.

```math
A =
\begin{bmatrix}
4 & 4 & 4\\
4 & 4 & 4
\end{bmatrix}
```

The zero matrix and the all-ones matrix are special constant matrices.

### 3.3. All-Ones Matrix

An **all-ones matrix** contains 1 in every position and is often denoted by $J$ or $\mathbf{1}$.

```math
J_{2 \times 3} =
\begin{bmatrix}
1 & 1 & 1\\
1 & 1 & 1
\end{bmatrix}
```

### 3.4. Binary Matrix

A **binary matrix** contains only 0 and 1.

```math
A =
\begin{bmatrix}
1 & 0 & 1\\
0 & 1 & 0\\
1 & 1 & 0
\end{bmatrix}
```

Binary matrices can represent masks, graph connections, or encoded categorical information.

### 3.5. Sparse Matrix

A **sparse matrix** contains mostly zeros.

```math
A =
\begin{bmatrix}
0 & 0 & 5\\
0 & 0 & 0\\
2 & 0 & 0
\end{bmatrix}
```

Sparse storage formats record only the nonzero entries and their positions. This can reduce memory usage and computation when large matrices contain relatively few nonzero values.

### 3.6. Dense Matrix

A **dense matrix** contains relatively few zero entries.

```math
A =
\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6\\
7 & 8 & 9
\end{bmatrix}
```

The distinction between sparse and dense depends on how many entries are zero relative to the matrix size; there is no universal percentage that separates them.

---

## 4. Diagonal and Triangular Matrices

### 4.1. Diagonal Matrix

A **diagonal matrix** is a square matrix whose entries outside the main diagonal are all zero.

```math
D =
\begin{bmatrix}
2 & 0 & 0\\
0 & 5 & 0\\
0 & 0 & 8
\end{bmatrix}
```

The diagonal entries may be zero or nonzero.

Multiplication by a diagonal matrix can scale different coordinates independently.

### 4.2. Scalar Matrix

A **scalar matrix** is a diagonal matrix whose diagonal entries are all equal.

```math
A =
\begin{bmatrix}
4 & 0 & 0\\
0 & 4 & 0\\
0 & 0 & 4
\end{bmatrix}
= 4I
```

Every scalar matrix is diagonal, but not every diagonal matrix is scalar.

### 4.3. Identity Matrix

An **identity matrix** is a diagonal matrix with 1 on the main diagonal.

```math
I_3 =
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1
\end{bmatrix}
```

The identity matrix is the multiplicative identity:

```math
AI = IA = A
```

Its size must be compatible with the matrix being multiplied.

### 4.4. Upper Triangular Matrix

An **upper triangular matrix** is a square matrix whose entries below the main diagonal are zero.

```math
U =
\begin{bmatrix}
2 & 3 & 1\\
0 & 5 & 4\\
0 & 0 & 7
\end{bmatrix}
```

### 4.5. Lower Triangular Matrix

A **lower triangular matrix** is a square matrix whose entries above the main diagonal are zero.

```math
L =
\begin{bmatrix}
2 & 0 & 0\\
3 & 5 & 0\\
1 & 4 & 7
\end{bmatrix}
```

Triangular matrices are especially useful when solving linear systems. Their determinant is the product of their diagonal entries.

---

## 5. Matrices Defined by Transposition

The **transpose** of a matrix exchanges its rows and columns.

If $A$ is $m \times n$, then $A^T$ is $n \times m$.

```math
A =
\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6
\end{bmatrix},
\qquad
A^T =
\begin{bmatrix}
1 & 4\\
2 & 5\\
3 & 6
\end{bmatrix}
```

### 5.1. Symmetric Matrix

A **symmetric matrix** is a square matrix that equals its transpose:

```math
A^T = A
```

For example,

```math
A =
\begin{bmatrix}
2 & 1 & 4\\
1 & 3 & 5\\
4 & 5 & 6
\end{bmatrix}
```

The entries mirror each other across the main diagonal. Covariance matrices are symmetric.

### 5.2. Skew-Symmetric Matrix

A **skew-symmetric matrix** is a square matrix satisfying:

```math
A^T = -A
```

For example,

```math
A =
\begin{bmatrix}
0 & 2 & -1\\
-2 & 0 & 3\\
1 & -3 & 0
\end{bmatrix}
```

Every diagonal entry of a real skew-symmetric matrix must be zero.

---

## 6. Matrices Defined by Determinants and Inverses

### 6.1. Singular Matrix

A **singular matrix** is a square matrix that has no inverse.

Equivalently,

```math
\det(A) = 0
```

For example,

```math
A =
\begin{bmatrix}
1 & 2\\
2 & 4
\end{bmatrix}
```

The second row is twice the first, so the rows are linearly dependent and the matrix is singular.

### 6.2. Nonsingular Matrix

A **nonsingular matrix**, also called an **invertible matrix**, has an inverse.

Equivalently,

```math
\det(A) \ne 0
```

For example,

```math
A =
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
```

Its determinant is $-2$, so its inverse exists.

### 6.3. Orthogonal Matrix

A real square matrix $Q$ is **orthogonal** if its columns and rows are orthonormal.

It satisfies:

```math
Q^T Q = QQ^T = I
```

Therefore,

```math
Q^{-1} = Q^T
```

For example,

```math
Q =
\begin{bmatrix}
0 & -1\\
1 & 0
\end{bmatrix}
```

This matrix represents a $90^\circ$ rotation. Orthogonal transformations preserve lengths and angles.

---

## 7. Matrices Defined by Repeated Multiplication

### 7.1. Idempotent Matrix

An **idempotent matrix** is a square matrix satisfying:

```math
A^2 = A
```

For example,

```math
A =
\begin{bmatrix}
1 & 0\\
0 & 0
\end{bmatrix}
```

Projection matrices are idempotent because applying the same projection twice has the same effect as applying it once.

### 7.2. Involutory Matrix

An **involutory matrix** is a square matrix satisfying:

```math
A^2 = I
```

For example,

```math
A =
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
```

This matrix swaps two coordinates. Applying it twice returns the original vector.

### 7.3. Nilpotent Matrix

A **nilpotent matrix** is a square matrix for which some positive integer $k$ satisfies:

```math
A^k = O
```

For example,

```math
A =
\begin{bmatrix}
0 & 1\\
0 & 0
\end{bmatrix}
```

For this matrix, $A^2 = O$.

---

## 8. Matrices Defined by Quadratic Forms

These definitions apply to real symmetric matrices.

### 8.1. Positive Definite Matrix

A symmetric matrix $A$ is **positive definite** if:

```math
\mathbf{x}^T A \mathbf{x} > 0
```

for every nonzero vector $\mathbf{x}$.

For example,

```math
A =
\begin{bmatrix}
2 & 0\\
0 & 3
\end{bmatrix}
```

All eigenvalues of a real symmetric positive definite matrix are positive. Such matrices appear in optimization, covariance modeling, and second-order methods.

### 8.2. Positive Semidefinite Matrix

A symmetric matrix $A$ is **positive semidefinite** if:

```math
\mathbf{x}^T A \mathbf{x} \ge 0
```

for every vector $\mathbf{x}$.

For example,

```math
A =
\begin{bmatrix}
1 & 1\\
1 & 1
\end{bmatrix}
```

Its eigenvalues are 2 and 0, so it is positive semidefinite but not positive definite.

### 8.3. Negative Definite and Negative Semidefinite Matrices

A symmetric matrix is:

- **negative definite** if $\mathbf{x}^T A \mathbf{x} < 0$ for every nonzero $\mathbf{x}$
- **negative semidefinite** if $\mathbf{x}^T A \mathbf{x} \le 0$ for every $\mathbf{x}$

For example,

```math
A =
\begin{bmatrix}
-2 & 0\\
0 & -3
\end{bmatrix}
```

is negative definite.

### 8.4. Indefinite Matrix

A symmetric matrix is **indefinite** if $\mathbf{x}^T A \mathbf{x}$ can be positive for some vectors and negative for others.

```math
A =
\begin{bmatrix}
1 & 0\\
0 & -1
\end{bmatrix}
```

This behavior is associated with saddle-shaped curvature in optimization.

---

## 9. Special Structural Matrices

### 9.1. Permutation Matrix

A **permutation matrix** is obtained by rearranging the rows of an identity matrix. It has exactly one 1 in each row and each column, with all other entries equal to zero.

```math
P =
\begin{bmatrix}
0 & 1 & 0\\
0 & 0 & 1\\
1 & 0 & 0
\end{bmatrix}
```

Left multiplication by $P$ rearranges the rows of another matrix. Right multiplication rearranges its columns.

### 9.2. Block Matrix

A **block matrix** is partitioned into smaller matrices called blocks.

```math
A =
\begin{bmatrix}
A_{11} & A_{12}\\
A_{21} & A_{22}
\end{bmatrix}
```

For example,

```math
A =
\left[
\begin{array}{cc|c}
1 & 0 & 2\\
0 & 1 & 3\\
\hline
4 & 5 & 6
\end{array}
\right]
```

Block matrices make large systems easier to describe and manipulate.

### 9.3. Toeplitz Matrix

A **Toeplitz matrix** has constant values along every diagonal from upper left to lower right.

```math
T =
\begin{bmatrix}
a & b & c\\
d & a & b\\
e & d & a
\end{bmatrix}
```

Toeplitz structure appears in signal processing and convolution-related computations.

### 9.4. Stochastic Matrix

A **stochastic matrix** is a nonnegative square matrix whose rows or columns sum to 1.

A row-stochastic example is:

```math
P =
\begin{bmatrix}
0.7 & 0.3\\
0.4 & 0.6
\end{bmatrix}
```

Each row sums to 1. Stochastic matrices are used to represent transition probabilities in Markov chains.

### 9.5. Adjacency Matrix

An **adjacency matrix** represents connections in a graph. For an unweighted graph, $a_{ij}=1$ indicates a connection from node $i$ to node $j$, while $a_{ij}=0$ indicates no connection.

```math
A =
\begin{bmatrix}
0 & 1 & 1\\
1 & 0 & 0\\
1 & 0 & 0
\end{bmatrix}
```

The adjacency matrix of an undirected graph is symmetric.

---

## 10. Matrices Commonly Seen in Machine Learning

### 10.1. Design Matrix

A **design matrix**, often written as $X$, stores the input data used by a model.

```math
X =
\begin{bmatrix}
170 & 68\\
165 & 55\\
180 & 82
\end{bmatrix}
```

This $3 \times 2$ example contains three observations and two features. Depending on the convention being used, a column of ones may be added to represent an intercept term.

### 10.2. Weight Matrix

A **weight matrix** stores trainable parameters that connect inputs to outputs or one neural-network layer to another.

```math
W =
\begin{bmatrix}
0.2 & -0.5 & 0.7\\
0.1 & 0.4 & -0.3
\end{bmatrix}
```

This $2 \times 3$ matrix can map a three-dimensional input to a two-dimensional output when used in $W\mathbf{x}$.

### 10.3. Covariance Matrix

A **covariance matrix** records the variance of each variable on its diagonal and pairwise covariances outside the diagonal.

```math
\Sigma =
\begin{bmatrix}
4 & 1.5\\
1.5 & 9
\end{bmatrix}
```

A covariance matrix is symmetric and positive semidefinite.

### 10.4. Correlation Matrix

A **correlation matrix** records pairwise correlations between variables.

```math
R =
\begin{bmatrix}
1 & 0.6 & -0.2\\
0.6 & 1 & 0.1\\
-0.2 & 0.1 & 1
\end{bmatrix}
```

It is symmetric, its diagonal entries are 1, and every entry lies between $-1$ and $1$.

### 10.5. Gram Matrix

Given a matrix $X$, a **Gram matrix** contains inner products between vectors.

Depending on whether the vectors are stored as columns or rows, it can be written as:

```math
G = X^T X
```

or:

```math
G = XX^T
```

A Gram matrix is symmetric and positive semidefinite. Gram matrices appear in kernel methods, similarity calculations, and geometric analysis.

### 10.6. Hessian Matrix

The **Hessian matrix** contains the second-order partial derivatives of a scalar-valued function.

For a function $f(x_1,x_2)$,

```math
H =
\begin{bmatrix}
\dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1 \partial x_2}\\
\dfrac{\partial^2 f}{\partial x_2 \partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2}
\end{bmatrix}
```

When the mixed partial derivatives are continuous, the Hessian is symmetric. It describes local curvature and is used in optimization.

### 10.7. Confusion Matrix

A **confusion matrix** summarizes the predictions of a classification model by comparing predicted classes with actual classes.

For binary classification,

|  | Predicted Positive | Predicted Negative |
|---|---:|---:|
| Actual Positive | True Positive | False Negative |
| Actual Negative | False Positive | True Negative |

Unlike most matrices in this appendix, a confusion matrix is primarily a reporting structure rather than an object used for linear transformations.

---

## 11. Important Relationships

Some matrix types are special cases of others:

- Every identity matrix is a scalar matrix.
- Every scalar matrix is a diagonal matrix.
- Every diagonal matrix is both upper triangular and lower triangular.
- Every diagonal matrix is symmetric.
- Every symmetric, skew-symmetric, triangular, orthogonal, singular, or nonsingular matrix is square.
- A matrix can belong to several categories at the same time.

For example,

```math
I_2 =
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
```

is simultaneously:

- square
- diagonal
- scalar
- identity
- symmetric
- upper triangular
- lower triangular
- nonsingular
- orthogonal
- positive definite
- idempotent
- involutory

---

## 12. Quick Reference

| Matrix Type | Defining Feature | Typical Shape |
|---|---|---|
| Row | Exactly one row | $1 \times n$ |
| Column | Exactly one column | $m \times 1$ |
| Rectangular | Different numbers of rows and columns | $m \times n$, where $m \ne n$ |
| Tall | More rows than columns | $m \times n$, where $m > n$ |
| Wide | More columns than rows | $m \times n$, where $m < n$ |
| Square | Same number of rows and columns | $n \times n$ |
| Zero | Every entry is 0 | Any shape |
| Constant | Every entry has the same value | Any shape |
| Binary | Every entry is 0 or 1 | Any shape |
| Sparse | Most entries are 0 | Any shape |
| Diagonal | Entries outside the main diagonal are 0 | $n \times n$ |
| Scalar | Diagonal entries are equal; other entries are 0 | $n \times n$ |
| Identity | Diagonal entries are 1; other entries are 0 | $n \times n$ |
| Upper triangular | Entries below the main diagonal are 0 | $n \times n$ |
| Lower triangular | Entries above the main diagonal are 0 | $n \times n$ |
| Symmetric | $A^T=A$ | $n \times n$ |
| Skew-symmetric | $A^T=-A$ | $n \times n$ |
| Singular | $\det(A)=0$ | $n \times n$ |
| Nonsingular | $\det(A)\ne 0$ | $n \times n$ |
| Orthogonal | $Q^TQ=I$ | $n \times n$ |
| Idempotent | $A^2=A$ | $n \times n$ |
| Involutory | $A^2=I$ | $n \times n$ |
| Nilpotent | $A^k=O$ for some positive integer $k$ | $n \times n$ |
| Positive definite | $\mathbf{x}^TA\mathbf{x}>0$ for nonzero $\mathbf{x}$ | $n \times n$ |
| Positive semidefinite | $\mathbf{x}^TA\mathbf{x}\ge 0$ | $n \times n$ |
| Permutation | One 1 in each row and column | $n \times n$ |
| Stochastic | Nonnegative entries; rows or columns sum to 1 | $n \times n$ |

The most useful first step when examining any matrix is to identify its shape. After that, inspect its entries and determine whether it has additional structure that can simplify computation or reveal useful mathematical properties.
