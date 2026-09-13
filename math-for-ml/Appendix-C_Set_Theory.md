# Appendix C: Set Theory

This appendix is a compact reference for the set-theory concepts that appear throughout the study set. The main chapters introduce these ideas in their machine-learning or probability context; this appendix collects the notation, relationships, and core operations in one place.

## 1. Basic Notation

A **set** is a collection of distinct objects, called **elements** or **members**. Sets are commonly named with capital letters and written using braces:

```math
A = \{1,2,3\}
```

Membership is written as:

```math
2 \in A
```

Non-membership is written as:

```math
5 \notin A
```

## 2. Types of Sets

### 2.1. Empty Set

The **empty set** contains no elements:

```math
\varnothing = \{\}
```

### 2.2. Universal Set

The **universal set**, often written as $U$, contains every object under consideration in a particular problem. Its meaning depends on the context.

### 2.3. Finite and Infinite Sets

A finite set has a limited number of elements. An infinite set has no final element, such as:

```math
\mathbb{N} = \{1,2,3,\ldots\}
```

### 2.4. Singleton Set

A singleton set contains exactly one element:

```math
\{x\}
```

## 3. Subsets and Set Equality

### 3.1. Subset

Set $A$ is a subset of set $B$ when every element of $A$ is also an element of $B$:

```math
A \subseteq B
\quad\Longleftrightarrow\quad
\forall x,\; x\in A \Rightarrow x\in B
```

Every set is a subset of itself, and the empty set is a subset of every set.

### 3.2. Proper Subset

Set $A$ is a proper subset of $B$ when $A$ is a subset of $B$ but $A$ and $B$ are not equal:

```math
A \subset B
```

### 3.3. Set Equality

Two sets are equal when they contain exactly the same elements:

```math
A = B
\quad\Longleftrightarrow\quad
A \subseteq B \text{ and } B \subseteq A
```

## 4. Set Operations

### 4.1. Union

The union contains every element that belongs to $A$, to $B$, or to both:

```math
A \cup B = \{x : x\in A \text{ or } x\in B\}
```

### 4.2. Intersection

The intersection contains the elements shared by both sets:

```math
A \cap B = \{x : x\in A \text{ and } x\in B\}
```

### 4.3. Complement

The complement of $A$ contains the elements in the universal set that are not in $A$:

```math
A^c = U \setminus A
```

### 4.4. Difference

The difference $A\setminus B$ contains elements that belong to $A$ but not to $B$:

```math
A\setminus B = \{x : x\in A \text{ and } x\notin B\}
```

### 4.5. Symmetric Difference

The symmetric difference contains elements that belong to exactly one of the two sets:

```math
A\triangle B = (A\setminus B)\cup(B\setminus A)
```

## 5. Important Set Relationships

### 5.1. Disjoint Sets

Two sets are disjoint when they have no elements in common:

```math
A\cap B = \varnothing
```

### 5.2. De Morgan's Laws

```math
(A\cup B)^c = A^c\cap B^c
```

```math
(A\cap B)^c = A^c\cup B^c
```

### 5.3. Inclusion-Exclusion

For finite sets, the number of elements in a union is:

```math
|A\cup B| = |A| + |B| - |A\cap B|
```

## 6. Cardinality and Counting

The **cardinality** of a finite set is its number of elements, written $|A|$.

For a finite set with $n$ elements, its power set—the set of all subsets—has:

```math
|\mathcal{P}(A)| = 2^n
```

## 7. Cartesian Products and Relations

### 7.1. Cartesian Product

The Cartesian product $A\times B$ is the set of ordered pairs whose first element comes from $A$ and second element comes from $B$:

```math
A\times B = \{(a,b): a\in A,\; b\in B\}
```

If $A$ and $B$ are finite:

```math
|A\times B| = |A||B|
```

### 7.2. Relation

A relation from $A$ to $B$ is a subset of the Cartesian product $A\times B$. Relations can describe which objects are associated with one another.

## 8. Functions as Set Mappings

A function $f$ from a domain $A$ to a codomain $B$ assigns exactly one element of $B$ to each element of $A$:

```math
f:A\to B
```

The **domain** is the set of valid inputs, and the **codomain** is the declared set of possible outputs. The **range** or image is the subset of the codomain that the function actually produces:

```math
\text{range}(f) = \{f(a):a\in A\}\subseteq B
```

### 8.1. Injective Function

A function is injective when distinct inputs have distinct outputs:

```math
f(a_1)=f(a_2)\Rightarrow a_1=a_2
```

### 8.2. Surjective Function

A function is surjective when every element of the codomain is produced by at least one input:

```math
\forall b\in B,\;\exists a\in A\text{ such that }f(a)=b
```

### 8.3. Bijective Function

A function is bijective when it is both injective and surjective.

## 9. Set Theory in Probability and Machine Learning

### 9.1. Sample Spaces and Events

In probability, the sample space $\Omega$ is the set of all possible outcomes. An event $A$ is a subset of that sample space:

```math
A\subseteq\Omega
```

| Set operation | Probability interpretation |
| --- | --- |
| $A\cup B$ | $A$ or $B$ occurs |
| $A\cap B$ | Both $A$ and $B$ occur |
| $A^c$ | $A$ does not occur |
| $A\cap B=\varnothing$ | $A$ and $B$ are mutually exclusive |

### 9.2. Datasets and Subsets

A dataset can be viewed as a finite set of observations. Training, validation, and test datasets are subsets created for different roles in model development. These subsets should be defined carefully so evaluation data does not leak into training.

### 9.3. Feature and Label Spaces

If a feature vector has $d$ coordinates, its feature space can be represented as a subset of $\mathbb{R}^d$:

```math
\mathbf{x}\in\mathcal{X}\subseteq\mathbb{R}^d
```

Likewise, a classification target takes values from a label set $\mathcal{Y}$.

### 9.4. Set Similarity

The Jaccard similarity of two nonempty finite sets is:

```math
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
```

It measures the proportion of the union shared by both sets.

## 10. Quick Reference

| Concept | Meaning |
|---------|---------|
| $x\in A$ | $x$ belongs to $A$ |
| $A\subseteq B$ | Every element of $A$ belongs to $B$ |
| $A\cup B$ | Elements in either set |
| $A\cap B$ | Elements shared by both sets |
| $A^c$ | Elements outside $A$ but inside $U$ |
| $A\setminus B$ | Elements in $A$ but not in $B$ |
| $\lvert A\rvert$ | Number of elements in a finite set |
| $A\times B$ | Set of ordered pairs from $A$ and $B$ |
| $f:A\to B$ | Function from domain $A$ to codomain $B$ |

## 11. Related Study Sections

The main study documents introduce these concepts in their application context:

- [Sample Spaces and Events](03-Probability_for_Machine_Learning.md#experiments-outcomes-sample-spaces-and-events)
- [Basic Probability Rules](03-Probability_for_Machine_Learning.md#basic-probability-rules)
- [Mutually Exclusive Events](03-Probability_for_Machine_Learning.md#mutually-exclusive-events)
- [Span and Linear Combinations](01-Linear_Algebra.md#span-and-linear-combinations)
- [Functions, Slopes, and Rate of Change](02-Calculus.md#functions-slopes-and-rate-of-change)
- [Jaccard Similarity](../classical-ml/11-Similarity_Measures.md#jaccard-similarity)
- [Population and Sample](04-Statistics_for_Machine_Learning.md#population-and-sample)
