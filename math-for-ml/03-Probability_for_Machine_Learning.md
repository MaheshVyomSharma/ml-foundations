# 03. Probability for Machine Learning

Probability is the mathematics of **uncertainty**.

In deterministic mathematics, the same inputs produce a precisely determined result. In probability, we deal with situations where the exact outcome is uncertain, but the possible outcomes and their likelihoods can still be described mathematically.

This makes probability fundamental to machine learning.

A machine learning model rarely says:

> "This outcome is absolutely certain."

Instead, it often reasons more like:

> "Given what I have observed, this outcome is more likely than the alternatives."

Classification probabilities, uncertainty estimates, probability distributions, likelihood functions, Bayesian inference, and many loss functions all arise from this idea.

> **ML connection:** Machine learning is largely about learning patterns from incomplete and noisy observations. Probability provides the mathematical language for expressing the uncertainty that remains.

---

## 1. Experiments, Outcomes, Sample Spaces, and Events

Probability begins with four basic ideas.

### 1.1. Random Experiment

A **random experiment** is a process whose exact outcome cannot be known beforehand.

Examples:

- tossing a coin
- rolling a die
- selecting a customer from a population
- observing whether a user clicks an advertisement
- measuring tomorrow's temperature
- determining whether an email is spam

The word *experiment* does not necessarily mean a laboratory experiment. It simply refers to some process that produces an uncertain outcome.

### 1.2. Outcome

An **outcome** is one possible result of the experiment.

For a coin toss:

```math
H
```

is one outcome and

```math
T
```

is another.

For a six-sided die, the possible outcomes are:

```math
1,2,3,4,5,6
```

### 1.3. Sample Space

The **sample space**, usually represented by $S$ or $\Omega$, is the set of **all possible outcomes**.

For a coin:

```math
S = \{H,T\}
```

For a six-sided die:

```math
S = \{1,2,3,4,5,6\}
```

The sample space defines the universe within which the probability problem exists.

### 1.4. Event

An **event** is a set containing one or more outcomes from the sample space.

Suppose a die is rolled.

The sample space is:

```math
S = \{1,2,3,4,5,6\}
```

Define event $A$ as:

> Rolling an even number.

Then:

```math
A = \{2,4,6\}
```

Therefore, an event is simply a **subset of the sample space**:

```math
A \subseteq S
```

This connection between probability and set theory becomes important because events can be combined using operations such as:

- union
- intersection
- complement

---

## 2. Probability of an Event

The probability of an event $A$ is written as:

```math
P(A)
```

and satisfies:

```math
0 \leq P(A) \leq 1
```

The extremes have straightforward interpretations:

```math
P(A)=0
```

means the event is impossible, while:

```math
P(A)=1
```

means the event is certain.

For equally likely outcomes:

```math
P(A)=
\frac{\text{number of favourable outcomes}}
{\text{total number of possible outcomes}}
```

For example, consider the probability of rolling an even number on a fair six-sided die.

There are three favourable outcomes:

```math
A=\{2,4,6\}
```

and six possible outcomes:

```math
P(A)=\frac{3}{6}=\frac{1}{2}
```

So:

```math
P(A)=0.5
```

### 2.1. Probability as Long-Run Frequency

Probability can also be understood through repeated observations.

If a fair coin is tossed many times, the proportion of heads will tend toward:

```math
0.5
```

as the number of tosses becomes very large.

For a finite number of experiments, however, the observed proportion does not have to equal the theoretical probability.

For example, ten coin tosses might produce:

```math
7H,\;3T
```

giving an observed proportion:

```math
\frac{7}{10}=0.7
```

This does **not** imply that the probability of heads has become $0.7$. It is simply random variation in a small sample.

This distinction between:

```math
\text{theoretical probability}
```

and

```math
\text{observed frequency}
```

will later lead directly into statistical sampling and the **Law of Large Numbers**.

> **ML connection:** A machine learning model only sees a finite sample of data. It tries to learn patterns that represent the larger, unseen population. Much of statistics and ML is therefore concerned with distinguishing genuine underlying patterns from random variation in the observed sample.

---

## 3. Basic Probability Rules

Several simple rules form the foundation for more advanced probability.

### 3.1. Complement Rule

The **complement** of event $A$, written $A^c$, means that $A$ does **not** occur.

Since either $A$ occurs or it does not:

```math
P(A)+P(A^c)=1
```

Therefore:

```math
P(A^c)=1-P(A)
```

Suppose:

```math
P(\text{spam})=0.2
```

Then:

```math
P(\text{not spam})=1-0.2=0.8
```

### 3.2. Union — OR

The union of two events:

```math
A \cup B
```

means:

> $A$ occurs **or** $B$ occurs, or both occur.

For two general events:

```math
P(A\cup B)
=
P(A)+P(B)-P(A\cap B)
```

The intersection is subtracted because outcomes belonging to both events would otherwise be counted twice.

### 3.3. Intersection — AND

The intersection:

```math
A\cap B
```

means:

> Both $A$ **and** $B$ occur.

For example:

- $A$: customer is under 30
- $B$: customer purchases the product

Then:

```math
A\cap B
```

represents customers who are **both under 30 and purchased the product**.

This distinction is worth fixing firmly:

```math
\cup \quad \longrightarrow \quad \text{OR}
```

```math
\cap \quad \longrightarrow \quad \text{AND}
```

These same set relationships will appear repeatedly when working with conditional probability, independence, classification outcomes, and statistical reasoning.

---

## 4. Mutually Exclusive Events

Two events are **mutually exclusive** if they cannot occur simultaneously.

Therefore:

```math
A\cap B=\varnothing
```

and consequently:

```math
P(A\cap B)=0
```

For mutually exclusive events, the union rule simplifies to:

```math
P(A\cup B)=P(A)+P(B)
```

For example, on a single die roll:

- $A=\{2\}$
- $B=\{5\}$

The die cannot simultaneously show both 2 and 5.

Therefore:

```math
P(A\cap B)=0
```

and:

```math
P(A\cup B)
=
\frac{1}{6}+\frac{1}{6}
=
\frac{1}{3}
```

### 4.1. Mutually Exclusive Is Not the Same as Independent

This distinction is extremely important.

**Mutually exclusive** means:

> If one event occurs, the other cannot occur.

**Independent** means:

> The occurrence of one event does not change the probability of the other.

In fact, two events with non-zero probability that are mutually exclusive **cannot be independent**.

If $A$ occurs and therefore makes $B$ impossible, then knowing that $A$ occurred clearly gives us information about $B$.

We will formalize independence after introducing conditional probability.

> **ML connection:** Much of machine learning ultimately involves relationships between events and variables. Whether information about one variable changes what we believe about another is at the heart of feature relationships, probabilistic classifiers, Bayesian models, and statistical inference.

---

### 4.2. Mental Model

The foundations of probability can be compressed into one picture:

```math
\boxed{
\text{Experiment}
\rightarrow
\text{Outcome}
\rightarrow
\text{Sample Space}
\rightarrow
\text{Event}
\rightarrow
\text{Probability}
}
```

And probability itself gives us a mathematical way of answering:

> **How strongly should I believe that this event will occur, given what I currently know?**

That final phrase — **"given what I currently know"** — leads directly to one of the most important ideas in probability and machine learning: 

**conditional probability.**

---

## 5. Conditional Probability

Ordinary probability asks:

> What is the probability that event $A$ occurs?

Conditional probability asks a more interesting question:

> What is the probability that event $A$ occurs **given that we already know event $B$ has occurred?**

It is written as:

```math
P(A \mid B)
```

and read as:

> "Probability of $A$, given $B$."

The vertical bar $\mid$ means **given**.

This simple idea is enormously important in machine learning because predictions are almost always made **given some observed information**.

For example:

```math
P(\text{spam} \mid \text{words in email})
```

```math
P(\text{disease} \mid \text{test result})
```

```math
P(\text{customer churns} \mid \text{customer behaviour})
```

A model observes some evidence and changes its estimate of what is likely to happen.

> **ML connection:** Prediction itself can often be viewed as conditional probability — estimate the probability of an output **given the available features**.

---

### 5.1. Understanding Conditional Probability

Suppose 100 customers visit a website.

- 60 are mobile users.
- 30 customers make a purchase.
- 24 of those purchasers are mobile users.

Let:

- $A$ = customer makes a purchase
- $B$ = customer uses a mobile device

Without knowing anything else:

```math
P(A)=\frac{30}{100}=0.30
```

But suppose we already know that the customer is a mobile user.

We no longer care about all 100 customers.

Our relevant population has been reduced to the 60 mobile users.

Among those 60 users, 24 purchased.

Therefore:

```math
P(A \mid B)=\frac{24}{60}=0.40
```

The probability has changed:

```math
P(A)=0.30
```

but:

```math
P(A \mid B)=0.40
```

Knowing that the customer is a mobile user provided **additional information**.

This is the central intuition behind conditional probability:

> **Conditioning changes the universe under consideration.**

Instead of asking how often $A$ occurs everywhere, we ask how often $A$ occurs **inside the region where $B$ is already known to be true**.

---

### 5.2. Conditional Probability Formula

Formally:

```math
P(A \mid B)
=
\frac{P(A \cap B)}
{P(B)}
```

provided:

```math
P(B)>0
```

The numerator represents cases where **both $A$ and $B$** occur.

The denominator restricts our attention to cases where $B$ occurs.

Using the customer example:

```math
P(A \cap B)=\frac{24}{100}=0.24
```

and:

```math
P(B)=\frac{60}{100}=0.60
```

Therefore:

```math
P(A \mid B)
=
\frac{0.24}{0.60}
=
0.40
```

The denominator is doing something conceptually important:

> It **renormalizes the probability space** after we restrict ourselves to event $B$.

The probabilities inside this reduced space must once again add up to 1.

---

### 5.3. Direction Matters

Conditional probability is directional.

In general:

```math
P(A \mid B)
\neq
P(B \mid A)
```

Using our customer example:

```math
P(\text{purchase} \mid \text{mobile})
=
\frac{24}{60}
=
0.40
```

But:

```math
P(\text{mobile} \mid \text{purchase})
=
\frac{24}{30}
=
0.80
```

These answer completely different questions.

The first asks:

> Among mobile users, how many purchase?

The second asks:

> Among purchasers, how many are mobile users?

This distinction becomes extremely important when we reach **Bayes' theorem**, which gives us a mathematical way to reverse this direction.

---

## 6. The Multiplication Rule

Starting with the conditional probability definition:

```math
P(A \mid B)
=
\frac{P(A \cap B)}
{P(B)}
```

we can rearrange it:

```math
P(A \cap B)
=
P(A \mid B)P(B)
```

Similarly:

```math
P(A \cap B)
=
P(B \mid A)P(A)
```

Therefore:

```math
P(A \mid B)P(B)
=
P(B \mid A)P(A)
```

Keep that final relationship in mind.

It is the mathematical doorway to **Bayes' theorem**.

---

## 7. Independent Events

Two events are **independent** when knowing that one occurred provides no information about whether the other occurs.

Mathematically, if $A$ and $B$ are independent:

```math
P(A \mid B)=P(A)
```

and:

```math
P(B \mid A)=P(B)
```

In plain English:

> Knowing $B$ happened does not change the probability of $A$.

Consider tossing two fair coins.

Let:

- $A$ = first coin produces heads
- $B$ = second coin produces heads

Then:

```math
P(A)=\frac{1}{2}
```

Even after learning that the second coin produced heads:

```math
P(A \mid B)=\frac{1}{2}
```

The second toss tells us nothing about the first.

Therefore the events are independent.

---

### 7.1. Multiplication Rule for Independent Events

Recall the general multiplication rule:

```math
P(A \cap B)
=
P(A \mid B)P(B)
```

For independent events:

```math
P(A \mid B)=P(A)
```

Therefore:

```math
P(A \cap B)
=
P(A)P(B)
```

For two independent coin tosses:

```math
P(H_1 \cap H_2)
=
\frac{1}{2}
\times
\frac{1}{2}
=
\frac{1}{4}
```

So the probability of obtaining two heads is $0.25$.

---

### 7.2. Dependence

If knowing $B$ changes the probability of $A$, the events are **dependent**.

That is:

```math
P(A \mid B)\neq P(A)
```

Suppose:

```math
P(\text{purchase})=0.30
```

but:

```math
P(\text{purchase} \mid \text{mobile})=0.40
```

Then device type and purchase behaviour are not independent.

Knowing the device type changes what we believe about the probability of purchase.

This idea becomes particularly important in machine learning because useful features generally contain **information about the target**.

If:

```math
P(Y \mid X)=P(Y)
```

then knowing $X$ does not change our knowledge of $Y$.

In that probabilistic sense, $X$ contains no predictive information about $Y$.

> **ML connection:** Machine learning searches for information in features $X$ that changes what we can infer about target $Y$. A useful predictive relationship therefore often manifests itself as some form of statistical dependence.

---

### 7.3. Independence vs Mutually Exclusive Events

These two concepts sound similar but mean almost opposite things.

For **mutually exclusive events**:

```math
P(A \cap B)=0
```

If $A$ happens, $B$ cannot happen.

For **independent events**:

```math
P(A \cap B)=P(A)P(B)
```

If $A$ happens, it tells us nothing about whether $B$ happens.

Consider rolling a die.

Let:

- $A$ = roll a 2
- $B$ = roll a 5

These are mutually exclusive because both cannot occur on the same roll.

But they are **not independent**.

Before observing the die:

```math
P(B)=\frac{1}{6}
```

After learning that $A$ occurred:

```math
P(B \mid A)=0
```

Knowledge of $A$ dramatically changed the probability of $B$.

Therefore they are dependent.

A useful mental distinction is:

> **Mutually exclusive:** one occurring prevents the other.

> **Independent:** one occurring tells us nothing about the other.

---

### 7.4. A Useful ML Interpretation

Suppose $X$ represents input features and $Y$ represents the target.

Machine learning is fundamentally interested in:

```math
P(Y \mid X)
```

For classification, for example:

```math
P(Y=1 \mid X)
```

might represent the probability that an observation belongs to class 1 given its features.

If the features provide useful information:

```math
P(Y \mid X)
\neq
P(Y)
```

The model's knowledge of $X$ changes its belief about $Y$.

That is exactly what we want.

A classifier can therefore be viewed conceptually as a machine that takes evidence $X$ and estimates:

```math
\boxed{P(Y \mid X)}
```

This idea sits underneath logistic regression, probabilistic classifiers, Bayesian methods, and much of modern machine learning.

---

### 7.5. Mental Model

Conditional probability:

```math
\boxed{
P(A \mid B)
=
\text{Probability of A after learning B}
}
```

Independence:

```math
\boxed{
P(A \mid B)=P(A)
}
```

means:

> Learning $B$ taught us nothing about $A$.

Dependence:

```math
\boxed{
P(A \mid B)\neq P(A)
}
```

means:

> Learning $B$ changed what we believe about $A$.

And that gives us the central probabilistic pattern behind prediction:

```math
\boxed{
\text{Prior knowledge}
+
\text{new evidence}
\rightarrow
\text{updated belief}
}
```

The mathematics that formalizes exactly this process is **Bayes' theorem**.

---

## 8. Bayes' Theorem

Bayes' theorem describes how we should **update a probability when new evidence becomes available**.

Its basic idea is remarkably simple:

> Start with what you believed before seeing the evidence, incorporate the evidence, and obtain an updated belief.

This pattern appears throughout machine learning:

```math
\text{prior belief}
+
\text{observed evidence}
\rightarrow
\text{updated belief}
```

Bayes' theorem is therefore not merely a probability formula. It provides a general mathematical framework for **learning from evidence**.

---

### 8.1. Deriving Bayes' Theorem

Recall the multiplication rule:

```math
P(A \cap B)
=
P(A \mid B)P(B)
```

The same intersection can also be written from the opposite direction:

```math
P(A \cap B)
=
P(B \mid A)P(A)
```

Since both expressions describe the same quantity:

```math
P(A \mid B)P(B)
=
P(B \mid A)P(A)
```

Dividing by $P(B)$:

```math
\boxed{
P(A \mid B)
=
\frac{P(B \mid A)P(A)}
{P(B)}
}
```

This is **Bayes' theorem**.

The algebra is simple.

The meaning of the terms is far more important.

---

### 8.2. The Four Parts of Bayes' Theorem

In machine learning and statistics, Bayes' theorem is commonly written as:

```math
P(H \mid D)
=
\frac{P(D \mid H)P(H)}
{P(D)}
```

where:

- $H$ represents a **hypothesis**
- $D$ represents observed **data or evidence**

Each term has a specific interpretation.

#### 8.2.1. Prior

```math
P(H)
```

The **prior probability** represents what we believe about the hypothesis **before observing the new evidence**.

For example:

```math
P(\text{disease})=0.01
```

means that before performing a test, the probability that a randomly selected person has the disease is 1%.

This is our starting belief.

---

#### 8.2.2. Likelihood

```math
P(D \mid H)
```

The **likelihood** describes how probable the observed evidence is **assuming the hypothesis is true**.

For example:

```math
P(\text{positive test} \mid \text{disease})
```

asks:

> If the person actually has the disease, how likely is the test to return positive?

Notice the direction carefully.

Likelihood is **not**:

```math
P(H \mid D)
```

It tells us how compatible the observed evidence is with a particular hypothesis.

> **ML connection:** The term *likelihood* becomes extremely important when we reach **maximum likelihood estimation**, where model parameters are chosen to make the observed training data as likely as possible.

---

#### 8.2.3. Evidence

```math
P(D)
```

The denominator is called the **evidence**, sometimes also called the **marginal likelihood**.

It represents the overall probability of observing the evidence, regardless of which hypothesis produced it.

Its important role is to normalize the result so that the posterior remains a valid probability.

---

#### 8.2.4. Posterior

```math
P(H \mid D)
```

The **posterior probability** is our updated belief about the hypothesis **after observing the evidence**.

Thus Bayes' theorem can be remembered conceptually as:

```math
\boxed{
\text{Posterior}
=
\frac{
\text{Likelihood}\times\text{Prior}
}{
\text{Evidence}
}
}
```

Or even more compactly:

```math
\boxed{
\text{Posterior}
\propto
\text{Likelihood}\times\text{Prior}
}
```

The proportional form is commonly encountered because the evidence acts as a normalization factor.

---

### 8.3. Example: A Medical Test

Suppose a disease occurs in 1% of a population:

```math
P(D)=0.01
```

Therefore:

```math
P(D^c)=0.99
```

Suppose a diagnostic test correctly detects the disease 95% of the time:

```math
P(+ \mid D)=0.95
```

but also produces a positive result in 5% of healthy people:

```math
P(+ \mid D^c)=0.05
```

A person receives a positive test.

What is:

```math
P(D \mid +)
```

?

A tempting answer might be **95%**.

But that would confuse:

```math
P(+ \mid D)
```

with:

```math
P(D \mid +)
```

These are not the same probability.

Bayes' theorem gives:

```math
P(D \mid +)
=
\frac{
P(+ \mid D)P(D)
}{
P(+)
}
```

We first need the overall probability of receiving a positive test.

A positive result can occur in two ways:

1. the person has the disease and tests positive;
2. the person does not have the disease but still tests positive.

Therefore:

```math
P(+)
=
P(+ \mid D)P(D)
+
P(+ \mid D^c)P(D^c)
```

Substituting:

```math
P(+)
=
(0.95)(0.01)
+
(0.05)(0.99)
```

```math
P(+)
=
0.0095+0.0495
=
0.059
```

Now apply Bayes' theorem:

```math
P(D \mid +)
=
\frac{
(0.95)(0.01)
}{
0.059
}
```

```math
P(D \mid +)
\approx
0.161
```

So:

```math
\boxed{
P(D \mid +)\approx16.1\%
}
```

Despite the test detecting 95% of diseased people, a positive result corresponds to only about a 16% probability of actually having the disease in this example.

Why?

Because the disease is **rare**.

The prior probability:

```math
P(D)=0.01
```

matters enormously.

Among a large population, false positives from the 99% who are healthy can greatly outnumber true positives from the 1% who have the disease.

This is an important lesson:

> **Evidence should not be interpreted without considering how probable the underlying hypothesis was in the first place.**

---

### 8.4. A Frequency View

The same example becomes intuitive if we imagine 10,000 people.

Approximately:

```math
100
```

have the disease.

Of these, 95% test positive:

```math
95
```

true positives.

The remaining:

```math
9900
```

people are healthy.

Five percent of them test positive:

```math
495
```

false positives.

Therefore the total number of positive tests is:

```math
95+495=590
```

Of these 590 positive tests, only 95 correspond to people who actually have the disease:

```math
\frac{95}{590}
\approx
0.161
```

which again gives:

```math
16.1\%
```

This frequency interpretation is often the easiest way to build intuition for Bayes' theorem.

---

### 8.5. Bayes as Belief Updating

The deeper idea is not the medical example.

Suppose we begin with:

```math
P(H)
```

Then we observe evidence $D$.

Bayes' theorem transforms our belief into:

```math
P(H \mid D)
```

That posterior can then become the prior when additional evidence arrives.

Conceptually:

```math
\text{Prior}
\rightarrow
\text{Evidence}
\rightarrow
\text{Posterior}
```

Then:

```math
\text{Posterior}_1
\rightarrow
\text{new Prior}
\rightarrow
\text{new Evidence}
\rightarrow
\text{Posterior}_2
```

Thus beliefs can be updated repeatedly as new information arrives.

This is one reason Bayesian reasoning fits naturally with machine learning: **learning itself is fundamentally an updating process**.

---

### 8.6. Bayes' Theorem and Classification

Suppose a classifier wants to determine the class $Y$ of an observation with features $X$.

We ultimately want:

```math
P(Y \mid X)
```

the probability of a class given the observed features.

Bayes' theorem allows us to write:

```math
P(Y \mid X)
=
\frac{
P(X \mid Y)P(Y)
}{
P(X)
}
```

Here:

```math
P(Y)
```

is the prior probability of the class.

```math
P(X \mid Y)
```

describes how likely those features are for observations belonging to that class.

And:

```math
P(Y \mid X)
```

is the posterior probability of the class after observing the features.

A classifier can then choose the class having the largest posterior probability.

This idea forms the foundation of classifiers such as **Naive Bayes**.

---

### 8.7. A Useful Distinction

Bayesian terminology can initially become confusing because several probabilities look almost identical.

Keep their directions explicit:

```math
P(H)
```

**Prior:** How likely was the hypothesis before seeing the evidence?

```math
P(D \mid H)
```

**Likelihood:** If the hypothesis were true, how likely would this evidence be?

```math
P(D)
```

**Evidence:** How likely is this evidence overall?

```math
P(H \mid D)
```

**Posterior:** After seeing the evidence, how likely is the hypothesis?

The most common conceptual mistake is confusing:

```math
P(D \mid H)
```

with:

```math
P(H \mid D)
```

Bayes' theorem exists precisely because these two quantities are **not generally equal**.

---

### 8.8. Mental Model

The mathematical form:

```math
\boxed{
P(H \mid D)
=
\frac{
P(D \mid H)P(H)
}{
P(D)
}
}
```

The conceptual form:

```math
\boxed{
\text{Posterior}
=
\frac{
\text{Likelihood}\times\text{Prior}
}{
\text{Evidence}
}
}
```

And the ML interpretation:

```math
\boxed{
\text{What I believed}
+
\text{what the data tells me}
\rightarrow
\text{what I should believe now}
}
```

> **ML connection:** Bayes' theorem formalizes learning from evidence. Even when an ML algorithm is not explicitly Bayesian, the concepts of likelihood, prior information, conditional probability, and uncertainty appear throughout statistical machine learning.

---

## 9. From Events to Random Variables

So far, probability has dealt mainly with **events**:

```math
P(A), \qquad P(A \mid B)
```

But machine learning usually works with numerical quantities:

- age
- income
- temperature
- price
- number of purchases
- model errors
- predicted probabilities

To apply probability mathematics to such quantities, we need a way to associate numerical values with uncertain outcomes.

That object is called a **random variable**.

A random variable forms the bridge between:

```math
\boxed{
\text{random outcomes}
\rightarrow
\text{numbers}
\rightarrow
\text{probability distributions}
}
```

Probability distributions, in turn, are among the most important mathematical objects in statistics and machine learning.

---

## 10. Random Variables

A **random variable** is a numerical variable whose value depends on the outcome of a random experiment.

Despite its name, a random variable is technically a **function** that maps outcomes from a sample space to numerical values.

Suppose two coins are tossed.

The sample space is:

```math
S=\{HH,HT,TH,TT\}
```

Define a random variable $X$ as:

> Number of heads obtained.

Then:

```math
X(HH)=2
```

```math
X(HT)=1
```

```math
X(TH)=1
```

```math
X(TT)=0
```

Thus the original outcomes:

```math
HH,\ HT,\ TH,\ TT
```

have been converted into numerical values:

```math
0,\ 1,\ 2
```

This allows us to apply mathematical and statistical operations to uncertain outcomes.

> **ML connection:** ML datasets are ultimately collections of variables whose observed values differ across samples. Probability models allow us to describe the uncertainty and variability associated with these values.

---

### 10.1. Random Variable vs Observed Value

It is useful to distinguish between the random variable and a particular value it takes.

Conventionally:

```math
X
```

represents the random variable, while:

```math
x
```

represents a particular observed value.

For example, $X$ might represent the number of purchases made by a randomly selected customer.

An observation might then be:

```math
X=3
```

Similarly, in machine learning we frequently use:

```math
X
```

for input variables or the feature matrix and:

```math
Y
```

for the target variable.

The notation is related, although in ML $X$ often represents many variables collectively rather than one random variable.

---

## 11. Discrete and Continuous Random Variables

Random variables are broadly divided into two types.

### 11.1. Discrete Random Variables

A **discrete random variable** takes values from a finite or countably infinite set.

Examples include:

- number of heads in 10 coin tosses
- number of customers arriving at a shop
- number of defects in a product
- number of clicks on an advertisement
- class labels such as $0$ and $1$

For example:

```math
X \in \{0,1,2,3,\ldots\}
```

The possible values can be individually enumerated.

---

### 11.2. Continuous Random Variables

A **continuous random variable** can take any value within some interval.

Examples include:

- height
- weight
- temperature
- time
- distance
- house price
- measurement error

For example:

```math
X \in [0,\infty)
```

may contain infinitely many possible real-number values.

Between two values such as:

```math
1.5
```

and:

```math
1.6
```

there are infinitely many possible values:

```math
1.51,\ 1.511,\ 1.5111,\ldots
```

This difference between discrete and continuous variables changes how probabilities are represented.

---

## 12. Probability Mass Function — PMF

For a **discrete random variable**, probabilities are described using a **Probability Mass Function (PMF)**.

It gives the probability that $X$ takes a particular value $x$:

```math
P(X=x)
```

Consider the two-coin example where $X$ is the number of heads.

The possible values are:

```math
X\in\{0,1,2\}
```

Their probabilities are:

```math
P(X=0)=\frac{1}{4}
```

```math
P(X=1)=\frac{2}{4}=\frac{1}{2}
```

```math
P(X=2)=\frac{1}{4}
```

The complete probability distribution is therefore:

| $x$ | $P(X=x)$ |
|---:|---:|
| 0 | $0.25$ |
| 1 | $0.50$ |
| 2 | $0.25$ |

A valid PMF must satisfy two conditions.

Every probability must lie between 0 and 1:

```math
0\leq P(X=x)\leq1
```

And the probabilities of all possible values must sum to 1:

```math
\sum_x P(X=x)=1
```

For our example:

```math
0.25+0.50+0.25=1
```

The word **mass** is useful here: probability can be thought of as being placed directly on individual possible values.

---

## 13. Probability Density Function — PDF

For a **continuous random variable**, the situation is different.

Because infinitely many possible values exist, assigning a non-zero probability to every individual value would not work.

Instead, continuous distributions are described using a **Probability Density Function (PDF)**, usually written:

```math
f(x)
```

The crucial idea is:

> For a continuous random variable, probability corresponds to **area under the density curve over an interval**.

For example:

```math
P(a\leq X\leq b)
=
\int_a^b f(x)\,dx
```

This should look familiar from calculus.

The integral accumulates the probability density between $a$ and $b$.

The total area under the PDF must equal 1:

```math
\int_{-\infty}^{\infty} f(x)\,dx=1
```

This is one of the places where the calculus we studied earlier enters probability directly.

> **Math connection:** Integration accumulates infinitely many tiny quantities. In probability, it accumulates probability density over an interval.

---

### 13.1. Density Is Not Probability

This distinction is important.

For a continuous random variable:

```math
f(x)
```

is a **density**, not the probability that $X=x$.

In fact:

```math
P(X=x)=0
```

for any exact individual value of a truly continuous random variable.

This initially seems strange.

Suppose human height is modeled continuously. The probability that a randomly selected person's height is **exactly**:

```math
170.000000000\ldots\text{ cm}
```

is zero.

But the probability that their height lies within an interval such as:

```math
169\leq X\leq171
```

can be positive.

It is the **interval**, and therefore the area under the PDF, that carries probability.

So:

```math
\boxed{
\text{PMF: probability at a value}
}
```

whereas:

```math
\boxed{
\text{PDF: probability density around values}
}
```

---

## 14. Cumulative Distribution Function — CDF

There is another way to describe a random variable that works for **both discrete and continuous variables**.

The **Cumulative Distribution Function (CDF)** is defined as:

```math
F(x)=P(X\leq x)
```

It answers:

> What is the probability that the random variable takes a value less than or equal to $x$?

The word **cumulative** is the key.

It accumulates probability from the left up to $x$.

---

### 14.1. CDF for a Discrete Variable

Return to the two-coin example.

The PMF is:

| $x$ | $P(X=x)$ |
|---:|---:|
| 0 | $0.25$ |
| 1 | $0.50$ |
| 2 | $0.25$ |

Then:

```math
F(0)=P(X\leq0)=0.25
```

```math
F(1)=P(X\leq1)=0.25+0.50=0.75
```

```math
F(2)=P(X\leq2)=1
```

So the CDF accumulates the probability mass as $x$ increases.

---

### 14.2. CDF for a Continuous Variable

For a continuous random variable:

```math
F(x)
=
\int_{-\infty}^{x}f(t)\,dt
```

Thus the PDF and CDF are closely related.

The CDF accumulates the PDF.

From calculus:

```math
\frac{d}{dx}F(x)=f(x)
```

So:

```math
\boxed{
\text{PDF}
\xrightarrow{\text{integration}}
\text{CDF}
}
```

and:

```math
\boxed{
\text{CDF}
\xrightarrow{\text{differentiation}}
\text{PDF}
}
```

This is a direct application of the Fundamental Theorem of Calculus.

---

### 14.3. Using the CDF to Calculate Probabilities

Because:

```math
F(x)=P(X\leq x)
```

the probability of lying between two values can be obtained from:

```math
P(a<X\leq b)
=
F(b)-F(a)
```

Conceptually:

```math
\text{probability up to }b
-
\text{probability up to }a
```

leaves only the probability between $a$ and $b$.

---

## 15. PMF vs PDF vs CDF

These three terms are easy to mix up, so keep their jobs separate.

| Function | Used for | Meaning |
|---|---|---|
| PMF | Discrete variables | Probability at each possible value |
| PDF | Continuous variables | Probability density |
| CDF | Both | Probability accumulated up to a value |

A compact memory hook:

```math
\boxed{
\begin{aligned}
\text{PMF} &: \text{How much probability is AT }x?\\
\text{PDF} &: \text{How dense is probability AROUND }x?\\
\text{CDF} &: \text{How much probability has accumulated UP TO }x?
\end{aligned}
}
```

---

### 15.1. ML Connection

Probability distributions allow us to move beyond individual observations and describe the **behaviour of an entire variable**.

Instead of merely observing:

```math
x_1,x_2,\ldots,x_n
```

we can ask what underlying probability distribution might have generated those observations.

This idea appears throughout machine learning.

Examples include:

- modeling residual errors in linear regression
- representing class probabilities
- estimating uncertainty
- generating synthetic samples
- probabilistic classification
- anomaly detection
- Bayesian models

A dataset gives us **observations**.

Probability distributions give us a mathematical model of the **process that could have produced those observations**.

```math
\boxed{
\text{Observed data}
\rightarrow
\text{Random variable}
\rightarrow
\text{Probability distribution}
}
```

---

### 15.2. Mental Model

A random variable converts uncertain outcomes into numbers:

```math
\boxed{
\text{Outcome}
\xrightarrow{\text{random variable}}
\text{Number}
}
```

A probability distribution then tells us how likely those numbers are.

For discrete variables:

```math
\boxed{\text{PMF}}
```

For continuous variables:

```math
\boxed{\text{PDF}}
```

For accumulated probability in either case:

```math
\boxed{\text{CDF}}
```

Once probability has been distributed over possible numerical values, we can ask another fundamental question:

> **If this random process were repeated many times, what value should we expect on average?**

That leads to **expected value, variance, and standard deviation**.

---

## 16. Expected Value

The **expected value** of a random variable represents its probability-weighted long-run average.

It is usually written as:

```math
E[X]
```

or:

```math
\mathbb{E}[X]
```

The expected value does **not** necessarily represent a value that must actually occur.

Instead, it describes the average value we would expect if the random experiment were repeated a very large number of times.

---

### 16.1. Expected Value of a Discrete Random Variable

For a discrete random variable:

```math
\mathbb{E}[X]
=
\sum_x xP(X=x)
```

Each possible value is multiplied by its probability, and the results are added.

Consider a fair six-sided die.

The possible values are:

```math
1,2,3,4,5,6
```

Each occurs with probability:

```math
\frac{1}{6}
```

Therefore:

```math
\mathbb{E}[X]
=
1\left(\frac{1}{6}\right)
+
2\left(\frac{1}{6}\right)
+
3\left(\frac{1}{6}\right)
+
4\left(\frac{1}{6}\right)
+
5\left(\frac{1}{6}\right)
+
6\left(\frac{1}{6}\right)
```

which gives:

```math
\mathbb{E}[X]
=
\frac{21}{6}
=
3.5
```

A die can never actually produce $3.5$.

That is perfectly acceptable.

The expected value is not necessarily a possible outcome. It is the **long-run average outcome**.

---

### 16.2. Expected Value of a Continuous Random Variable

For a continuous random variable, summation becomes integration:

```math
\mathbb{E}[X]
=
\int_{-\infty}^{\infty}x f(x)\,dx
```

where $f(x)$ is the probability density function.

This is the continuous counterpart of:

```math
\sum_x xP(X=x)
```

The principle remains identical:

> Multiply each possible value by how much probability is associated with it, then accumulate across all possible values.

---

### 16.3. Expected Value as the Mean

The expected value of a probability distribution is also called its **population mean**:

```math
\mu=\mathbb{E}[X]
```

This connects probability with the familiar statistical concept of the mean.

There is, however, an important distinction.

```math
\mu
```

describes the theoretical mean of the underlying population or probability distribution.

Whereas:

```math
\bar{x}
```

usually represents the mean calculated from an observed sample.

This distinction becomes important in statistics:

```math
\boxed{
\text{Population parameter: }\mu
\qquad
\text{Sample statistic: }\bar{x}
}
```

The sample mean is used to **estimate** the unknown population mean.

---

### 16.4. Linearity of Expectation

Expected values have a very useful property.

For random variables $X$ and $Y$:

```math
\mathbb{E}[X+Y]
=
\mathbb{E}[X]+\mathbb{E}[Y]
```

More generally:

```math
\mathbb{E}[aX+bY]
=
a\mathbb{E}[X]+b\mathbb{E}[Y]
```

This property holds even when $X$ and $Y$ are not independent.

This makes expectation particularly convenient mathematically and is one reason it appears constantly in probability, statistics, and machine learning.

---

## 17. Variance

Expected value tells us where a distribution is **centered**.

It does not tell us how widely values are spread around that center.

Consider two variables:

```math
X=\{49,50,51\}
```

and:

```math
Y=\{0,50,100\}
```

Both may have the same mean:

```math
\mu=50
```

but their spreads are obviously very different.

We therefore need another quantity.

The **variance** measures the expected squared distance of a random variable from its mean.

```math
\mathrm{Var}(X)
=
\mathbb{E}\left[(X-\mu)^2\right]
```

Since:

```math
\mu=\mathbb{E}[X]
```

we can also write:

```math
\boxed{
\mathrm{Var}(X)
=
\mathbb{E}
\left[
(X-\mathbb{E}[X])^2
\right]
}
```

Variance is commonly represented by:

```math
\sigma^2
```

Therefore:

```math
\sigma^2
=
\mathrm{Var}(X)
```

---

### 17.1. Why Square the Deviations?

A natural first thought might be to calculate:

```math
X-\mu
```

and average these deviations.

But positive and negative deviations cancel each other.

For example:

```math
(-2)+(-1)+0+1+2=0
```

even though the observations clearly have some spread.

Squaring solves this problem:

```math
(X-\mu)^2
```

All deviations become non-negative, and larger deviations receive greater weight.

This squared-error idea should already look familiar.

> **ML connection:** Mean Squared Error uses exactly the same mathematical mechanism — square deviations so that positive and negative errors cannot cancel and large errors are penalized more heavily.

---

### 17.2. Alternative Variance Formula

Expanding the variance expression gives a useful identity:

```math
\boxed{
\mathrm{Var}(X)
=
\mathbb{E}[X^2]
-
(\mathbb{E}[X])^2
}
```

In words:

> Variance = expected square minus square of expected value.

This form is frequently useful in probability derivations.

---

## 18. Standard Deviation

Variance has one inconvenience.

Because deviations are squared, its units are also squared.

If $X$ is measured in kilograms, variance is measured in:

```math
kg^2
```

To return to the original units, we take the square root.

The **standard deviation** is:

```math
\boxed{
\sigma
=
\sqrt{\mathrm{Var}(X)}
}
```

Thus:

```math
\text{Variance}
\rightarrow
\text{squared spread}
```

while:

```math
\text{Standard deviation}
\rightarrow
\text{spread in the original units}
```

A larger standard deviation means observations tend to lie farther from the mean.

A smaller standard deviation means observations tend to cluster more closely around it.

---

### 18.1. Mean and Standard Deviation Together

Two quantities provide a useful first description of many distributions:

```math
\boxed{
\mu=\text{location}
\qquad
\sigma=\text{spread}
}
```

This becomes particularly important when we encounter the Gaussian distribution.

Its shape can be described entirely using:

```math
\mu
```

and:

```math
\sigma^2
```

---

## 19. Covariance

Variance describes how **one variable varies around its own mean**.

Covariance extends this idea to **two variables**.

The covariance between random variables $X$ and $Y$ is:

```math
\boxed{
\mathrm{Cov}(X,Y)
=
\mathbb{E}
\left[
(X-\mu_X)(Y-\mu_Y)
\right]
}
```

where:

```math
\mu_X=\mathbb{E}[X]
```

and:

```math
\mu_Y=\mathbb{E}[Y]
```

Covariance asks:

> When $X$ moves away from its mean, does $Y$ tend to move in the same direction or the opposite direction?

---

### 19.1. Positive Covariance

If larger values of $X$ tend to occur with larger values of $Y$:

```math
\mathrm{Cov}(X,Y)>0
```

For example:

```math
\text{height} \uparrow
\qquad
\text{weight} \uparrow
```

may produce positive covariance.

Both variables tend to move together.

---

### 19.2. Negative Covariance

If larger values of $X$ tend to occur with smaller values of $Y$:

```math
\mathrm{Cov}(X,Y)<0
```

For example:

```math
\text{product price} \uparrow
\qquad
\text{quantity demanded} \downarrow
```

may produce negative covariance.

The variables tend to move in opposite directions.

---

### 19.3. Zero Covariance

If:

```math
\mathrm{Cov}(X,Y)=0
```

there is no **linear covariance relationship** between the variables.

However, this requires an important warning:

> **Zero covariance does not necessarily mean independence.**

Two variables can have a strong nonlinear relationship while having zero covariance.

For example, suppose:

```math
Y=X^2
```

and $X$ is distributed symmetrically around zero.

Clearly, $Y$ depends completely on $X$.

Yet positive and negative values of $X$ can cancel in the covariance calculation, potentially producing:

```math
\mathrm{Cov}(X,Y)=0
```

So:

```math
\boxed{
\text{Independence}
\Rightarrow
\text{zero covariance}
}
```

under ordinary finite-moment conditions, but in general:

```math
\boxed{
\text{zero covariance}
\not\Rightarrow
\text{independence}
}
```

This distinction becomes important when working with nonlinear relationships in ML.

---

### 19.4. Variance Is Covariance with Itself

There is an elegant connection:

```math
\mathrm{Cov}(X,X)
=
\mathbb{E}
\left[
(X-\mu_X)^2
\right]
```

But this is exactly the definition of variance.

Therefore:

```math
\boxed{
\mathrm{Cov}(X,X)
=
\mathrm{Var}(X)
}
```

So variance is really a special case of covariance.

Variance asks:

> How does $X$ vary with itself?

Covariance asks:

> How do $X$ and $Y$ vary together?

---

## 20. Covariance Matrix

Machine learning datasets rarely contain only two variables.

Suppose we have features:

```math
X_1,X_2,\ldots,X_n
```

We may want to know how every feature varies with every other feature.

These relationships can be arranged into a **covariance matrix**:

```math
\Sigma=
\begin{bmatrix}
\mathrm{Var}(X_1) &
\mathrm{Cov}(X_1,X_2) &
\cdots \\
\mathrm{Cov}(X_2,X_1) &
\mathrm{Var}(X_2) &
\cdots \\
\vdots &
\vdots &
\ddots
\end{bmatrix}
```

Notice the diagonal:

```math
\mathrm{Cov}(X_i,X_i)
=
\mathrm{Var}(X_i)
```

Therefore, the diagonal of a covariance matrix contains the **variances of the individual variables**.

The off-diagonal elements contain their pairwise covariances.

Also:

```math
\mathrm{Cov}(X,Y)
=
\mathrm{Cov}(Y,X)
```

so the covariance matrix is symmetric:

```math
\boxed{
\Sigma=\Sigma^T
}
```

This reconnects probability directly with linear algebra.

> **ML connection:** Covariance matrices appear in PCA, multivariate Gaussian distributions, dimensionality reduction, feature analysis, probabilistic models, and many other ML techniques.

---

## 21. Covariance vs Correlation

Covariance tells us the **direction** of a linear relationship, but its magnitude depends on the units and scales of the variables.

Correlation removes this scale dependence by normalizing covariance.

The Pearson correlation coefficient is:

```math
\boxed{
\rho_{X,Y}
=
\frac{
\mathrm{Cov}(X,Y)
}{
\sigma_X\sigma_Y
}
}
```

Because of this normalization:

```math
-1\leq\rho_{X,Y}\leq1
```

Interpretation:

```math
\rho\approx1
```

indicates a strong positive linear relationship.

```math
\rho\approx-1
```

indicates a strong negative linear relationship.

```math
\rho\approx0
```

indicates little or no linear relationship.

Again, the word **linear** matters.

A correlation close to zero does not prove that two variables are unrelated.

---

### 21.1. ML Connection

These quantities appear everywhere in machine learning.

**Expectation** gives us the average behaviour of a random quantity:

```math
\mathbb{E}[X]
```

**Variance** describes uncertainty or spread:

```math
\mathrm{Var}(X)
```

**Covariance** describes how two quantities vary together:

```math
\mathrm{Cov}(X,Y)
```

**Correlation** gives a scale-independent measure of their linear relationship:

```math
\rho_{X,Y}
```

Together:

```math
\boxed{
\text{Expectation}
\rightarrow
\text{centre}
}
```

```math
\boxed{
\text{Variance}
\rightarrow
\text{individual spread}
}
```

```math
\boxed{
\text{Covariance}
\rightarrow
\text{joint movement}
}
```

```math
\boxed{
\text{Correlation}
\rightarrow
\text{standardized linear relationship}
}
```

These concepts provide the mathematical vocabulary for describing how data is distributed and how variables relate to one another.

---

### 21.2. Mental Model

We can now describe much more than whether an event occurs.

Starting with a random variable $X$:

```math
\boxed{
\text{Distribution}
\rightarrow
\begin{cases}
\mathbb{E}[X] & \text{Where is it centered?}\\
\mathrm{Var}(X) & \text{How widely does it spread?}
\end{cases}
}
```

With two variables:

```math
\boxed{
(X,Y)
\rightarrow
\mathrm{Cov}(X,Y)
\rightarrow
\text{How do they move together?}
}
```

And with many variables:

```math
\boxed{
X_1,\ldots,X_n
\rightarrow
\Sigma
\rightarrow
\text{Covariance matrix}
}
```

We now have the machinery needed to study specific **probability distributions** — mathematical patterns describing how probability is allocated across possible values of a random variable.

---

## 22. Probability Distributions

A **probability distribution** describes how probability is distributed across the possible values of a random variable.

Different random processes produce different characteristic patterns.

For example:

- a single yes/no outcome naturally leads to a **Bernoulli distribution**
- repeated yes/no trials lead to a **Binomial distribution**
- many continuous measurements approximately follow a **Gaussian distribution**
- counts of events occurring over an interval can often be modeled using a **Poisson distribution**

Choosing a probability distribution means making a mathematical assumption about **how the random variable behaves**.

> **ML connection:** Many machine learning algorithms contain implicit or explicit assumptions about the probability distributions generating their data or errors. Understanding these distributions helps explain why particular loss functions and models have the forms they do.

---

## 23. Bernoulli Distribution

The **Bernoulli distribution** models a single experiment having exactly two possible outcomes.

Conventionally:

```math
X \in \{0,1\}
```

where:

```math
X=1
```

represents success and:

```math
X=0
```

represents failure.

The words *success* and *failure* are mathematical labels; they do not imply anything desirable or undesirable.

Examples include:

- clicked / did not click
- spam / not spam
- fraud / legitimate
- churn / remain
- disease / no disease

If the probability of success is $p$:

```math
P(X=1)=p
```

then:

```math
P(X=0)=1-p
```

The Bernoulli PMF can be compactly written as:

```math
P(X=x)
=
p^x(1-p)^{1-x},
\qquad x\in\{0,1\}
```

Check both possibilities.

If:

```math
x=1
```

then:

```math
P(X=1)
=
p^1(1-p)^0
=
p
```

If:

```math
x=0
```

then:

```math
P(X=0)
=
p^0(1-p)^1
=
1-p
```

---

### 23.1. Mean and Variance

For a Bernoulli random variable:

```math
\boxed{
\mathbb{E}[X]=p
}
```

and:

```math
\boxed{
\mathrm{Var}(X)=p(1-p)
}
```

The expected value has a particularly intuitive meaning.

If:

```math
p=0.7
```

then over many Bernoulli trials, approximately 70% of the observations are expected to be $1$.

---

### 23.2. Bernoulli Distribution and Binary Classification

Suppose the target in a binary classification problem is:

```math
Y\in\{0,1\}
```

A model may estimate:

```math
P(Y=1\mid X)=p
```

Then:

```math
P(Y=0\mid X)=1-p
```

The target can therefore be modeled as a Bernoulli random variable conditioned on the features:

```math
Y\mid X
\sim
\mathrm{Bernoulli}(p)
```

This is precisely the probabilistic structure underlying **logistic regression**.

There is an even deeper connection.

The Bernoulli likelihood eventually leads to the **binary cross-entropy / log-loss** function used in binary classification.

So the familiar classification loss:

```math
-\left[
y\log(p)
+
(1-y)\log(1-p)
\right]
```

is not an arbitrary formula.

It emerges naturally from the probability model assumed for a binary target.

> **ML connection:** Binary classification, logistic regression, sigmoid outputs, Bernoulli probability and binary cross-entropy are all parts of the same mathematical story.

---

## 24. Binomial Distribution

A Bernoulli distribution describes **one** binary trial.

The **Binomial distribution** describes the number of successes across multiple independent Bernoulli trials.

Suppose an experiment is repeated $n$ times.

Each trial:

- has two possible outcomes
- has probability of success $p$
- is independent of the other trials

Let $X$ represent the number of successes.

Then:

```math
X
\sim
\mathrm{Binomial}(n,p)
```

The probability of obtaining exactly $k$ successes is:

```math
\boxed{
P(X=k)
=
\binom{n}{k}
p^k
(1-p)^{n-k}
}
```

where:

```math
\binom{n}{k}
=
\frac{n!}{k!(n-k)!}
```

counts the number of different ways in which $k$ successes can occur among $n$ trials.

---

### 24.1. Example

Suppose a fair coin is tossed 10 times.

Then:

```math
n=10
```

and:

```math
p=0.5
```

The probability of obtaining exactly 6 heads is:

```math
P(X=6)
=
\binom{10}{6}
(0.5)^6
(0.5)^4
```

Since:

```math
\binom{10}{6}=210
```

we obtain:

```math
P(X=6)
=
210(0.5)^{10}
```

approximately:

```math
P(X=6)\approx0.205
```

So there is about a 20.5% probability of obtaining exactly six heads in ten fair coin tosses.

---

### 24.2. Mean and Variance

For:

```math
X\sim\mathrm{Binomial}(n,p)
```

the expected value is:

```math
\boxed{
\mathbb{E}[X]=np
}
```

and the variance is:

```math
\boxed{
\mathrm{Var}(X)=np(1-p)
}
```

This makes intuitive sense.

If 100 independent trials each have a success probability of $0.7$:

```math
\mathbb{E}[X]
=
100(0.7)
=
70
```

We expect approximately 70 successes.

---

### 24.3. Bernoulli vs Binomial

The distinction is simple:

```math
\boxed{
\text{Bernoulli}
=
\text{one binary trial}
}
```

```math
\boxed{
\text{Binomial}
=
\text{number of successes across }n\text{ binary trials}
}
```

In fact, Bernoulli can be viewed as the special case:

```math
n=1
```

of the Binomial distribution.

---

## 25. Gaussian or Normal Distribution

The **Gaussian distribution**, also called the **Normal distribution**, is one of the most important probability distributions in statistics and machine learning.

It has the familiar bell-shaped curve.

A Gaussian random variable is written:

```math
X
\sim
\mathcal{N}(\mu,\sigma^2)
```

where:

```math
\mu
```

controls the **centre** of the distribution and:

```math
\sigma^2
```

controls its **spread**.

The probability density function is:

```math
\boxed{
f(x)
=
\frac{1}
{\sigma\sqrt{2\pi}}
\exp
\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
}
```

The formula looks intimidating, but its structure is much more important than memorising it.

Look at its central term:

```math
(x-\mu)^2
```

It measures the squared distance from the mean.

As $x$ moves farther from $\mu$, this squared distance increases.

Because it appears inside a negative exponential:

```math
\exp
\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
```

values far from the mean receive progressively lower density.

Thus:

```math
\boxed{
\text{farther from mean}
\rightarrow
\text{lower probability density}
}
```

This produces the characteristic bell shape.

---

### 25.1. Role of the Mean

The mean determines where the Gaussian is centered.

Changing:

```math
\mu
```

moves the entire distribution left or right.

For a Gaussian distribution:

```math
\text{mean}
=
\text{median}
=
\text{mode}
```

because the distribution is symmetric around its centre.

---

### 25.2. Role of the Variance

The variance determines how widely the Gaussian is spread.

Small:

```math
\sigma^2
```

produces a narrow distribution concentrated around the mean.

Large:

```math
\sigma^2
```

produces a wider distribution.

Thus:

```math
\boxed{
\mu
\rightarrow
\text{location}
}
```

```math
\boxed{
\sigma^2
\rightarrow
\text{spread}
}
```

---

### 25.3. The 68–95–99.7 Rule

For a Gaussian distribution, approximately:

```math
68\%
```

of observations lie within one standard deviation of the mean:

```math
\mu-\sigma
\leq
X
\leq
\mu+\sigma
```

Approximately:

```math
95\%
```

lie within two standard deviations:

```math
\mu-2\sigma
\leq
X
\leq
\mu+2\sigma
```

And approximately:

```math
99.7\%
```

lie within three standard deviations:

```math
\mu-3\sigma
\leq
X
\leq
\mu+3\sigma
```

This is commonly remembered as:

```math
\boxed{
68\%-95\%-99.7\%
}
```

It gives standard deviation an immediate probabilistic interpretation when the data is approximately Gaussian.

---

## 26. Why the Gaussian Distribution Matters So Much

The Gaussian distribution appears throughout statistics and ML for several reasons.

### 26.1. Many Natural Measurements Are Approximately Gaussian

Quantities influenced by many small independent factors often become approximately normally distributed.

Examples can include:

- measurement errors
- biological measurements
- manufacturing variations
- aggregated noise

Not every real-world variable is Gaussian, and assuming normality blindly is a mistake.

But the Gaussian is common enough to be exceptionally useful.

---

### 26.2. The Central Limit Theorem

One major reason for the importance of the Gaussian distribution is the **Central Limit Theorem (CLT)**.

Very roughly:

> Under broad conditions, the distribution of sample means approaches a Gaussian distribution as the sample size becomes sufficiently large, even when the original population itself is not Gaussian.

Symbolically:

```math
\bar{X}
\approx
\mathcal{N}
\left(
\mu,
\frac{\sigma^2}{n}
\right)
```

for sufficiently large $n$, under suitable assumptions.

We will treat the CLT properly in the Statistics chapter.

For now, the important idea is:

```math
\boxed{
\text{Averages of many random observations}
\rightarrow
\text{often approximately Gaussian}
}
```

This is one reason the Normal distribution sits at the heart of statistical inference.

---

### 26.3. Gaussian Noise and Regression

Suppose a linear regression model is:

```math
y
=
\beta_0
+
\beta_1x_1
+
\cdots
+
\beta_nx_n
+
\varepsilon
```

A common assumption is:

```math
\varepsilon
\sim
\mathcal{N}(0,\sigma^2)
```

This means the unexplained error is assumed to be normally distributed around zero.

The Gaussian density contains:

```math
(x-\mu)^2
```

When Gaussian error assumptions are combined with **maximum likelihood estimation**, maximizing the probability of the observed data turns out to be equivalent to minimizing the **sum of squared errors**.

This gives us a profound connection:

```math
\boxed{
\text{Gaussian error assumption}
\rightarrow
\text{maximum likelihood}
\rightarrow
\text{squared-error minimization}
}
```

So once again, Mean Squared Error is not merely a convenient formula chosen by somebody designing an algorithm.

It has a probabilistic foundation.

> **ML connection:** The familiar squared-error loss of linear regression can be derived by assuming that regression errors follow a Gaussian distribution.

---

## 27. Standard Normal Distribution and Z-Scores

A special Gaussian distribution has:

```math
\mu=0
```

and:

```math
\sigma=1
```

It is called the **standard normal distribution**:

```math
Z
\sim
\mathcal{N}(0,1)
```

Any normally distributed value can be converted into this standardized scale using the **z-score**:

```math
\boxed{
z
=
\frac{x-\mu}{\sigma}
}
```

The z-score tells us how many standard deviations an observation lies from the mean.

For example:

```math
z=2
```

means the observation lies two standard deviations above the mean.

Similarly:

```math
z=-1.5
```

means it lies 1.5 standard deviations below the mean.

This idea is closely related to feature standardization in machine learning:

```math
x'
=
\frac{x-\mu}{\sigma}
```

After standardization, a feature typically has approximately:

```math
\mu=0
```

and:

```math
\sigma=1
```

> **ML connection:** StandardScaler-style feature scaling is mathematically the same centering-and-scaling operation used to calculate z-scores, although feature standardization does **not** imply that the resulting data becomes Gaussian.

That last distinction matters.

Standardization changes the **location and scale** of a distribution.

It does not magically change its **shape**.

---

## 28. Poisson Distribution

The **Poisson distribution** models the number of events occurring within a fixed interval of time, space, area, or another exposure measure when events occur independently at an approximately constant average rate.

Examples include:

- number of calls arriving per minute
- number of website requests per second
- number of defects per manufactured unit
- number of failures during a time interval

Let:

```math
X
\sim
\mathrm{Poisson}(\lambda)
```

where:

```math
\lambda
```

represents the expected number of events within the interval.

The PMF is:

```math
P(X=k)
=
\frac{
e^{-\lambda}\lambda^k
}{
k!
}
```

for:

```math
k=0,1,2,\ldots
```

A useful property of the Poisson distribution is:

```math
\boxed{
\mathbb{E}[X]=\lambda
}
```

and:

```math
\boxed{
\mathrm{Var}(X)=\lambda
}
```

Thus its mean and variance are equal.

For our ML foundations, the formula itself is less important than recognizing the pattern:

```math
\boxed{
\text{Poisson}
\rightarrow
\text{count of events within an interval}
}
```

Poisson-based models appear in count-data modeling, event prediction, reliability analysis, traffic modeling, and related statistical ML problems.

---

## 29. Choosing the Distribution by the Question

A useful way to distinguish these distributions is to ask what kind of random quantity is being modeled.

| Question | Typical Distribution |
|---|---|
| Did one binary event happen? | Bernoulli |
| How many successes occurred in $n$ binary trials? | Binomial |
| How is a continuous quantity distributed around a mean? | Gaussian |
| How many events occurred within an interval? | Poisson |

The corresponding mental map is:

```math
\boxed{
\begin{aligned}
\text{Bernoulli} &\rightarrow \text{one yes/no outcome}\\
\text{Binomial} &\rightarrow \text{number of successes}\\
\text{Gaussian} &\rightarrow \text{continuous variation around a mean}\\
\text{Poisson} &\rightarrow \text{number of events in an interval}
\end{aligned}
}
```

---

### 29.1. Distribution Assumptions Matter

A probability distribution is not merely a curve fitted to data.

Choosing a distribution imposes assumptions about how the data behaves.

For example, saying:

```math
Y\sim\mathrm{Bernoulli}(p)
```

means $Y$ has two possible outcomes.

Saying:

```math
X\sim\mathcal{N}(\mu,\sigma^2)
```

means we are modeling $X$ as continuous, symmetric around its mean, with Gaussian-shaped tails.

These assumptions influence the mathematics of the model.

This gives us an important ML principle:

> **The loss function, model structure, and probability assumptions are often different views of the same underlying mathematical model.**

For example:

```math
\boxed{
\text{Bernoulli target}
\rightarrow
\text{logistic regression}
\rightarrow
\text{binary cross-entropy}
}
```

while:

```math
\boxed{
\text{Gaussian errors}
\rightarrow
\text{linear regression}
\rightarrow
\text{squared-error loss}
}
```

These connections are worth remembering because they turn apparently unrelated ML formulas into consequences of a common probabilistic framework.

---

### 29.2. Mental Model

Probability distributions provide mathematical models for uncertainty:

```math
\boxed{
\text{Random variable}
+
\text{distribution}
=
\text{mathematical model of uncertainty}
}
```

The distribution tells us:

- which values are possible
- which values are more likely
- where values tend to concentrate
- how widely they vary

And once a distribution is assumed, observed data can tell us something about the unknown parameters of that distribution.

That brings probability directly to one of the central ideas of machine learning:

```math
\boxed{
\text{Observed data}
\rightarrow
\text{Likelihood}
\rightarrow
\text{Estimate model parameters}
}
```

This is the foundation of **maximum likelihood estimation**.

---

## 30. Joint, Marginal, and Conditional Probability Distributions

Machine learning rarely deals with one random variable in isolation.

A dataset usually contains many variables:

```math
X_1,X_2,\ldots,X_n,Y
```

and we are interested not only in their individual distributions, but also in **how they behave together**.

Three related ideas become important:

- joint distributions
- marginal distributions
- conditional distributions

---

### 30.1. Joint Distribution

A **joint probability distribution** describes the probability behaviour of two or more random variables simultaneously.

For two discrete random variables $X$ and $Y$:

```math
P(X=x,Y=y)
```

asks:

> What is the probability that $X=x$ **and** $Y=y$?

For example, suppose:

- $X$ = customer device type
- $Y$ = whether the customer purchases

Then:

```math
P(X=\text{mobile},Y=\text{purchase})
```

describes the probability of observing a customer who is both a mobile user **and** a purchaser.

The joint distribution contains information about the variables **together**.

Conceptually:

```math
\boxed{
\text{Joint}
\rightarrow
\text{What happens to }X\text{ AND }Y\text{ together?}
}
```

---

### 30.2. Marginal Distribution

Suppose we know the joint distribution of $X$ and $Y$, but only care about $X$.

We can remove $Y$ by summing over all its possible values:

```math
P(X=x)
=
\sum_y P(X=x,Y=y)
```

This produces the **marginal distribution** of $X$.

For continuous variables, summation becomes integration:

```math
f_X(x)
=
\int_{-\infty}^{\infty}
f_{X,Y}(x,y)\,dy
```

The idea is more important than the notation:

> A marginal distribution describes one variable after the other variables have been summed or integrated away.

Conceptually:

```math
\boxed{
\text{Joint distribution}
\xrightarrow{\text{remove }Y}
\text{Marginal distribution of }X
}
```

---

### 30.3. Conditional Distribution

A **conditional distribution** describes the distribution of one random variable when another variable is known.

For example:

```math
P(Y\mid X)
```

describes how the probability distribution of $Y$ changes after $X$ is observed.

For discrete variables:

```math
P(Y=y\mid X=x)
=
\frac{
P(X=x,Y=y)
}{
P(X=x)
}
```

This is simply the conditional probability rule applied to random variables.

And this expression should now look extremely familiar:

```math
\boxed{
P(Y\mid X)
}
```

because supervised machine learning is often fundamentally interested in exactly this quantity.

Given features $X$, what can we infer about target $Y$?

---

### 30.4. Joint, Marginal, and Conditional — One Mental Model

These three concepts can be remembered as:

```math
\boxed{
\begin{aligned}
P(X,Y) &\rightarrow \text{X and Y together}\\
P(X) &\rightarrow \text{X regardless of Y}\\
P(Y\mid X) &\rightarrow \text{Y after X is known}
\end{aligned}
}
```

They are different views of the same underlying probabilistic system.

> **ML connection:** Supervised learning commonly attempts to learn the relationship between $X$ and $Y$, often directly or indirectly approximating the conditional distribution $P(Y\mid X)$.

---

## 31. Probability vs Likelihood

Probability and likelihood use closely related mathematics, but they ask different questions.

This distinction is extremely important in machine learning.

Suppose a probability model has parameter $\theta$:

```math
P(X\mid\theta)
```

### 31.1. Probability

When the parameter $\theta$ is known and the data $X$ is unknown, we ask:

> Given this model, how probable is a particular observation?

Conceptually:

```math
\boxed{
\text{Known model}
\rightarrow
\text{probability of possible data}
}
```

For example, if a coin is known to have:

```math
p=0.7
```

we can calculate the probability of observing heads.

Here the model parameter is fixed, while the outcome is uncertain.

---

### 31.2. Likelihood

In machine learning, the situation is often reversed.

We have already observed the data:

```math
x_1,x_2,\ldots,x_n
```

but do not know the best model parameter:

```math
\theta
```

We therefore treat the observed data as fixed and ask:

> Which value of $\theta$ makes the data we actually observed most plausible?

The resulting function is the **likelihood function**:

```math
L(\theta)
=
P(\text{data}\mid\theta)
```

The same mathematical expression is now viewed as a function of the unknown parameter.

This gives the key distinction:

```math
\boxed{
\begin{aligned}
\text{Probability:} &\quad \theta\text{ fixed, data varies}\\
\text{Likelihood:} &\quad \text{data fixed, }\theta\text{ varies}
\end{aligned}
}
```

This is subtle but fundamental.

> **Likelihood is not the probability of the parameter. It measures how well a parameter value explains the observed data.**

---

## 32. Maximum Likelihood Estimation

Suppose we observe training data:

```math
x_1,x_2,\ldots,x_n
```

and assume the observations were generated by a probability distribution controlled by parameter $\theta$.

The likelihood is:

```math
L(\theta)
=
P(x_1,x_2,\ldots,x_n\mid\theta)
```

If the observations are assumed independent:

```math
L(\theta)
=
\prod_{i=1}^{n}
P(x_i\mid\theta)
```

Different values of $\theta$ produce different likelihoods.

**Maximum Likelihood Estimation (MLE)** chooses the parameter that makes the observed training data most likely:

```math
\boxed{
\hat{\theta}_{MLE}
=
\arg\max_{\theta}
L(\theta)
}
```

The notation:

```math
\arg\max
```

means:

> Return the **argument**, or parameter value, that maximizes the function.

Thus:

```math
\max_\theta L(\theta)
```

asks for the maximum likelihood value itself, whereas:

```math
\arg\max_\theta L(\theta)
```

asks which value of $\theta$ produces that maximum.

In machine learning, we usually want the latter.

We want the **parameters**.

---

### 32.1. A Simple Coin Example

Suppose we toss an unknown coin 10 times and observe:

```math
H,H,H,H,H,H,H,T,T,T
```

There are:

```math
7
```

heads and:

```math
3
```

tails.

Let $p$ represent the unknown probability of heads.

Assuming independent tosses, the likelihood of the observed sequence is:

```math
L(p)
=
p^7(1-p)^3
```

Now consider different possible values of $p$.

If:

```math
p=0.1
```

observing seven heads would be very unlikely.

If:

```math
p=0.5
```

the observations are more plausible.

If:

```math
p=0.7
```

they are even more compatible with what we observed.

MLE asks:

```math
\hat p
=
\arg\max_p
p^7(1-p)^3
```

The maximizing value is:

```math
\boxed{
\hat p=0.7
}
```

So the maximum likelihood estimate of the coin's probability of heads is simply the observed proportion of heads.

This illustrates the basic philosophy:

```math
\boxed{
\text{Choose parameters that make the observed data most plausible}
}
```

That is remarkably close to what we mean when we say a machine learning model **learns from training data**.

---

## 33. Why Log-Likelihood Is Used

Likelihoods frequently involve products:

```math
L(\theta)
=
\prod_{i=1}^{n}
P(x_i\mid\theta)
```

For large datasets, multiplying many small probabilities can produce extremely tiny numbers.

Products are also inconvenient to differentiate.

We therefore take the logarithm:

```math
\ell(\theta)
=
\log L(\theta)
```

Using the logarithm rule:

```math
\log(ab)
=
\log a+\log b
```

the product becomes a sum:

```math
\ell(\theta)
=
\sum_{i=1}^{n}
\log P(x_i\mid\theta)
```

This is the **log-likelihood**.

Because logarithm is strictly increasing, the parameter that maximizes likelihood also maximizes log-likelihood:

```math
\boxed{
\arg\max_\theta L(\theta)
=
\arg\max_\theta \log L(\theta)
}
```

So we gain mathematical convenience without changing the optimum.

This is another place where the logarithms encountered throughout ML acquire a clear purpose.

---

## 34. From Maximum Likelihood to Loss Functions

Machine learning algorithms are usually described as **minimizing a loss function**, while MLE is described as **maximizing likelihood**.

These may sound like different procedures.

They are often the same optimization problem viewed from opposite directions.

If we maximize:

```math
\log L(\theta)
```

we can equivalently minimize its negative:

```math
-\log L(\theta)
```

This is called the **negative log-likelihood (NLL)**.

Therefore:

```math
\boxed{
\text{Maximize likelihood}
\Longleftrightarrow
\text{Maximize log-likelihood}
\Longleftrightarrow
\text{Minimize negative log-likelihood}
}
```

And negative log-likelihood frequently appears in ML under another familiar name:

**loss function**.

---

### 34.1. Bernoulli → Binary Cross-Entropy

For a Bernoulli target:

```math
Y\in\{0,1\}
```

with predicted probability $p$, the probability of observing $y$ is:

```math
P(Y=y)
=
p^y(1-p)^{1-y}
```

Taking the negative logarithm gives:

```math
-\log P(Y=y)
=
-\left[
y\log p
+
(1-y)\log(1-p)
\right]
```

This is exactly **binary cross-entropy**.

Therefore:

```math
\boxed{
\text{Bernoulli model}
\rightarrow
\text{Maximum likelihood}
\rightarrow
\text{Binary cross-entropy}
}
```

This explains why logistic regression uses log-loss.

The loss function is a direct consequence of the probabilistic model.

---

### 34.2. Gaussian → Squared Error

Suppose regression errors are assumed Gaussian:

```math
\varepsilon
\sim
\mathcal{N}(0,\sigma^2)
```

The Gaussian likelihood contains the squared residual:

```math
(y-\hat y)^2
```

When its negative log-likelihood is simplified, the parameter-dependent part becomes proportional to:

```math
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
```

Therefore maximizing Gaussian likelihood is equivalent, under the usual fixed-variance assumption, to minimizing squared error.

So:

```math
\boxed{
\text{Gaussian error model}
\rightarrow
\text{Maximum likelihood}
\rightarrow
\text{Squared-error loss}
}
```

This connects probability directly to linear regression.

---

## 35. Probability as the Mathematical Foundation of Learning

We can now connect the entire chapter.

We began with uncertain outcomes:

```math
\boxed{
\text{Experiment}
\rightarrow
\text{Outcome}
\rightarrow
\text{Event}
\rightarrow
P(A)
}
```

Then we introduced evidence:

```math
\boxed{
P(A\mid B)
}
```

which allowed probabilities to change when new information became available.

Bayes' theorem formalized this update:

```math
\boxed{
\text{Prior}
+
\text{Evidence}
\rightarrow
\text{Posterior}
}
```

Random variables converted uncertain outcomes into numerical quantities:

```math
\boxed{
\text{Outcome}
\rightarrow
\text{Random variable}
\rightarrow
\text{Distribution}
}
```

Expectation and variance summarized those distributions:

```math
\boxed{
\begin{aligned}
\mathbb{E}[X] &\rightarrow \text{centre}\\
\mathrm{Var}(X) &\rightarrow \text{spread}
\end{aligned}
}
```

Covariance described relationships between variables:

```math
\boxed{
\mathrm{Cov}(X,Y)
\rightarrow
\text{joint variation}
}
```

Probability distributions gave us mathematical models for different forms of uncertainty:

```math
\boxed{
\begin{aligned}
\text{Bernoulli} &\rightarrow \text{binary outcome}\\
\text{Binomial} &\rightarrow \text{repeated binary outcomes}\\
\text{Gaussian} &\rightarrow \text{continuous variation}\\
\text{Poisson} &\rightarrow \text{event counts}
\end{aligned}
}
```

Finally, likelihood turned observed data into a mechanism for learning model parameters:

```math
\boxed{
\text{Data}
\rightarrow
\text{Likelihood}
\rightarrow
\text{Optimization}
\rightarrow
\text{Learned parameters}
}
```

---

### 35.1. The Probability → Machine Learning Bridge

The entire connection can be compressed into one chain:

```math
\boxed{
\text{Unknown real-world process}
\rightarrow
\text{Observed data}
\rightarrow
\text{Probability model}
\rightarrow
\text{Likelihood}
\rightarrow
\text{Loss}
\rightarrow
\text{Optimization}
\rightarrow
\text{Learned model}
}
```

This chain ties together much of the mathematics encountered across machine learning.

**Linear regression:**

```math
\text{Gaussian errors}
\rightarrow
\text{squared-error loss}
\rightarrow
\text{optimize coefficients}
```

**Logistic regression:**

```math
\text{Bernoulli target}
\rightarrow
\text{log-loss}
\rightarrow
\text{optimize coefficients}
```

**Bayesian learning:**

```math
\text{prior}
+
\text{likelihood}
\rightarrow
\text{posterior}
```

**Classification:**

```math
X
\rightarrow
P(Y\mid X)
\rightarrow
\text{predicted class}
```

Probability is therefore not merely another mathematical prerequisite for ML.

It provides the language for describing **uncertainty**, while machine learning provides mechanisms for **learning useful patterns within that uncertainty**.

> **ML connection:** A deterministic optimization algorithm may calculate the parameters, but probability often explains what those parameters mean, why a particular loss function is appropriate, and what uncertainty remains in the model's predictions.

---

### 35.2. Final Mental Model

If only one idea from this chapter is retained, let it be this:

```math
\boxed{
\text{Machine learning}
=
\text{learning from finite observations under uncertainty}
}
```

Probability supplies the mathematics of that uncertainty.

Statistics, which comes next, tackles the complementary problem:

> **Given the finite data we actually observed, what can we reliably infer about the larger process that generated it?**

That is the bridge from **Probability** to **Statistics**.
