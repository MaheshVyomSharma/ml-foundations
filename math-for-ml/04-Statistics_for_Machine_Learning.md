# 04. Statistics for Machine Learning

Probability and statistics are closely related, but they approach uncertainty from opposite directions.

**Probability** begins with an assumed model or population and asks:

> If this is how the world behaves, what outcomes should we expect?

**Statistics** begins with observed data and asks:

> Given the outcomes we actually observed, what can we infer about the larger population or process that generated them?

This distinction can be summarized as:

```math
\boxed{
\text{Probability: Population/Model}
\rightarrow
\text{Data}
}
```

```math
\boxed{
\text{Statistics: Data}
\rightarrow
\text{Population/Model}
}
```

Machine learning relies heavily on both directions.

During training, we observe a finite dataset and attempt to learn something about the underlying process:

```math
\text{Training data}
\rightarrow
\text{Learned model}
```

Once the model has been learned, we use it to make predictions about previously unseen observations:

```math
\text{Learned model}
\rightarrow
\text{Predictions}
```

Statistics therefore provides much of the mathematical reasoning behind **learning from finite data and generalizing beyond it**.

> **ML connection:** A machine learning model is trained on a sample but is expected to perform on observations it has never seen. Statistics provides the framework for reasoning about whether conclusions drawn from the sample are likely to generalize to the larger population.

---

## 1.1 Descriptive and Inferential Statistics

Statistics can broadly be divided into two branches:

- descriptive statistics
- inferential statistics

The distinction is fundamental.

### Descriptive Statistics

**Descriptive statistics** summarize and describe the data that has actually been observed.

Suppose we have a dataset containing the salaries of 1,000 employees.

We might calculate:

- mean salary
- median salary
- minimum and maximum
- variance
- standard deviation
- percentiles
- interquartile range
- frequency distributions

These quantities describe the dataset in front of us.

They do not, by themselves, make claims about employees outside that dataset.

Conceptually:

```math
\boxed{
\text{Observed data}
\rightarrow
\text{Summarize observed data}
}
```

Descriptive statistics answers:

> **What does my data look like?**

This is a major part of **Exploratory Data Analysis (EDA)**.

---

### Inferential Statistics

**Inferential statistics** goes further.

It uses a sample of observations to draw conclusions about a larger population.

Conceptually:

```math
\boxed{
\text{Sample}
\rightarrow
\text{Inference}
\rightarrow
\text{Population}
}
```

Examples include:

- estimating a population mean from a sample
- estimating uncertainty around that estimate
- determining whether two groups differ meaningfully
- testing whether an observed relationship could plausibly result from random variation
- estimating unknown population parameters

Inferential statistics therefore asks:

> **What can my observed data tell me about data I have not observed?**

This question lies very close to the fundamental goal of machine learning.

---

### Descriptive vs Inferential Statistics

The distinction can be remembered as:

```math
\boxed{
\begin{aligned}
\text{Descriptive} &\rightarrow \text{Describe what I observed}\\
\text{Inferential} &\rightarrow \text{Generalize beyond what I observed}
\end{aligned}
}
```

Suppose a dataset contains 1,000 customer transactions.

Calculating:

```math
\bar{x} = ₹2{,}450
```

and saying:

> The average transaction value **in this dataset** is ₹2,450.

is descriptive statistics.

Using that sample to estimate:

```math
\mu
```

the average transaction value of **all customers**, is inferential statistics.

The number may initially look similar.

The claim being made is fundamentally different.

---

## 1.2 Population and Sample

Statistical inference depends on distinguishing between a **population** and a **sample**.

### Population

A **population** is the complete collection of observations or entities we ultimately care about.

Examples include:

- every customer of a company
- every manufactured component from a production process
- every possible future transaction
- all patients belonging to a target group
- all emails that a spam classifier may encounter

The population does not necessarily mean every human being or every physically existing object.

It means:

> **The complete set about which we want to make conclusions.**

Suppose we want to predict house prices in Bengaluru.

The population of interest might conceptually be:

```math
\text{all houses belonging to the target housing market}
```

including houses that are not present in our dataset.

---

### Sample

A **sample** is a subset of the population that we actually observe.

If:

```math
N
```

represents population size and:

```math
n
```

represents sample size, then usually:

```math
n < N
```

For example:

```math
\text{Population}
=
\text{all customers}
```

while:

```math
\text{Sample}
=
\text{10,000 customers in our dataset}
```

The central statistical problem is that we usually cannot observe the entire population.

Instead, we observe a sample and attempt to learn something about the population from it.

```math
\boxed{
\text{Population}
\rightarrow
\text{Sample}
\rightarrow
\text{Statistical inference about Population}
}
```

---

### Why Sampling Is Necessary

In many real-world problems, observing the entire population is:

- impossible
- expensive
- slow
- continuously changing
- conceptually infinite

Consider a machine learning model predicting whether a customer will churn.

The real population of interest may include **future customers and future behaviour that do not yet exist**.

No dataset can contain all future observations.

We therefore learn from a finite sample and hope that the patterns found within it represent the underlying population sufficiently well.

This introduces one of the central problems of statistics and ML:

> **How representative is the sample of the population we care about?**

---

## 1.3 Parameters and Statistics

Population quantities and sample quantities are given different names.

A numerical quantity describing a **population** is called a **parameter**.

A numerical quantity calculated from a **sample** is called a **statistic**.

For example:

| Population Parameter | Sample Statistic |
|---|---|
| Mean $\mu$ | Mean $\bar{x}$ |
| Variance $\sigma^2$ | Variance $s^2$ |
| Standard deviation $\sigma$ | Standard deviation $s$ |
| Proportion $p$ | Proportion $\hat{p}$ |

The distinction is important because population parameters are often **unknown**.

We calculate sample statistics in order to estimate them.

For example:

```math
\bar{x}
\rightarrow
\text{estimate of }
\mu
```

Similarly:

```math
s^2
\rightarrow
\text{estimate of }
\sigma^2
```

and:

```math
\hat{p}
\rightarrow
\text{estimate of }
p
```

This creates the basic structure of statistical estimation:

```math
\boxed{
\text{Unknown population parameter}
\leftarrow
\text{estimated using sample statistic}
}
```

---

### A Concrete Example

Suppose we want to know the average amount spent by customers of an online store.

The true population mean is:

```math
\mu
```

But we cannot observe every present and future customer.

Instead, we sample \(n\) customers and calculate:

```math
\bar{x}
=
\frac{1}{n}
\sum_{i=1}^{n}x_i
```

We then use:

```math
\bar{x}
```

as an estimate of:

```math
\mu
```

But another random sample would probably produce a slightly different value of:

```math
\bar{x}
```

That variation between samples is not a mistake.

It is a fundamental consequence of sampling.

And understanding that variation will become one of the central themes of inferential statistics.

---

## 1.4 Describing the Centre of Data

Before making inferences from a sample, we need ways to summarize its structure.

One of the first questions is:

> Where is the data centered?

The three common measures of central tendency are:

- mean
- median
- mode

---

### Mean

For observations:

```math
x_1,x_2,\ldots,x_n
```

the sample mean is:

```math
\boxed{
\bar{x}
=
\frac{1}{n}
\sum_{i=1}^{n}x_i
}
```

The mean uses every observation in the dataset.

This makes it informative, but also sensitive to extreme values.

Consider:

```math
10,\ 12,\ 13,\ 15,\ 100
```

The mean is:

```math
\bar{x}
=
\frac{10+12+13+15+100}{5}
=
30
```

But four of the five observations lie between 10 and 15.

The single value:

```math
100
```

has pulled the mean strongly upward.

---

### Median

The **median** is the middle observation after the data is sorted.

For:

```math
10,\ 12,\ 13,\ 15,\ 100
```

the median is:

```math
13
```

Unlike the mean, the median is relatively resistant to extreme observations.

This makes it particularly useful for skewed distributions.

For example, variables such as:

- income
- house prices
- transaction values

often contain a small number of extremely large values.

In such cases:

```math
\text{median}
```

may describe a typical observation better than:

```math
\text{mean}
```

---

### Mode

The **mode** is the most frequently occurring value.

For example:

```math
1,\ 2,\ 2,\ 2,\ 3,\ 4
```

has mode:

```math
2
```

Unlike the mean and median, the mode can also be useful for categorical data.

For example, if:

```math
\text{payment method}
=
\{\text{UPI},\text{Card},\text{Cash}\}
```

the mode might be:

```math
\text{UPI}
```

even though calculating a numerical mean would make no sense.

---

### Mean, Median, and Skewness

For a roughly symmetric distribution:

```math
\text{mean}
\approx
\text{median}
\approx
\text{mode}
```

For a **right-skewed** distribution, a long tail extends toward larger values.

Extreme high values tend to pull the mean to the right:

```math
\text{mode}
<
\text{median}
<
\text{mean}
```

For a **left-skewed** distribution, the opposite tendency occurs:

```math
\text{mean}
<
\text{median}
<
\text{mode}
```

These relationships are useful intuition rather than rigid laws for every possible dataset.

The important principle is:

> **The mean is influenced strongly by extreme observations; the median is much more resistant to them.**

---

### ML Connection

Measures of centre are among the first things examined during EDA.

They can reveal:

- skewed features
- unusual values
- inappropriate assumptions about distributions
- potential transformations
- missing-value imputation strategies

For example, replacing missing numerical values with:

```math
\text{mean}
```

can work reasonably for symmetric data.

For heavily skewed data, using:

```math
\text{median}
```

may be more robust.

Thus even elementary descriptive statistics can influence preprocessing decisions made before an ML model is trained.

---

### Mental Model

The distinction developed so far can be compressed into:

```math
\boxed{
\text{Population}
\xrightarrow{\text{sampling}}
\text{Sample}
\xrightarrow{\text{statistics}}
\text{Inference about Population}
}
```

Within the sample:

```math
\boxed{
\begin{aligned}
\text{Mean} &\rightarrow \text{arithmetic centre}\\
\text{Median} &\rightarrow \text{middle observation}\\
\text{Mode} &\rightarrow \text{most frequent observation}
\end{aligned}
}
```

But the centre alone does not describe a dataset.

Two datasets can have exactly the same mean while behaving completely differently.

To understand that difference, we next need to measure **spread, position, and shape**.

---

## 1.5 Measuring Spread

Measures of central tendency tell us where data is centered.

They do not tell us how widely observations are distributed around that centre.

Two datasets can have the same mean but very different spread.

Consider:

```math
A=\{48,49,50,51,52\}
```

and:

```math
B=\{10,30,50,70,90\}
```

Both have mean:

```math
\bar{x}=50
```

but dataset \(B\) is much more dispersed.

Measures of spread quantify this difference.

---

### Range

The **range** is the simplest measure of spread:

```math
\boxed{
\text{Range}
=
x_{\max}
-
x_{\min}
}
```

For:

```math
10,\ 30,\ 50,\ 70,\ 90
```

the range is:

```math
90-10=80
```

The range is easy to calculate, but it depends only on the two most extreme observations.

This makes it very sensitive to outliers.

---

### Sample Variance

For a sample containing:

```math
x_1,x_2,\ldots,x_n
```

the sample variance is:

```math
\boxed{
s^2
=
\frac{
\sum_{i=1}^{n}
(x_i-\bar{x})^2
}{
n-1
}
}
```

This looks very similar to the population variance encountered in Probability.

The important difference is the denominator.

For a population:

```math
\sigma^2
=
\frac{
\sum_{i=1}^{N}
(x_i-\mu)^2
}{
N
}
```

For a sample:

```math
s^2
=
\frac{
\sum_{i=1}^{n}
(x_i-\bar{x})^2
}{
n-1
}
```

The use of:

```math
n-1
```

instead of:

```math
n
```

is called **Bessel's correction**.

---

### Why Divide by \(n-1\)?

When calculating sample variance, the population mean:

```math
\mu
```

is usually unknown.

We estimate it using:

```math
\bar{x}
```

from the same sample.

Because:

```math
\bar{x}
```

is chosen from the sample itself, the observations tend to appear slightly closer to \(\bar{x}\) than they would to the unknown true population mean.

If we divided by:

```math
n
```

the resulting variance would tend to underestimate the population variance.

Using:

```math
n-1
```

corrects this downward bias.

Thus:

```math
\boxed{
s^2
=
\text{approximately unbiased estimator of }
\sigma^2
}
```

The deeper statistical reason comes from **degrees of freedom**, which we will encounter again later.

For now, the practical distinction is:

```math
\boxed{
\begin{aligned}
\text{Population variance} &\rightarrow N\\
\text{Sample variance} &\rightarrow n-1
\end{aligned}
}
```

---

## 1.6 Standard Deviation

The sample standard deviation is:

```math
\boxed{
s=\sqrt{s^2}
}
```

Just as with the population standard deviation, it expresses spread in the original units of the variable.

For example, if salaries are measured in rupees:

```math
s^2
```

has units of:

```math
₹^2
```

while:

```math
s
```

returns to:

```math
₹
```

This makes standard deviation easier to interpret directly.

A larger standard deviation means observations tend to lie farther from the sample mean.

A smaller standard deviation means they tend to cluster more closely around it.

---

### Spread and Model Behaviour

Variation in data matters directly in ML.

Features with very different scales can influence optimization differently.

For example:

```math
\text{age}\approx 20\text{ to }80
```

while:

```math
\text{income}\approx 10^5\text{ to }10^7
```

A scale-sensitive algorithm may be dominated numerically by the larger-magnitude feature.

This motivates standardization:

```math
z
=
\frac{x-\bar{x}}{s}
```

which rescales observations according to how many standard deviations they lie from the sample mean.

> **ML connection:** Standard deviation is not only a descriptive statistic. It is directly involved in feature scaling, anomaly detection, uncertainty estimation, and many statistical learning procedures.

---

## 1.7 Percentiles

A **percentile** describes the relative position of an observation within a dataset.

The \(p\)-th percentile is the value below which approximately \(p\%\) of observations lie.

For example:

```math
P_{90}
```

is the value below which approximately:

```math
90\%
```

of the data lies.

If a student's score is at the 90th percentile, this does **not** mean the student scored 90%.

It means the score is greater than or equal to approximately 90% of the observed scores, depending on the percentile convention used.

Percentiles describe **rank position**, not absolute magnitude.

---

## 1.8 Quartiles

Quartiles divide ordered data into four approximately equal parts.

The three important quartiles are:

```math
Q_1
```

the first quartile, corresponding approximately to the 25th percentile;

```math
Q_2
```

the second quartile, corresponding to the median or 50th percentile;

and:

```math
Q_3
```

the third quartile, corresponding approximately to the 75th percentile.

Thus:

```math
\boxed{
\begin{aligned}
Q_1 &\approx P_{25}\\
Q_2 &= P_{50}=\text{median}\\
Q_3 &\approx P_{75}
\end{aligned}
}
```

Quartiles provide a robust summary of the central portion of a distribution.

---

## 1.9 Interquartile Range

The **Interquartile Range (IQR)** measures the spread of the middle 50% of observations.

It is defined as:

```math
\boxed{
IQR
=
Q_3-Q_1
}
```

Because it ignores the lowest 25% and highest 25% when calculating the spread, the IQR is much less sensitive to extreme values than the ordinary range.

This makes it especially useful for skewed data and outlier detection.

Conceptually:

```math
\boxed{
\text{Range}
\rightarrow
\text{overall spread}
}
```

while:

```math
\boxed{
\text{IQR}
\rightarrow
\text{spread of the central 50\%}
}
```

---

## 1.10 Detecting Outliers with the IQR Rule

A commonly used rule identifies potential outliers using:

```math
1.5\times IQR
```

The lower boundary is:

```math
\boxed{
Q_1-1.5(IQR)
}
```

and the upper boundary is:

```math
\boxed{
Q_3+1.5(IQR)
}
```

Observations outside these limits are commonly flagged as potential outliers.

That is:

```math
x
<
Q_1-1.5(IQR)
```

or:

```math
x
>
Q_3+1.5(IQR)
```

This is the rule used by the conventional box plot.

---

### Example

Suppose:

```math
Q_1=20
```

and:

```math
Q_3=40
```

Then:

```math
IQR=40-20=20
```

The lower boundary is:

```math
20-1.5(20)
=
-10
```

and the upper boundary is:

```math
40+1.5(20)
=
70
```

Therefore observations below:

```math
-10
```

or above:

```math
70
```

would be flagged as potential outliers.

---

### An Outlier Is Not Automatically an Error

This distinction is important.

An outlier may represent:

- a data-entry error
- a measurement error
- an unusual but genuine observation
- a rare event
- a separate subgroup in the population
- exactly the phenomenon we are trying to detect

For example, in fraud detection, unusual transactions may be the most valuable observations in the entire dataset.

Therefore:

> **Outlier detection should trigger investigation, not automatic deletion.**

Removing extreme values blindly can destroy real information.

---

## 1.11 Box Plots

A **box plot** summarizes a distribution using quartiles and the IQR.

Its central box usually spans:

```math
Q_1
\rightarrow
Q_3
```

with a line marking:

```math
Q_2
=
\text{median}
```

The width of the box therefore represents:

```math
IQR
```

The whiskers typically extend to the most extreme observations lying within the \(1.5\times IQR\) boundaries.

Values beyond them are displayed separately as potential outliers.

A box plot therefore gives a compact view of:

- centre
- spread
- skewness
- possible outliers

It is particularly useful for comparing distributions across multiple groups.

> **ML connection:** Box plots are common during EDA for spotting scale differences, skewed features, group differences, and anomalous observations before model training.

---

## 1.12 Shape of a Distribution

Centre and spread still do not completely describe a dataset.

We also care about its **shape**.

Two important shape characteristics are:

- skewness
- kurtosis

---

### Skewness

**Skewness** describes asymmetry in a distribution.

A symmetric distribution has approximately:

```math
\text{skewness}=0
```

A distribution with a long right tail has:

```math
\text{positive skewness}
```

and is called **right-skewed**.

A distribution with a long left tail has:

```math
\text{negative skewness}
```

and is called **left-skewed**.

A useful intuition is:

```math
\boxed{
\text{Skewness}
\rightarrow
\text{direction and degree of asymmetry}
}
```

Right-skewed variables commonly include:

- income
- house prices
- transaction values
- waiting times

These often contain many moderate observations and a smaller number of very large values.

---

### Why Skewness Matters in ML

Strongly skewed features can:

- make the mean unrepresentative
- create extreme values
- affect scale-sensitive algorithms
- violate assumptions of some statistical models
- produce highly uneven residual behaviour

A common transformation for positive right-skewed variables is:

```math
x'
=
\log(1+x)
```

The transformation compresses large values more strongly than small ones and can make the distribution less skewed.

This is exactly why transformations such as:

```math
\log(1+\text{price})
```

often appear in regression problems with highly skewed target variables.

> **ML connection:** Distribution shape can directly influence preprocessing choices, model assumptions, and the loss landscape seen during optimization.

---

## 1.13 Kurtosis

**Kurtosis** describes characteristics of a distribution's tails and the tendency to produce extreme observations.

It is commonly compared with the Gaussian distribution.

A distribution with relatively heavy tails has a greater tendency to produce observations far from the mean.

A lighter-tailed distribution produces extreme observations less frequently.

For ML foundations, the useful mental model is:

```math
\boxed{
\text{Skewness}
\rightarrow
\text{asymmetry}
}
```

```math
\boxed{
\text{Kurtosis}
\rightarrow
\text{tail heaviness / extreme-value tendency}
}
```

The exact numerical definitions are less important here than recognizing what these statistics tell us about data behaviour.

---

## 1.14 Robust vs Non-Robust Statistics

Some statistics are strongly influenced by extreme observations.

These include:

```math
\text{mean}
```

```math
\text{variance}
```

```math
\text{standard deviation}
```

Others are much less affected:

```math
\text{median}
```

```math
IQR
```

These are often called **robust statistics**.

A useful comparison is:

| Sensitive to Outliers | More Robust |
|---|---|
| Mean | Median |
| Variance | IQR |
| Standard deviation | Median absolute deviation |

The choice depends on the structure of the data.

For approximately symmetric data without extreme values, mean and standard deviation may provide excellent summaries.

For strongly skewed or contaminated data, median and IQR may provide a more representative description.

---

### ML Connection

Descriptive statistics provide a first diagnostic layer before any model is trained.

They help answer questions such as:

- Are features on very different scales?
- Is the target heavily skewed?
- Are extreme observations present?
- Is the mean representative?
- Should a feature be transformed?
- Should missing values be imputed using mean or median?
- Are suspicious observations likely errors or valid rare cases?

This makes descriptive statistics part of the reasoning behind **EDA and feature engineering**, not merely a reporting exercise.

---

### Mental Model

A dataset can now be summarized from several complementary directions:

```math
\boxed{
\begin{aligned}
\text{Centre} &\rightarrow \text{mean, median, mode}\\
\text{Spread} &\rightarrow \text{range, variance, standard deviation, IQR}\\
\text{Position} &\rightarrow \text{percentiles, quartiles}\\
\text{Shape} &\rightarrow \text{skewness, kurtosis}
\end{aligned}
}
```

But all of these quantities describe the **sample we observed**.

Inferential statistics begins when we ask a harder question:

> **If we had drawn a different sample from the same population, how much would these statistics change?**

To answer that, we need the concept of **sampling distributions**.

---

## 1.15 Sampling Distributions

A **sampling distribution** is the probability distribution of a statistic calculated from many possible samples drawn from the same population.

This is one of the most important conceptual steps in inferential statistics.

Suppose a population has true mean:

```math
\mu
```

We draw a sample of size \(n\) and calculate its sample mean:

```math
\bar{x}
```

If we draw another random sample of the same size, we will usually obtain a slightly different sample mean.

Repeating this process many times gives:

```math
\bar{x}_1,\bar{x}_2,\bar{x}_3,\ldots
```

These sample means themselves form a distribution.

That distribution is called the **sampling distribution of the sample mean**.

Conceptually:

```math
\boxed{
\text{Population}
\rightarrow
\text{many samples}
\rightarrow
\text{many sample statistics}
\rightarrow
\text{sampling distribution}
}
```

The key idea is:

> A statistic such as the sample mean is itself a random variable because its value depends on which sample happens to be drawn.

This explains why two researchers drawing different samples from the same population may obtain slightly different estimates even when both calculations are perfectly correct.

---

### Sampling Variability

The variation among statistics calculated from different samples is called **sampling variability**.

For example:

```math
\bar{x}_1 \neq \bar{x}_2 \neq \bar{x}_3
```

in general.

A small amount of variation between samples is expected.

This does not necessarily indicate poor data collection or an incorrect calculation.

It is a natural consequence of observing only a finite subset of the population.

This leads to a fundamental principle:

```math
\boxed{
\text{Different random samples}
\rightarrow
\text{different sample statistics}
}
```

Inferential statistics attempts to quantify this uncertainty.

> **ML connection:** Training a model on a different random sample of the same population can produce different fitted parameters and predictions. Sampling variability is one of the statistical roots of model variance.

---

## 1.16 Standard Error

The **standard error** measures how much a sample statistic is expected to vary across repeated samples.

For the sample mean, the standard error is:

```math
\boxed{
SE(\bar{X})
=
\frac{\sigma}{\sqrt{n}}
}
```

where:

```math
\sigma
```

is the population standard deviation and:

```math
n
```

is the sample size.

When the population standard deviation is unknown, it is commonly estimated using the sample standard deviation:

```math
\boxed{
SE(\bar{X})
\approx
\frac{s}{\sqrt{n}}
}
```

---

### Standard Deviation vs Standard Error

These two quantities are related but describe different forms of variation.

**Standard deviation** describes variation among individual observations:

```math
\boxed{
SD
\rightarrow
\text{spread of data points}
}
```

**Standard error** describes variation among estimates calculated from different samples:

```math
\boxed{
SE
\rightarrow
\text{spread of sample statistics}
}
```

A useful distinction is:

```math
\boxed{
\begin{aligned}
\text{Standard deviation} &\rightarrow \text{How variable are the observations?}\\
\text{Standard error} &\rightarrow \text{How variable is my estimate?}
\end{aligned}
}
```

---

### Why Larger Samples Produce Smaller Standard Error

From:

```math
SE(\bar{X})
=
\frac{\sigma}{\sqrt{n}}
```

increasing \(n\) increases the denominator.

Therefore:

```math
n\uparrow
\quad\Rightarrow\quad
SE\downarrow
```

Larger samples generally produce more stable estimates.

For example, suppose:

```math
\sigma=20
```

For:

```math
n=25
```

we obtain:

```math
SE
=
\frac{20}{\sqrt{25}}
=
4
```

For:

```math
n=100
```

we obtain:

```math
SE
=
\frac{20}{\sqrt{100}}
=
2
```

Increasing the sample size by a factor of four reduces the standard error by a factor of two.

This follows from the square-root relationship:

```math
SE
\propto
\frac{1}{\sqrt{n}}
```

So doubling the sample size does **not** halve the standard error.

To halve the standard error, we need approximately four times as many observations.

> **ML connection:** More training data generally makes parameter estimates more stable, although the benefit grows with diminishing returns because uncertainty often decreases approximately with \(1/\sqrt{n}\) rather than \(1/n\).

---

## 1.17 Law of Large Numbers

The **Law of Large Numbers (LLN)** explains why sample averages become more reliable as the number of observations increases.

Suppose:

```math
X_1,X_2,\ldots,X_n
```

are independent observations drawn from the same distribution with finite expected value:

```math
\mathbb{E}[X]=\mu
```

The sample mean is:

```math
\bar{X}_n
=
\frac{1}{n}
\sum_{i=1}^{n}X_i
```

As the sample size becomes large:

```math
\boxed{
\bar{X}_n
\rightarrow
\mu
}
```

in the probabilistic sense described by the Law of Large Numbers.

The central intuition is:

> As more independent observations are collected, random fluctuations tend to average out and the sample mean moves closer to the true population mean.

---

### Coin Toss Intuition

Suppose a fair coin has:

```math
P(H)=0.5
```

After only 10 tosses, we might observe:

```math
7H,3T
```

giving an observed proportion:

```math
\hat{p}=0.7
```

After 100 tosses, the proportion may be closer to:

```math
0.54
```

After 10,000 tosses, it may be closer still to:

```math
0.501
```

The exact sequence varies, but the overall tendency is:

```math
\boxed{
\hat{p}
\rightarrow
p
\quad\text{as}\quad
n\rightarrow\infty
}
```

The LLN does **not** say that every larger sample must be closer than the previous one.

Random fluctuations can still occur.

It says that the estimate becomes increasingly concentrated around the true population value as sample size grows.

---

### LLN and Machine Learning

Machine learning depends heavily on the idea that empirical quantities calculated from finite training data can approximate underlying population quantities.

For example, a model may minimize average training loss:

```math
\hat{R}(f)
=
\frac{1}{n}
\sum_{i=1}^{n}
L(y_i,f(x_i))
```

where:

```math
\hat{R}(f)
```

is the **empirical risk** measured on the training sample.

The true quantity we care about is the expected loss over the underlying population:

```math
R(f)
=
\mathbb{E}[L(Y,f(X))]
```

Conceptually, with sufficiently representative data:

```math
\boxed{
\text{Average training behaviour}
\rightarrow
\text{expected population behaviour}
}
```

The Law of Large Numbers provides part of the mathematical intuition behind why learning from samples is possible at all.

---

## 1.18 Central Limit Theorem

The **Central Limit Theorem (CLT)** is one of the most important results in statistics.

It explains the distribution of sample means.

Suppose observations are independent and identically distributed with population mean:

```math
\mu
```

and finite population variance:

```math
\sigma^2
```

For sufficiently large sample size \(n\), the sampling distribution of the sample mean becomes approximately Gaussian:

```math
\boxed{
\bar{X}
\approx
\mathcal{N}
\left(
\mu,
\frac{\sigma^2}{n}
\right)
}
```

Equivalently, its standard deviation is:

```math
\boxed{
SE(\bar{X})
=
\frac{\sigma}{\sqrt{n}}
}
```

The striking part is that the original population itself does **not** need to be normally distributed.

Under broad conditions, the distribution of sample means tends toward a Gaussian distribution as sample size increases.

---

### Population Distribution vs Sampling Distribution

This distinction is essential.

Suppose individual customer spending is strongly right-skewed.

The population distribution might therefore look nothing like a Gaussian distribution.

But if we repeatedly draw samples of 100 customers and calculate the mean spending for each sample, the distribution of those sample means may be approximately Gaussian.

Thus:

```math
\boxed{
\text{Population data need not be Gaussian}
}
```

while:

```math
\boxed{
\text{Sampling distribution of the mean can become approximately Gaussian}
}
```

These are two different distributions.

---

### Why Averaging Produces Greater Stability

The variance of the sample mean is:

```math
\mathrm{Var}(\bar{X})
=
\frac{\sigma^2}{n}
```

As sample size increases:

```math
n\uparrow
\quad\Rightarrow\quad
\mathrm{Var}(\bar{X})\downarrow
```

Therefore sample means cluster increasingly tightly around:

```math
\mu
```

This explains why averages calculated from large samples are generally much more stable than individual observations.

---

### Standardizing the Sample Mean

The CLT can also be expressed using a standardized variable:

```math
\boxed{
Z
=
\frac{
\bar{X}-\mu
}{
\sigma/\sqrt{n}
}
\approx
\mathcal{N}(0,1)
}
```

This expression is important because it allows probabilities involving sample means to be calculated using the standard normal distribution.

It also forms the foundation for many confidence intervals and hypothesis tests.

---

## 1.19 Law of Large Numbers vs Central Limit Theorem

The LLN and CLT are related but answer different questions.

The **Law of Large Numbers** asks:

> What happens to the sample mean as the sample size grows?

Its answer is:

```math
\boxed{
\bar{X}
\rightarrow
\mu
}
```

The **Central Limit Theorem** asks:

> What does the distribution of sample means look like across repeated samples?

Its answer is approximately:

```math
\boxed{
\bar{X}
\sim
\mathcal{N}
\left(
\mu,
\frac{\sigma^2}{n}
\right)
}
```

A compact distinction is:

```math
\boxed{
\begin{aligned}
\text{LLN} &\rightarrow \text{Where does the sample mean go?}\\
\text{CLT} &\rightarrow \text{What distribution does the sample mean follow?}
\end{aligned}
}
```

The LLN gives us **consistency of averaging**.

The CLT gives us the **shape and uncertainty of that average**.

---

### Why the CLT Is So Important for Statistics

The CLT allows us to reason probabilistically about sample estimates.

If the sampling distribution is approximately Gaussian, we can quantify how far a sample mean is likely to lie from the population mean.

This gives rise to:

- standard errors
- confidence intervals
- z-tests
- many hypothesis-testing procedures

Thus the chain becomes:

```math
\boxed{
\text{Repeated sampling}
\rightarrow
\text{sampling distribution}
\rightarrow
\text{CLT}
\rightarrow
\text{standard error}
\rightarrow
\text{statistical inference}
}
```

---

### ML Connection

The same statistical logic helps explain several machine learning ideas.

Different training samples can produce different models:

```math
\boxed{
\text{Different samples}
\rightarrow
\text{different fitted parameters}
}
```

Larger representative datasets generally produce more stable estimates:

```math
\boxed{
n\uparrow
\rightarrow
\text{sampling uncertainty}\downarrow
}
```

And evaluation metrics calculated from finite test sets are themselves sample statistics.

For example, test accuracy:

```math
\widehat{\text{Accuracy}}
```

is an estimate of the model's unknown population-level predictive accuracy.

A different test sample would usually give a slightly different value.

Therefore model evaluation also contains sampling uncertainty.

> **ML connection:** A validation score is not a perfect property of a model. It is a statistic calculated from a finite sample, and therefore has sampling variability just like any other statistic.

---

### Mental Model

The core of sampling theory can be summarized as:

```math
\boxed{
\text{Population}
\rightarrow
\text{Random sample}
\rightarrow
\text{Statistic}
}
```

Repeat the sampling process:

```math
\boxed{
\text{Many samples}
\rightarrow
\text{many statistics}
\rightarrow
\text{sampling distribution}
}
```

Then:

```math
\boxed{
SE
\rightarrow
\text{uncertainty of the statistic}
}
```

```math
\boxed{
LLN
\rightarrow
\text{sample averages approach population averages}
}
```

```math
\boxed{
CLT
\rightarrow
\text{sample means become approximately Gaussian}
}
```

Once we know how sample estimates vary, the next question becomes:

> **How can we use a sample statistic to estimate an unknown population parameter, and how confident should we be in that estimate?**

That leads to **point estimation and confidence intervals**.

---

## 1.20 Point Estimation

A **point estimate** is a single numerical value used to estimate an unknown population parameter.

For example, if the true population mean is:

```math
\mu
```

and we calculate a sample mean:

```math
\bar{x}
```

then:

```math
\boxed{
\bar{x}
\rightarrow
\text{point estimate of }\mu
}
```

Likewise:

```math
\boxed{
\hat{p}
\rightarrow
\text{point estimate of }p
}
```

and:

```math
\boxed{
s^2
\rightarrow
\text{point estimate of }\sigma^2
}
```

A point estimate is useful because it gives a concrete value.

But it hides an important fact:

> A different random sample would usually produce a different estimate.

Therefore a point estimate should be accompanied, whenever possible, by some indication of its uncertainty.

---

### Estimator vs Estimate

These two terms are related but distinct.

An **estimator** is the rule or formula used to calculate an estimate.

For example:

```math
\bar{X}
=
\frac{1}{n}
\sum_{i=1}^{n}X_i
```

is an estimator of the population mean.

Once a particular sample is observed, the numerical result:

```math
\bar{x}=42.7
```

is the **estimate**.

So:

```math
\boxed{
\text{Estimator}
\rightarrow
\text{procedure}
}
```

```math
\boxed{
\text{Estimate}
\rightarrow
\text{numerical result}
}
```

---

## 1.21 Properties of a Good Estimator

Not every estimator is equally useful.

Several properties help describe the quality of an estimator.

### Unbiasedness

An estimator is **unbiased** if its expected value equals the true population parameter.

If:

```math
\hat{\theta}
```

estimates:

```math
\theta
```

then unbiasedness means:

```math
\boxed{
\mathbb{E}[\hat{\theta}]
=
\theta
}
```

The sample mean is an unbiased estimator of the population mean:

```math
\boxed{
\mathbb{E}[\bar{X}]
=
\mu
}
```

This means that over many repeated samples, the sample mean is centered on the true population mean.

Unbiased does **not** mean every individual estimate is exactly correct.

It means the estimator is correct **on average across repeated samples**.

---

### Bias

The bias of an estimator is:

```math
\boxed{
\mathrm{Bias}(\hat{\theta})
=
\mathbb{E}[\hat{\theta}]
-
\theta
}
```

A positive bias means the estimator tends to overestimate the parameter.

A negative bias means it tends to underestimate it.

An unbiased estimator has:

```math
\mathrm{Bias}(\hat{\theta})=0
```

> **ML connection:** Bias appears again in the bias-variance trade-off. There, bias refers more broadly to systematic error introduced by restrictive model assumptions, but the statistical intuition is similar: the procedure tends to miss the true relationship in a consistent direction.

---

### Consistency

An estimator is **consistent** if it approaches the true parameter as sample size increases.

Conceptually:

```math
\boxed{
\hat{\theta}_n
\rightarrow
\theta
\quad\text{as}\quad
n\rightarrow\infty
}
```

The sample mean is consistent for the population mean under standard conditions.

This connects directly to the Law of Large Numbers.

Consistency answers:

> If I collect more and more data, does my estimator move toward the truth?

---

### Efficiency

Suppose two estimators are both unbiased.

The estimator with smaller variance is generally preferred because it produces more stable estimates across repeated samples.

Conceptually:

```math
\boxed{
\text{Lower estimator variance}
\rightarrow
\text{greater precision}
}
```

This property is often described as **efficiency**.

So a useful estimator should ideally be:

- approximately unbiased
- consistent
- reasonably low-variance

These properties connect directly to the larger ML problem of learning stable, generalizable models from finite data.

---

## 1.22 Interval Estimation

A point estimate gives one value.

An **interval estimate** gives a range of plausible values for the population parameter.

Instead of reporting only:

```math
\bar{x}=50
```

we might report something like:

```math
45\leq\mu\leq55
```

The interval communicates uncertainty explicitly.

The most common form of interval estimate is a **confidence interval**.

---

## 1.23 Confidence Intervals

A **confidence interval** provides a range of plausible values for an unknown population parameter, based on the sample data and the sampling distribution of the estimator.

A generic confidence interval has the form:

```math
\boxed{
\text{Estimate}
\pm
\text{Margin of Error}
}
```

The margin of error is usually:

```math
\boxed{
\text{Critical value}
\times
\text{Standard error}
}
```

Therefore:

```math
\boxed{
\text{Confidence Interval}
=
\text{Estimate}
\pm
(
\text{Critical value}
\times
SE
)
}
```

---

### Confidence Interval for a Mean

If the population standard deviation is known and the sampling distribution of the mean is approximately normal, a confidence interval for the population mean is:

```math
\boxed{
\bar{x}
\pm
z_{\alpha/2}
\frac{\sigma}{\sqrt{n}}
}
```

where:

```math
z_{\alpha/2}
```

is the critical value from the standard normal distribution.

For a 95% confidence interval:

```math
z_{\alpha/2}
\approx
1.96
```

So:

```math
\boxed{
95\%\text{ CI}
=
\bar{x}
\pm
1.96
\frac{\sigma}{\sqrt{n}}
}
```

When the population standard deviation is unknown, it is typically replaced by the sample standard deviation and the **t-distribution** is used instead:

```math
\boxed{
\bar{x}
\pm
t_{\alpha/2,\,n-1}
\frac{s}{\sqrt{n}}
}
```

The t-distribution has heavier tails than the standard normal distribution, reflecting the additional uncertainty introduced by estimating the population standard deviation from the sample.

As sample size increases, the t-distribution approaches the standard normal distribution.

---

### Example

Suppose a sample has:

```math
\bar{x}=100
```

with standard error:

```math
SE=5
```

Using a 95% normal-based confidence interval:

```math
100
\pm
1.96(5)
```

The margin of error is:

```math
1.96\times5
=
9.8
```

Therefore:

```math
\boxed{
90.2
\leq
\mu
\leq
109.8
}
```

The interval is centered at the point estimate, with its width determined by the uncertainty of that estimate.

---

## 1.24 What Does 95% Confidence Mean?

This is one of the most commonly misunderstood ideas in statistics.

A 95% confidence interval does **not** mean:

> There is a 95% probability that the fixed population parameter lies inside this particular interval.

In classical frequentist statistics, the population parameter is treated as fixed.

The interval is random because it depends on the sample.

The correct interpretation is:

> If we repeatedly drew samples and constructed confidence intervals using the same procedure, approximately 95% of those intervals would contain the true population parameter.

Conceptually:

```math
\boxed{
\text{Repeated samples}
\rightarrow
\text{many confidence intervals}
\rightarrow
\text{about 95\% capture the true parameter}
}
```

Once one particular interval has been calculated, it either contains the true parameter or it does not.

The 95% refers to the **long-run success rate of the procedure**.

---

### Confidence Level and Interval Width

A higher confidence level requires a wider interval.

For example:

```math
99\%\text{ CI}
```

uses a larger critical value than:

```math
95\%\text{ CI}
```

Therefore:

```math
\boxed{
\text{Higher confidence}
\rightarrow
\text{wider interval}
}
```

This is a fundamental trade-off.

To be more confident that the interval captures the true value, we must accept less precision.

---

### Sample Size and Interval Width

Recall:

```math
SE
=
\frac{\sigma}{\sqrt{n}}
```

As sample size increases:

```math
SE\downarrow
```

Therefore:

```math
\boxed{
n\uparrow
\rightarrow
\text{narrower confidence interval}
}
```

Larger samples allow more precise estimates.

This gives us two distinct ways an interval can become wider:

```math
\boxed{
\begin{aligned}
\text{Higher confidence level} &\rightarrow \text{wider interval}\\
\text{Greater sampling uncertainty} &\rightarrow \text{wider interval}
\end{aligned}
}
```

and one major way it can become narrower:

```math
\boxed{
\text{Larger sample size}
\rightarrow
\text{smaller SE}
\rightarrow
\text{narrower interval}
}
```

---

## 1.25 Confidence Intervals in Machine Learning

The same idea of interval estimation applies to ML evaluation.

Suppose a classifier achieves:

```math
\widehat{\text{Accuracy}}=0.91
```

on a finite test set.

The value:

```math
0.91
```

is a point estimate of the model's unknown population-level accuracy.

A different test sample might produce:

```math
0.89
```

or:

```math
0.93
```

Therefore, reporting only a single metric can create a false impression of precision.

A confidence interval can communicate the uncertainty around the estimate.

The same applies to metrics such as:

- accuracy
- precision
- recall
- AUC
- mean error
- regression coefficients

> **ML connection:** Evaluation metrics are sample statistics. Confidence intervals help distinguish a genuinely precise estimate from a score that may vary substantially across plausible test samples.

---

### Confidence Intervals and Model Comparison

Suppose model A achieves:

```math
0.91
```

accuracy and model B achieves:

```math
0.92
```

accuracy.

It is tempting to conclude that model B is better.

But if both estimates have substantial sampling uncertainty, the observed difference:

```math
0.01
```

may be too small to support a meaningful conclusion.

This leads naturally to the next statistical question:

> Is an observed difference large enough to be considered evidence of a genuine effect, or could it plausibly have arisen from random sampling variation?

That question is answered using **hypothesis testing**.

---

### Mental Model

Point estimation asks:

```math
\boxed{
\text{What single value best estimates the parameter?}
}
```

Interval estimation asks:

```math
\boxed{
\text{What range of values is reasonably compatible with the data?}
}
```

The full chain is:

```math
\boxed{
\text{Sample}
\rightarrow
\text{Point estimate}
\rightarrow
SE
\rightarrow
\text{Confidence interval}
}
```

And the logic of inference becomes:

```math
\boxed{
\text{Estimate}
+
\text{uncertainty}
\rightarrow
\text{more informative conclusion}
}
```

Confidence intervals quantify **how uncertain an estimate is**.

Hypothesis testing, which comes next, asks whether the observed data provides enough evidence to challenge a specific claim about the population.

---

## 1.26 Hypothesis Testing

**Hypothesis testing** is a formal framework for deciding whether observed sample evidence is strong enough to challenge a claim about a population.

The central question is:

> Is the observed result reasonably explainable by random sampling variation, or is it sufficiently unusual that we should reconsider our original assumption?

A hypothesis test begins with two competing statements.

### Null Hypothesis

The **null hypothesis**, written as:

```math
H_0
```

represents the default or baseline claim.

It commonly represents:

- no difference
- no effect
- no association
- no change

For example, suppose a manufacturer claims that the average lifetime of a component is 1,000 hours.

The null hypothesis might be:

```math
H_0:\mu=1000
```

### Alternative Hypothesis

The **alternative hypothesis**, written as:

```math
H_1
```

or:

```math
H_a
```

represents the competing claim for which we are looking for evidence.

For example:

```math
H_1:\mu\neq1000
```

The hypotheses therefore form a pair:

```math
\boxed{
\begin{aligned}
H_0 &: \text{baseline claim}\\
H_1 &: \text{competing claim}
\end{aligned}
}
```

A hypothesis test does not normally attempt to prove either hypothesis with certainty.

Instead, it asks whether the observed data provides enough evidence to **reject the null hypothesis**.

---

## 1.27 Test Statistics

A **test statistic** measures how far the observed sample result lies from what would be expected if the null hypothesis were true.

For a population mean with known population standard deviation, a z-statistic can be written as:

```math
\boxed{
z
=
\frac{
\bar{x}-\mu_0
}{
\sigma/\sqrt{n}
}
}
```

where:

```math
\mu_0
```

is the population mean assumed under the null hypothesis.

The numerator:

```math
\bar{x}-\mu_0
```

measures the observed difference from the null value.

The denominator:

```math
\frac{\sigma}{\sqrt{n}}
```

measures the expected sampling variability.

Therefore the test statistic essentially asks:

> How many standard errors away from the null expectation is the observed result?

A large absolute test statistic indicates that the observed result is difficult to reconcile with the null hypothesis.

---

## 1.28 The p-Value

The **p-value** measures how surprising the observed result would be if the null hypothesis were true.

More precisely:

> The p-value is the probability, assuming the null hypothesis is true, of obtaining a result at least as extreme as the one observed.

Conceptually:

```math
\boxed{
p
=
P(
\text{result at least as extreme as observed}
\mid
H_0
)
}
```

A small p-value means the observed data would be relatively unusual under the null hypothesis.

Therefore:

```math
\boxed{
\text{Small p-value}
\rightarrow
\text{evidence against }H_0
}
```

A large p-value means the data is reasonably compatible with the null hypothesis.

But this distinction requires careful interpretation.

---

### What a p-Value Does Not Mean

A p-value is **not**:

```math
P(H_0\mid\text{data})
```

It does not tell us the probability that the null hypothesis is true.

Instead, it is calculated under the assumption that the null hypothesis **is already true**:

```math
P(\text{data or more extreme}\mid H_0)
```

These are fundamentally different conditional probabilities.

A p-value also does not measure:

- the size of an effect
- the practical importance of a result
- the probability that the result occurred purely by chance

It measures compatibility between the observed data and the null hypothesis under a specified statistical model.

---

## 1.29 Significance Level

Before conducting a hypothesis test, we choose a **significance level**:

```math
\alpha
```

A common choice is:

```math
\alpha=0.05
```

The decision rule is commonly written as:

```math
\boxed{
\begin{aligned}
p \leq \alpha
&\rightarrow
\text{reject }H_0\\
p > \alpha
&\rightarrow
\text{fail to reject }H_0
\end{aligned}
}
```

If:

```math
p=0.02
```

and:

```math
\alpha=0.05
```

then:

```math
p<\alpha
```

and the result is called **statistically significant** at the 5% level.

---

### Reject vs Fail to Reject

The wording matters.

If the evidence is insufficient, we say:

> **Fail to reject the null hypothesis.**

We normally do **not** say:

> Accept the null hypothesis.

Failure to find sufficient evidence against a claim does not prove that the claim is true.

Conceptually:

```math
\boxed{
\text{Insufficient evidence against }H_0
\neq
\text{proof that }H_0\text{ is true}
}
```

This distinction is fundamental to statistical reasoning.

---

## 1.30 One-Tailed and Two-Tailed Tests

The alternative hypothesis determines the direction of the test.

A **two-tailed test** looks for a difference in either direction:

```math
H_1:\mu\neq\mu_0
```

For example:

> Has the average changed?

A **right-tailed test** looks specifically for an increase:

```math
H_1:\mu>\mu_0
```

A **left-tailed test** looks specifically for a decrease:

```math
H_1:\mu<\mu_0
```

Thus:

```math
\boxed{
\begin{aligned}
\mu\neq\mu_0 &\rightarrow \text{two-tailed}\\
\mu>\mu_0 &\rightarrow \text{right-tailed}\\
\mu<\mu_0 &\rightarrow \text{left-tailed}
\end{aligned}
}
```

The direction should be determined by the research question **before examining the result**, not chosen afterward merely to obtain statistical significance.

---

## 1.31 Type I and Type II Errors

Hypothesis testing makes decisions under uncertainty.

Therefore two kinds of errors are possible.

### Type I Error

A **Type I error** occurs when we reject the null hypothesis even though it is actually true.

```math
\boxed{
\text{Type I Error}
=
\text{Reject true }H_0
}
```

This is analogous to a **false positive**.

The probability of a Type I error is controlled by the significance level:

```math
\boxed{
P(\text{Type I Error})=\alpha
}
```

when the test assumptions and decision rule apply as specified.

For:

```math
\alpha=0.05
```

we are accepting a 5% long-run false-positive rate at the rejection boundary defined by the test.

---

### Type II Error

A **Type II error** occurs when we fail to reject the null hypothesis even though the alternative is actually true.

```math
\boxed{
\text{Type II Error}
=
\text{Fail to reject false }H_0
}
```

This is analogous to a **false negative**.

Its probability is represented by:

```math
\beta
```

Thus:

```math
\boxed{
P(\text{Type II Error})=\beta
}
```

The two errors can be summarized as:

| Reality | Reject $H_0$ | Fail to Reject $H_0$ |
|---|---|---|
| $H_0$ true | Type I error | Correct decision |
| $H_0$ false | Correct decision | Type II error |

A useful ML analogy is:

```math
\boxed{
\begin{aligned}
\text{Type I error} &\leftrightarrow \text{False Positive}\\
\text{Type II error} &\leftrightarrow \text{False Negative}
\end{aligned}
}
```

The analogy is not merely cosmetic: both frameworks involve making decisions under uncertainty and balancing different kinds of mistakes.

---

## 1.32 Statistical Power

The **power** of a statistical test is the probability that it correctly rejects the null hypothesis when a real effect exists.

Since:

```math
\beta
```

is the probability of a Type II error:

```math
\boxed{
\text{Power}
=
1-\beta
}
```

High power means the test has a good chance of detecting a genuine effect.

Power generally increases when:

- sample size increases
- the true effect is larger
- measurement variability decreases
- the significance threshold is made less strict, although this can increase Type I error

The sample-size relationship is particularly important:

```math
\boxed{
n\uparrow
\rightarrow
SE\downarrow
\rightarrow
\text{greater ability to detect real effects}
}
```

A test with very little data may fail to detect an important effect simply because the estimate is too noisy.

---

## 1.33 Statistical Significance vs Practical Significance

A statistically significant result is not automatically important in practice.

With a very large sample, even a tiny effect can produce a very small p-value.

Suppose a new model improves accuracy from:

```math
95.00\%
```

to:

```math
95.02\%
```

With enough observations, this difference might be statistically significant.

But whether a 0.02 percentage-point improvement matters operationally is a separate question.

Therefore:

```math
\boxed{
\text{Statistical significance}
\neq
\text{Practical significance}
}
```

A good analysis should consider:

- effect size
- uncertainty
- cost
- business or scientific importance
- consequences of errors

and not merely whether:

```math
p<0.05
```

> **ML connection:** A tiny improvement in a metric is not automatically worth deploying. Statistical evidence, effect size, computational cost, latency, interpretability, and business impact all matter.

---

## 1.34 Hypothesis Testing and Confidence Intervals

Confidence intervals and hypothesis tests are closely connected.

For a two-sided test at significance level:

```math
\alpha=0.05
```

a corresponding 95% confidence interval can often be used to reach the same decision.

Suppose the null hypothesis is:

```math
H_0:\mu=0
```

If the 95% confidence interval excludes:

```math
0
```

then the corresponding two-sided test typically rejects the null hypothesis at the 5% level.

If the interval contains:

```math
0
```

then the null hypothesis is typically not rejected.

Conceptually:

```math
\boxed{
\text{Hypothesis test}
\leftrightarrow
\text{Confidence interval}
}
```

The confidence interval is often more informative because it shows both:

- whether the null value is plausible
- the estimated magnitude and uncertainty of the effect

---

## 1.35 Common Statistical Tests

Different questions require different statistical tests.

For ML foundations, it is more important to understand what the major tests are designed to answer than to memorize every formula.

### z-Test

A **z-test** can be used for inference about means when the relevant sampling distribution can be treated as normal and the population variance is known or suitable large-sample conditions apply.

Its basic standardized form is:

```math
z
=
\frac{
\text{estimate}-\text{null value}
}{
SE
}
```

### t-Test

A **t-test** is commonly used for inference about means when the population standard deviation is unknown.

For a one-sample mean:

```math
\boxed{
t
=
\frac{
\bar{x}-\mu_0
}{
s/\sqrt{n}
}
}
```

Common forms include:

- one-sample t-test
- independent two-sample t-test
- paired t-test

### Chi-Square Test

A **chi-square test** is commonly used with categorical data.

One important use is testing whether two categorical variables are independent.

For example:

> Is customer churn associated with subscription type?

The chi-square statistic compares observed frequencies with frequencies expected under independence.

### ANOVA

**Analysis of Variance (ANOVA)** is used to test whether the means of multiple groups differ.

Instead of performing many pairwise t-tests, ANOVA can first test the broader question:

```math
H_0:
\mu_1=\mu_2=\cdots=\mu_k
```

against the alternative that at least one group mean differs.

The key mental map is:

```math
\boxed{
\begin{aligned}
\text{z/t tests} &\rightarrow \text{means}\\
\text{chi-square} &\rightarrow \text{categorical relationships}\\
\text{ANOVA} &\rightarrow \text{multiple group means}
\end{aligned}
}
```

These tests become easier to remember when viewed as different applications of the same general framework:

```math
\boxed{
\frac{
\text{Observed effect}
}{
\text{Expected random variation}
}
}
```

---

## 1.36 Statistics and Machine Learning

Statistics and machine learning overlap heavily, but their traditional emphasis differs.

Statistics often asks:

> What can we infer about the population or data-generating process?

Machine learning often asks:

> How accurately can we predict unseen observations?

But both depend on the same fundamental structure:

```math
\boxed{
\text{Finite observed data}
\rightarrow
\text{learn something that generalizes}
}
```

Several ideas developed in this chapter map directly into ML.

### Sampling and Train/Test Data

A training dataset is a sample from a larger data-generating process.

A test dataset is another finite sample used to estimate future predictive behaviour.

Therefore:

```math
\boxed{
\text{Test metric}
=
\text{sample estimate of generalization performance}
}
```

### Bias and Variance

Statistical estimators can have bias and variance.

Machine learning models exhibit the same broad tension.

A model that is too restrictive may systematically miss important structure:

```math
\text{high bias}
```

A model that reacts excessively to the particular training sample may change substantially when the data changes:

```math
\text{high variance}
```

This becomes the **bias-variance trade-off**, which we will encounter directly in model learning and regularization.

### Uncertainty in Model Evaluation

Accuracy, RMSE, MAE, precision, recall, F1 score, and other evaluation metrics are calculated from finite samples.

Therefore they are estimates, not perfectly known constants.

Conceptually:

```math
\boxed{
\text{Observed metric}
=
\text{underlying performance}
+
\text{sampling variation}
}
```

This is a conceptual decomposition rather than an exact universal equation, but it captures the important idea: the score we observe depends partly on which finite test sample we happened to evaluate.

### Statistical Testing in ML

Statistical reasoning can help answer questions such as:

- Is model B genuinely better than model A?
- Is an observed improvement larger than expected sampling noise?
- Is a feature meaningfully associated with the target?
- Did an A/B experiment produce evidence of a real change?
- How uncertain is an estimated coefficient or performance metric?

But statistical significance should never replace practical judgment.

A model can be statistically distinguishable from another model while providing almost no useful operational improvement.

---

## 1.37 Final Mental Model

The entire statistics chapter can be compressed into one chain.

We begin with an unknown population:

```math
\boxed{
\text{Population}
}
```

We observe only a sample:

```math
\boxed{
\text{Population}
\rightarrow
\text{Sample}
}
```

We summarize that sample:

```math
\boxed{
\text{Sample}
\rightarrow
\text{descriptive statistics}
}
```

We use sample statistics to estimate unknown population parameters:

```math
\boxed{
\text{Sample statistic}
\rightarrow
\text{population parameter estimate}
}
```

Because different samples produce different estimates, we quantify sampling uncertainty:

```math
\boxed{
\text{Sampling distribution}
\rightarrow
SE
\rightarrow
\text{confidence interval}
}
```

And when we want to evaluate a specific population claim:

```math
\boxed{
H_0
\rightarrow
\text{sample evidence}
\rightarrow
\text{test statistic}
\rightarrow
p\text{-value}
\rightarrow
\text{decision}
}
```

The larger ML connection is:

```math
\boxed{
\text{Observed training data}
\rightarrow
\text{learned model}
\rightarrow
\text{unseen population data}
}
```

Statistics gives us the language for asking whether that leap from **observed sample** to **unseen population** is justified.

That is the statistical foundation of **generalization** in machine learning.

---

## 1.38 Quick Reference

| Concept | Core Idea | ML Connection |
|---|---|---|
| Population | Complete target group/process | All data the model may encounter |
| Sample | Observed subset of population | Training/test dataset |
| Parameter | Population quantity | Unknown true relationship/performance |
| Statistic | Quantity calculated from sample | Metric or fitted estimate |
| Standard deviation | Variation among observations | Feature/target spread |
| Standard error | Variation of an estimator | Uncertainty in estimated metrics/parameters |
| LLN | Sample averages approach population averages | Empirical averages become more reliable with data |
| CLT | Sampling distribution of means becomes approximately normal | Basis for uncertainty calculations |
| Point estimate | Single estimate of parameter | Accuracy, coefficient, RMSE, etc. |
| Confidence interval | Range generated by an interval-estimation procedure | Uncertainty around model performance |
| Null hypothesis | Baseline claim | No improvement/no effect |
| p-value | Compatibility of observed result with $H_0$ | Evidence when comparing effects/models |
| Type I error | Rejecting a true $H_0$ | False-positive-like decision |
| Type II error | Failing to reject a false $H_0$ | False-negative-like decision |
| Power | Probability of detecting a specified real effect | Ability of an experiment to reveal improvement |
| Statistical significance | Evidence against a null model | Does not automatically imply useful ML improvement |

---

With probability describing **uncertainty in possible outcomes** and statistics describing **what can be learned from observed outcomes**, the mathematical groundwork is now in place for the central task of machine learning:

```math
\boxed{
\text{Learn patterns from finite data that generalize to unseen data}
}
```
