# Appendix B: Statistical Measures

This appendix is a formula-first reference for the statistical measures used throughout [Statistics for Machine Learning](04-Statistics_for_Machine_Learning.md). The main chapter explains how these ideas support machine learning; this appendix collects their notation, equations, and relationships in one place.

## 1. Notation

| Symbol | Meaning |
| -------- | --------- |
| $x_i$ | The $i$-th observation |
| $n$ | Number of observations in a sample |
| $N$ | Number of observations in a population |
| $\bar{x}$ | Sample mean |
| $\mu$ | Population mean |
| $s^2$ | Sample variance |
| $\sigma^2$ | Population variance |
| $s$ | Sample standard deviation |
| $\sigma$ | Population standard deviation |

Unless stated otherwise, $x_1,\ldots,x_n$ denotes a sample.

## 2. Measures of Centre

### 2.1. Arithmetic Mean

The arithmetic mean is the sum of the observations divided by their count.

```math
\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i
```

For a population, the corresponding mean is:

```math
\mu = \frac{1}{N}\sum_{i=1}^{N}x_i
```

The mean uses every observation but is sensitive to extreme values.

### 2.2. Weighted Mean

When observations have different weights $w_i$, the weighted mean is:

```math
\bar{x}_w = \frac{\sum_{i=1}^{n}w_i x_i}{\sum_{i=1}^{n}w_i}
```

### 2.3. Median

The median is the middle value after the observations are **sorted**. For an odd number of observations:

```math
\text{Median} = x_{\left(\frac{n+1}{2}\right)}
```

For an even number of observations, it is commonly the mean of the two middle values:

```math
\text{Median} =
\frac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2}+1\right)}}{2}
```

The median is less sensitive to extreme observations than the mean.

### 2.4. Mode

The mode is the value or category that occurs most frequently. A dataset may have one mode, multiple modes, or no unique mode.

## 3. Measures of Spread

### 3.1. Range

The range is the difference between the largest and smallest observations.

```math
\text{Range} = x_{\max} - x_{\min}
```

It is simple to calculate but depends only on two observations.

### 3.2. Mean Absolute Deviation

The mean absolute deviation about the sample mean is the average distance from the observations to the sample mean.

```math
\text{MAD}_{\text{mean}} =
\frac{1}{n}\sum_{i=1}^{n}\lvert x_i-\bar{x}\rvert
```

> The abbreviation MAD can also mean median absolute deviation, so the centre used in the formula should be stated explicitly.

### 3.3. Median Absolute Deviation

The median absolute deviation is the median of the distances from the observations to the sample median.

```math
\text{MAD}_{\text{median}} =
\text{Median}\left(\lvert x_i-\text{Median}(x)\rvert\right)
```

It is more resistant to outliers than variance and standard deviation.

### 3.4. Population Variance

Population variance is the average squared distance from the population mean.

```math
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2
```

### 3.5. Sample Variance

Sample variance estimates population variance from a sample. The denominator is $n-1$ rather than $n$:

```math
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2
```

The $n-1$ denominator is Bessel's correction. It compensates, on average, for estimating the population mean from the same sample.

### 3.6. Standard Deviation

Standard deviation is the square root of variance.

```math
\sigma = \sqrt{\sigma^2}
```

for a population, and:

```math
s = \sqrt{s^2}
```

for a sample. Unlike variance, standard deviation has the same units as the observations.

## 4. Measures of Relative Position

### 4.1. Percentile

The $p$-th percentile is a value below which approximately $p\%$ of the observations fall. Percentiles describe relative position, and exact calculation can depend on the chosen interpolation convention.

### 4.2. Quartiles

Quartiles divide ordered data into four parts:

| Quartile | Approximate position |
| ---------- | ---------------------- |
| $Q_1$ | 25th percentile |
| $Q_2$ | 50th percentile, the median |
| $Q_3$ | 75th percentile |

### 4.3. Interquartile Range

The interquartile range measures the spread of the middle 50% of observations.

```math
\text{IQR} = Q_3-Q_1
```

Because it excludes the lowest and highest quarters, the IQR is less sensitive to extreme values than the range.

## 5. Measures of Distribution Shape

### 5.1. Skewness

Skewness describes asymmetry around the centre of a distribution. A population standardized third central moment is:

```math
\gamma_1 = \frac{\mathbb{E}\left[(X-\mu)^3\right]}{\sigma^3}
```

Positive skewness indicates a relatively longer right tail; negative skewness indicates a relatively longer left tail. Sample implementations may use bias corrections.

### 5.2. Kurtosis

Kurtosis describes the standardized fourth central moment:

```math
\beta_2 = \frac{\mathbb{E}\left[(X-\mu)^4\right]}{\sigma^4}
```

Excess kurtosis subtracts the normal distribution's value of $3$:

```math
\gamma_2 = \beta_2-3
```

Different software packages may report kurtosis or excess kurtosis, so the convention should be checked before comparing values.

## 6. Measures of Relationship

### 6.1. Covariance

Population covariance measures whether two variables tend to vary together:

```math
\text{Cov}(X,Y) =
\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu_X)(y_i-\mu_Y)
```

Sample covariance is commonly calculated as:

```math
s_{XY} =
\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})
```

The sign indicates the direction of joint variation, but the magnitude depends on the units of both variables.

### 6.2. Pearson Correlation

Pearson correlation standardizes covariance by the standard deviations of the two variables:

```math
r_{XY} =
\frac{s_{XY}}{s_Xs_Y}
```

Equivalently:

```math
r_{XY} =
\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\left(\sum_{i=1}^{n}(x_i-\bar{x})^2\right)
\left(\sum_{i=1}^{n}(y_i-\bar{y})^2\right)}}
```

Its value lies between $-1$ and $1$:

| Value | Interpretation |
| ----- | -------------- |
| $r=1$ | Perfect positive linear relationship |
| $r=0$ | No linear relationship measured by Pearson correlation |
| $r=-1$ | Perfect negative linear relationship |

Pearson correlation measures linear association, not causation. A value near zero does not rule out a nonlinear relationship.

### 6.3. Covariance and Correlation Compared

| Property | Covariance | Pearson correlation |
| ---------- | ------------ | --------------------- |
| Units | Product of the two variables' units | Unitless |
| Scale | Changes when either variable is rescaled | Unchanged by positive rescaling |
| Range | Unbounded | Between $-1$ and $1$ |
| Interpretation | Direction and unstandardized joint variation | Direction and strength of linear association |

## 7. Quick Comparison

| Question | Useful measure |
| ---------- | ---------------- |
| Where is the centre? | Mean, median, or mode |
| How widely are observations spread? | Range, variance, standard deviation, or mean absolute deviation |
| How can spread be summarized robustly? | Median, IQR, or median absolute deviation |
| Where does an observation lie relative to others? | Percentile or quartile |
| Is the distribution asymmetric? | Skewness |
| How heavy are the tails? | Kurtosis |
| Do two variables vary together? | Covariance |
| How strong is their linear association? | Pearson correlation |

## 8. Related Chapter Sections

The main chapter retains the conceptual and machine-learning context for these measures:

- [Describing the Centre of Data](04-Statistics_for_Machine_Learning.md#describing-the-centre-of-data)
- [Measuring Spread](04-Statistics_for_Machine_Learning.md#measuring-spread)
- [Standard Deviation](04-Statistics_for_Machine_Learning.md#standard-deviation)
- [Percentiles](04-Statistics_for_Machine_Learning.md#percentiles)
- [Interquartile Range](04-Statistics_for_Machine_Learning.md#interquartile-range)
- [Shape of a Distribution](04-Statistics_for_Machine_Learning.md#shape-of-a-distribution)
