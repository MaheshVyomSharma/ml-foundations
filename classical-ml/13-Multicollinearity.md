# 13. Multicollinearity

> *When predictors repeat the same information, the model may struggle to decide which predictor deserves credit.*

---

## 1. Why Should I Care?

A regression model can make accurate predictions while producing coefficients that are difficult to trust or explain.

This often happens when two or more input features contain overlapping information. The model can use several different combinations of those features to make nearly the same prediction, so small changes in the data can cause large changes in the individual coefficients.

This problem is called **multicollinearity**.

It matters most when you need to:

- interpret the effect of individual predictors
- identify important variables
- test hypotheses about coefficients
- build a stable linear model

Multicollinearity is not automatically a prediction failure. It is primarily a problem of redundant information and unstable interpretation.

---

## 2. Learning Objectives

After reading this chapter, you should be able to:

- Define multicollinearity and distinguish it from perfect linear dependence.
- Explain why correlated predictors make coefficients unstable.
- Use correlation, VIF, and matrix diagnostics to investigate the problem.
- Distinguish prediction goals from coefficient-interpretation goals.
- Choose an appropriate response, such as removing, combining, or regularizing features.

---

## 3. What Is Multicollinearity?

**Multicollinearity** occurs when one predictor can be approximately explained by one or more other predictors in a regression model.

For example, a dataset might contain:

- house area in square metres
- house area in square feet
- number of rooms

The first two variables carry almost the same information. A model may have difficulty separating their individual effects.

Perfect multicollinearity is the extreme case in which one predictor is an exact linear combination of other predictors. In that case, the ordinary least-squares coefficient solution is not uniquely determined.

Most practical cases are **near multicollinearity**, where the relationship is strong but not exact.

---

## 4. A Simple Mathematical View

A multiple linear regression model can be written as:

```math
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_px_p
```

Suppose two predictors are nearly related by:

```math
x_2 \approx ax_1 + b
```

Then different values of $\beta_1$ and $\beta_2$ can produce very similar predictions. The model can move weight from one feature to the other without changing the fitted values very much.

In matrix form, the ordinary least-squares estimate is commonly written as:

```math
\hat{\boldsymbol{\beta}}
=
(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y}
```

When columns of $\mathbf{X}$ are highly dependent, $\mathbf{X}^{T}\mathbf{X}$ becomes ill-conditioned. Its inverse then becomes highly sensitive to small changes in the data.

If the columns are exactly dependent, the matrix is singular and the inverse does not exist.

---

## 5. What Causes It?

Common causes include:

- Including the same quantity in different units.
- Including a total together with components that make up that total.
- Creating many polynomial or interaction features.
- Measuring related aspects of the same underlying process.
- Using one-hot encoding while also retaining an intercept and every category column.
- Collecting variables that naturally move together over time.

Multicollinearity can also arise from a small or unrepresentative sample, even when the underlying population relationships are weaker.

---

## 6. What Problems Does It Cause?

### 6.1. Unstable Coefficients

Small changes to the training data can cause large changes in individual coefficient values.

A coefficient may:

- change magnitude
- change sign
- appear important in one sample and unimportant in another

The fitted predictions can remain similar because the correlated predictors compensate for one another.

### 6.2. Larger Standard Errors

When predictors overlap, there is less independent information available for estimating each separate coefficient. This increases coefficient uncertainty and often produces larger standard errors.

As a result, a genuinely useful predictor may have a large p-value when considered separately.

### 6.3. Difficult Interpretation

A coefficient describes the expected change in the target for a one-unit increase in that feature **while holding the other features constant**.

With strongly correlated predictors, that hypothetical change may be unrealistic or poorly supported by the data.

### 6.4. Numerical Instability

Near-dependent columns can make matrix calculations sensitive to rounding and floating-point precision. This is especially important when using an unregularized closed-form solution.

### 6.5. What It Usually Does Not Cause

Multicollinearity does not necessarily make predictions inaccurate. If the shared signal is useful, the model can still predict well even though the individual coefficients are unstable.

---

## 7. How to Detect It

No single diagnostic should be treated as a universal cutoff. Use several pieces of evidence and consider the purpose of the model.

### 7.1. Pairwise Correlations

A correlation matrix can reveal pairs of strongly linearly related predictors.

This is a useful first check, but it is incomplete. A predictor can be well explained by a combination of several other predictors even when no individual pair has an extreme correlation.

### 7.2. Variance Inflation Factor

The **Variance Inflation Factor (VIF)** for predictor $j$ is:

```math
\mathrm{VIF}_j = \frac{1}{1-R_j^2}
```

Here, $R_j^2$ is obtained by regressing predictor $x_j$ on all the other predictors.

Interpretation:

- $R_j^2$ near 0 means the other predictors explain little of $x_j$.
- $R_j^2$ near 1 means the other predictors explain most of $x_j$.
- As $R_j^2$ approaches 1, the VIF grows rapidly.

A VIF of 1 indicates no inflation from linear relationships with the other predictors. Rules such as 5 or 10 are common warnings, but they are heuristics rather than laws. Domain knowledge, sample size, and the model goal still matter.

The related **tolerance** is:

```math
\mathrm{Tolerance}_j = 1-R_j^2 = \frac{1}{\mathrm{VIF}_j}
```

### 7.3. Condition Number and Singular Values

The singular values of the design matrix describe the strength of its independent directions. A large condition number indicates that some directions are represented much more strongly than others and that the system may be numerically sensitive.

This connects multicollinearity to linear algebra concepts such as rank, singularity, and linear independence.

### 7.4. Coefficient Stability

Fit the model across bootstrap samples, time windows, or reasonable data splits. Large changes in coefficient values or signs are evidence that individual effects are not stable.

This check is often more informative than a threshold alone because it tests the stability you actually care about.

---

## 8. A Practical Investigation Workflow

1. Define the goal: prediction, explanation, inference, or feature selection.
2. Inspect feature definitions for duplicated or overlapping information.
3. Fit a baseline model and record coefficients, standard errors, and validation performance.
4. Inspect a predictor correlation matrix.
5. Calculate VIF after applying the same preprocessing used by the model.
6. Check coefficient stability across resamples or meaningful time periods.
7. Choose a remedy based on the goal and the data-generating process.
8. Refit and compare both predictive performance and interpretability.

Diagnostics must be calculated using training data within each validation split when they influence feature selection. Computing them on the full dataset can leak information from the validation or test set.

---

## 9. How to Address It

### 9.1. Remove Redundant Features

If two features represent the same concept, keep the one with the clearest definition, strongest data quality, or greatest usefulness for the intended task.

Do not remove a feature solely because it is correlated without considering domain meaning and downstream performance.

### 9.2. Combine Features

Create a meaningful index, total, average, or domain-specific summary when the combined quantity better represents the question being asked.

The transformation should be defined using training data and justified by the application.

### 9.3. Use Domain-Driven Feature Selection

When explanation matters, select one representative variable from a group of interchangeable predictors. This makes the coefficient interpretation more defensible.

### 9.4. Use Ridge Regression

Ridge regression adds an L2 penalty:

```math
\min_{\boldsymbol{\beta}}
\left[
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
+\lambda\sum_{j=1}^{p}\beta_j^2
\right]
```

Ridge usually keeps correlated predictors while shrinking their coefficients. It often improves numerical stability and prediction, but the coefficients should not be interpreted as completely independent effects.

### 9.5. Use Elastic Net

Elastic Net combines L1 and L2 penalties. It can shrink groups of correlated features and set some coefficients to zero, making it useful when both stability and feature selection are desired.

### 9.6. Use Principal Components

Principal Component Analysis (PCA) replaces correlated features with orthogonal components. This can remove linear redundancy, but the resulting components are combinations of the original features and may be harder to explain.

### 9.7. Collect Better Data

More observations do not automatically remove multicollinearity. If the data collection process never separates two predictors, a larger sample may continue to show the same dependence.

Collect data in situations where the predictors vary independently when possible.

---

## 10. What Does Not Solve It by Itself?

- **Standardization** changes units but does not remove correlation.
- **More data** helps only when it adds independent information.
- **A high R²** does not prove that individual coefficients are reliable.
- **Dropping every highly correlated feature** can discard useful signal.
- **Using a threshold mechanically** ignores the model's purpose and domain context.

---

## 11. Prediction Versus Explanation

The best response depends on the objective.

|Goal|Typical priority|Possible response|
|---|---|---|
|Prediction|Stable validation performance|Ridge, Elastic Net, or careful feature reduction|
|Coefficient interpretation|Defensible individual effects|Remove or combine redundant predictors using domain knowledge|
|Causal analysis|A valid causal design and assumptions|Do not treat VIF alone as a causal diagnostic|
|Dimensionality reduction|Compact independent representation|PCA or another representation method|

A model can be acceptable for prediction and unsuitable for explaining individual feature effects. These are different standards.

---

## 12. Example

Suppose a house-price model includes `area_m2` and `area_ft2`:

```text
area_ft2 \approx 10.764 \times area_m2
```

The model may produce coefficients such as:

```text
area_m2:  +1200
area_ft2:   -105
```

The signs and sizes look surprising, but their combined contribution may still be sensible because the two columns describe the same quantity.

A better design is to keep one area variable, or to use a regularized model when both representations must remain for a specific reason.

---

## 13. Frequently Asked Questions

### 13.1. Is multicollinearity the same as correlation?

No. Pairwise correlation is one way to detect a simple form of overlap. Multicollinearity can involve a predictor explained by a combination of several other predictors.

### 13.2. Is multicollinearity always bad?

No. It is mainly harmful when you need stable, separate coefficient interpretations or when numerical calculations become unstable. It may have little effect on predictive accuracy.

### 13.3. Should I always remove a feature with a high VIF?

No. A VIF threshold is a warning signal, not an automatic deletion rule. Consider the feature's meaning, data quality, model goal, and validation results.

### 13.4. Does regularization fix multicollinearity?

Regularization can make estimates more stable and improve prediction, but it changes the coefficient estimation problem. It does not make correlated predictors represent independent information.

### 13.5. Can multicollinearity occur in classification?

Yes. Correlated predictors can also make coefficients unstable in Logistic Regression and other models. The exact diagnostics and consequences depend on the estimator.

---

## 14. 30-Second Revision

- Multicollinearity means that predictors contain overlapping linear information.
- Perfect multicollinearity makes the ordinary least-squares solution non-unique.
- Near multicollinearity can make coefficients unstable and standard errors large.
- Pairwise correlation is useful but cannot detect every multicollinear relationship.
- VIF measures how much a coefficient's variance is inflated by the other predictors.
- Standardization changes scale but does not remove multicollinearity.
- Remove, combine, or transform features when interpretation matters.
- Ridge and Elastic Net can improve stability and prediction.
- Choose the remedy according to the model's purpose.

---

## Related Chapters

- [Linear Regression](01-Linear_Regression.md)
- [Regularization](04-Regularization.md)
- [Model Evaluation and Model Selection](12-Model_Evaluation_and_Model_Selection.md)
- [Linear Algebra](../math-for-ml/01-Linear_Algebra.md)
- [Types of Matrices](../math-for-ml/Appendix-A_Types_of_Matrices.md)
