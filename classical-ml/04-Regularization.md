# 04. Regularization

> *"A model should be complex enough to learn, but not so complex that it memorizes."*

---

## 1. Why Should I Care?

A Machine Learning model can perform extremely well on training data and still fail badly on new data.

This happens when the model learns not only the useful patterns, but also the noise and accidental details in the training set.

This problem is called **overfitting**.

Regularization helps control overfitting by discouraging the model from becoming unnecessarily complex.

---

## 2. Five-Minute Story

Imagine preparing for an exam using ten solved question papers.

One student memorizes the exact answers word for word.

Another student understands the underlying concepts.

If the actual exam contains the same questions, both students may perform well.

But if the questions are slightly different, the student who memorized everything may struggle.

The student who learned the general ideas is more likely to adapt.

Machine Learning models face the same problem.

A model that memorizes the training data may perform poorly on unseen data.

Regularization encourages the model to learn the broader pattern instead.

---

## 3. Learning Objectives

After reading this chapter, you should be able to:

- Explain why overfitting occurs.
- Describe the purpose of regularization.
- Understand the difference between L1 and L2 regularization.
- Explain how regularization affects model coefficients.
- Identify when regularization is useful.

---

## 4. What is Regularization?

**Regularization** is a technique used to reduce overfitting by adding a penalty for model complexity during training.

Instead of minimizing only prediction error, the model also considers the size of its learned coefficients.

This encourages the model to avoid unnecessarily large parameter values.

---

## 5. Problem It Solves

A complex model may fit the training data extremely closely.

However, that does not necessarily mean it has learned the true underlying pattern.

It may instead be fitting:

- Noise
- Random fluctuations
- Outliers
- Accidental relationships

Regularization reduces this tendency by discouraging excessive model complexity.

---

## 6. Key Terminology

| Term | Meaning |
|------|---------|
| Overfitting | Performing well on training data but poorly on unseen data. |
| Generalization | Ability of a model to perform well on new data. |
| Penalty | Additional cost imposed on model complexity. |
| Coefficient | Weight assigned to a feature by the model. |
| Regularization Strength | Controls how strongly complexity is penalized. |

---

## 7. Intuition

Suppose a model is allowed to assign extremely large weights to individual features.

It may use these large values to fit small irregularities in the training data.

Regularization tells the model:

> "You may improve prediction accuracy, but large coefficients come at a cost."

The model must therefore balance two goals:

1. Fit the data well.
2. Keep the model reasonably simple.

---

## 8. Mental Picture

Imagine fitting a flexible wire through a set of points.

Without constraints, you could bend the wire sharply so that it passes through almost every point.

That may look impressive, but the shape becomes unnecessarily complicated.

Regularization acts like stiffness in the wire.

It still bends enough to follow the overall trend, but resists extreme twists caused by noise.

---

## 9. How Regularization Works

Without regularization, training focuses mainly on minimizing prediction error.

With regularization, the objective becomes:

```text
Prediction Error
        +
Complexity Penalty
```

The model must now find parameters that provide good predictions without becoming excessively complex.

---

## 10. Regularization Strength

The amount of regularization is controlled by a parameter.

A larger regularization strength means:

- Stronger penalty
- Smaller coefficients
- Simpler model

A smaller regularization strength means:

- Weaker penalty
- More freedom for coefficients
- Greater risk of overfitting

Too much regularization can also be harmful because the model may become too simple.

This leads to **underfitting**.

---

## 11. L1 Regularization

L1 Regularization penalizes the **absolute values** of model coefficients.

It is commonly associated with **Lasso Regression**.

One important effect of L1 regularization is that some coefficients may become exactly zero.

This means L1 can effectively remove less useful features from the model.

---

## 12. L2 Regularization

L2 Regularization penalizes the **squared values** of model coefficients.

It is commonly associated with **Ridge Regression**.

Instead of forcing coefficients to zero, L2 usually makes them smaller.

As a result, most or all features remain in the model, but their influence is reduced.

---

## 13. L1 vs L2

| Property | L1 | L2 |
|----------|----|----|
| Common Name | Lasso | Ridge |
| Penalizes | Absolute coefficient values | Squared coefficient values |
| Can produce zero coefficients | Yes | Usually no |
| Can perform feature selection | Yes | No |
| Keeps most features | Not always | Usually yes |

---

## 14. Elastic Net

**Elastic Net** combines both L1 and L2 regularization.

It can reduce coefficient size while also allowing some coefficients to become zero.

This makes it useful when both regularization effects are desirable.

---

## 15. Advantages

- Reduces overfitting.
- Improves generalization.
- Controls excessive coefficient values.
- Can improve model stability.
- L1 can perform automatic feature selection.

---

## 16. Limitations

- Too much regularization can cause underfitting.
- Regularization strength must be chosen carefully.
- L1 may remove useful features if the penalty is too strong.
- Regularization does not solve every cause of poor model performance.

---

## 17. Common Applications

Regularization is commonly used with:

- Linear Regression
- Logistic Regression
- Neural Networks
- High-dimensional datasets
- Models with many correlated or weak features

---

## 18. Memory Hook

> **Regularization adds a price for complexity.**

And:

> **L1 can remove features.**
>
> **L2 usually shrinks them.**

---

## 19. Common Mistakes

- Assuming stronger regularization is always better.
- Confusing regularization with feature scaling.
- Believing regularization completely eliminates overfitting.
- Forgetting that excessive regularization can cause underfitting.
- Assuming L1 and L2 behave identically.

---

## 20. Frequently Asked Questions

### 20.1. Is regularization a Machine Learning algorithm?

No.

Regularization is a technique used while training certain Machine Learning models.

---

### 20.2. What is the difference between Ridge and Lasso Regression?

Ridge uses L2 regularization and usually shrinks coefficients.

Lasso uses L1 regularization and can shrink some coefficients exactly to zero.

---

### 20.3. Why can L1 perform feature selection?

Because its penalty can force some feature coefficients to become exactly zero.

A feature with a zero coefficient no longer contributes to the model's prediction.

---

### 20.4. Can Logistic Regression use regularization?

Yes.

Regularization is commonly applied to Logistic Regression to reduce overfitting and control coefficient size.

---

## 21. 30-Second Revision

- Regularization helps reduce overfitting.
- Adds a complexity penalty during training.
- Encourages smaller coefficients.
- L1 = Lasso.
- L2 = Ridge.
- L1 can produce zero coefficients.
- L2 usually shrinks coefficients without removing them.
- Elastic Net combines L1 and L2.
- Too much regularization can cause underfitting.

---

## 22. Looking Ahead

Regularization helps linear models control complexity.

But some datasets contain patterns that cannot be captured well by a single linear relationship or decision boundary.

The next chapter introduces **Decision Trees**, which learn by repeatedly splitting the data using simple decision rules.
