# 03. Logistic Regression

> *"Every algorithm exists because another algorithm had a limitation."*

---

## 1. Why Should I Care?

Many real-world problems require predicting **categories** rather than numbers.

Examples include:

- Is this email spam?
- Will a customer purchase a product?
- Does a patient have a disease?
- Is this transaction fraudulent?

These are classification problems.

Although Linear Regression performs well for predicting numerical values, it cannot reliably solve classification problems.

Logistic Regression was developed to overcome this limitation.

---

## 2. Five-Minute Story

Suppose a bank wants to decide whether to approve a loan application.

For each applicant, it knows:

- Annual income
- Credit score
- Existing debt
- Employment history

The final decision is simple:

- **Approve**
- **Reject**

At first glance, Linear Regression seems like a reasonable choice.

However, imagine the model predicts:

- 1.27
- -0.42
- 2.13

These values have no meaningful interpretation for a yes-or-no decision.

Instead, what we really want is something like:

> "There is an 89% chance this loan should be approved."

Now the prediction makes sense.

This is exactly what Logistic Regression does.

---

## 3. Learning Objectives

After reading this chapter, you should be able to:

- Explain why Linear Regression is unsuitable for classification.
- Describe how Logistic Regression performs classification.
- Understand the role of probability in Logistic Regression.
- Explain the purpose of the sigmoid function.
- Interpret predictions made by a Logistic Regression model.

---

## 4. What is Logistic Regression?

**Logistic Regression** is a **supervised learning algorithm** used for **classification**.

Instead of predicting a continuous value, it predicts the **probability** that a sample belongs to a particular class.

A decision threshold is then applied to convert that probability into a class label.

---

## 5. Problem It Solves

Linear Regression can produce predictions below 0 or above 1.

Probabilities, however, must always lie between 0 and 1.

Logistic Regression solves this problem by transforming its output into a valid probability before making a classification.

---

## 6. Key Terminology

| Term | Meaning |
|------|---------|
| Probability | Likelihood that an event occurs (between 0 and 1). |
| Class | The predicted category. |
| Positive Class | The event of interest (for example, Spam or Disease). |
| Threshold | Probability above which a sample is assigned to the positive class. |
| Sigmoid Function | A mathematical function that converts any real number into a probability between 0 and 1. |

---

## 7. Intuition

Imagine a security guard standing at the entrance of a building.

Every visitor approaches with different characteristics.

The guard estimates:

> "How likely is this person authorised to enter?"

If the confidence is high enough, the gate opens.

Otherwise, access is denied.

Logistic Regression behaves in a similar way.

It first estimates the probability that a sample belongs to the positive class.

It then compares that probability with a threshold to make the final decision.

---

## 8. Mental Picture

Imagine filling a bucket with water.

The water level gradually rises from empty to full.

At some point, the water crosses a marked line.

Below the line:

**No**

Above the line:

**Yes**

Logistic Regression behaves similarly.

It first computes a probability.

Crossing the threshold changes the prediction from one class to another.

---

## 9. How Logistic Regression Works

The learning process consists of the following steps:

1. Learn a weighted combination of the input features.
2. Pass the result through the sigmoid function.
3. Obtain a probability between 0 and 1.
4. Compare the probability with a decision threshold.
5. Assign the corresponding class.

```text
Input Features
       │
       ▼
Weighted Sum
       │
       ▼
Sigmoid Function
       │
       ▼
Probability
       │
       ▼
Decision Threshold
       │
       ▼
Predicted Class
```

---

## 10. The Sigmoid Function

The sigmoid function converts any real-valued input into a probability between 0 and 1.

Its characteristic S-shaped curve ensures that predictions never fall outside the valid probability range.

This makes it suitable for binary classification problems.

---

## 11. Decision Threshold

A threshold determines how probabilities are converted into class labels.

A commonly used threshold is **0.5**.

For example:

| Predicted Probability | Predicted Class |
|-----------------------|-----------------|
| 0.18 | Negative |
| 0.41 | Negative |
| 0.52 | Positive |
| 0.87 | Positive |

> **Note:** The threshold is not always 0.5. It may be adjusted depending on the application.

---

## 12. Training

Like Linear Regression, Logistic Regression learns by minimising a cost function.

The optimisation is typically performed using **Gradient Descent** or related optimisation algorithms.

---

## 13. Advantages

- Simple and interpretable.
- Produces probabilities.
- Fast to train.
- Effective for many binary classification problems.
- Strong baseline classification algorithm.

---

## 14. Limitations

- Assumes a linear decision boundary.
- Less effective for highly complex relationships.
- Sensitive to outliers.
- Performance depends on feature quality.

---

## 15. Common Evaluation Metrics

Classification models are commonly evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- ROC-AUC

These metrics are discussed in detail in the **Model Evaluation** chapter.

---

## 16. Common Applications

Logistic Regression is widely used in:

- Spam detection
- Credit approval
- Medical diagnosis
- Customer churn prediction
- Fraud detection
- Marketing response prediction

---

## 17. Memory Hook

> **Linear Regression predicts numbers.**
>
> **Logistic Regression predicts probabilities.**
>
> **A threshold converts probabilities into classes.**

---

## 18. Common Mistakes

- Assuming Logistic Regression predicts classes directly.
- Believing the threshold must always be 0.5.
- Using accuracy alone for imbalanced datasets.
- Using Logistic Regression for regression problems.

---

## 19. Frequently Asked Questions

### 19.1. Why is it called Logistic Regression if it performs classification?

Historically, the model estimates probabilities using a logistic (sigmoid) function applied to a regression-like linear predictor.

Despite its name, it is primarily used for classification.

---

### 19.2. Can Logistic Regression classify more than two classes?

Yes.

Extensions such as **One-vs-Rest (OvR)** and **Multinomial Logistic Regression** allow classification into multiple classes.

---

### 19.3. Does Logistic Regression use Gradient Descent?

Yes.

Gradient Descent and its variants are commonly used to optimise the model parameters.

---

## 20. 30-Second Revision

- Supervised learning algorithm.
- Used for classification.
- Predicts probabilities.
- Uses the sigmoid function.
- Threshold converts probability into a class.
- Common metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
- Often trained using Gradient Descent.

---

## 21. Looking Ahead

Logistic Regression produces a **linear decision boundary**.

However, many real-world datasets cannot be separated using a single straight line.

The next chapter introduces **Regularization**, a technique that helps prevent models from becoming overly complex and improves their ability to generalise to unseen data.
