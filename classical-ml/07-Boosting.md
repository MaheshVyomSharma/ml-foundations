# 07. Boosting

> *"Learn from your mistakes, then do better the next time."*

---

# Why Should I Care?

Random Forest improves performance by training many Decision Trees **independently**.

But what if each new tree could learn from the mistakes made by the previous trees?

Instead of treating every tree equally, we could allow each new tree to focus on the errors that still remain.

This is the central idea behind **Boosting**.

Boosting often produces some of the most accurate Machine Learning models used in practice.

---

# Five-Minute Story

Imagine a mathematics teacher correcting an exam.

The first revision session focuses on algebra.

Most students improve, but some still struggle with geometry.

The second revision session concentrates only on geometry.

After that, a few students still have difficulty with trigonometry.

The third session focuses mainly on trigonometry.

Instead of teaching every topic equally every time, the teacher spends more effort on the remaining weaknesses.

Boosting works in exactly the same way.

Each new model focuses on correcting the mistakes made by the previous models.

---

# Learning Objectives

After reading this chapter, you should be able to:

- Explain the idea behind Boosting.
- Understand how Boosting differs from Bagging.
- Describe how trees are trained sequentially.
- Identify common Boosting algorithms.
- Explain the advantages and limitations of Boosting.

---

# What is Boosting?

**Boosting** is a **supervised ensemble learning technique** in which multiple weak learners are trained **sequentially**, with each learner attempting to correct the errors made by the previous learners.

The predictions from all learners are then combined to produce a strong final model.

---

# Problem It Solves

Random Forest reduces overfitting by averaging many independent trees.

However, every tree is trained without considering the mistakes made by the others.

Boosting improves performance by allowing each new tree to pay more attention to the observations that previous trees predicted incorrectly.

---

# Key Terminology

| Term | Meaning |
|------|---------|
| Weak Learner | A simple model that performs only slightly better than random guessing. |
| Ensemble | Multiple models working together. |
| Sequential Learning | Training models one after another. |
| Residual | The error remaining after a prediction. |
| Learning Rate | Controls how much each new learner contributes to the final model. |

---

# Intuition

Suppose ten students are solving the same puzzle.

In Random Forest, all ten students work independently and then vote on the answer.

In Boosting, Student 2 first reviews Student 1's mistakes.

Student 3 reviews the remaining mistakes.

Student 4 reviews what is still incorrect.

Each student builds upon the work of the previous student.

The final solution is usually much better than the first attempt.

---

# Mental Picture

Imagine painting a wall.

The first coat covers most of the surface.

The second coat fills the missed patches.

The third coat improves the remaining imperfections.

Each coat improves the previous one rather than starting from scratch.

Boosting behaves similarly.

---

# How Boosting Works

The training process follows these steps:

1. Train a simple model.
2. Measure the remaining prediction errors.
3. Train another model that focuses more on those errors.
4. Repeat this process many times.
5. Combine the predictions from all models.

```text
Training Data
      │
      ▼
Tree 1
      │
 Remaining Errors
      ▼
Tree 2
      │
 Remaining Errors
      ▼
Tree 3
      │
      ▼
...
      │
      ▼
Combined Prediction
```

---

# Boosting vs Bagging

| Bagging | Boosting |
|---------|----------|
| Models are trained independently. | Models are trained sequentially. |
| All models have equal importance. | Later models focus on earlier mistakes. |
| Reduces variance. | Reduces bias and often improves accuracy. |
| Example: Random Forest | Examples: AdaBoost, Gradient Boosting, XGBoost |

---

# Popular Boosting Algorithms

## AdaBoost

AdaBoost increases the importance of incorrectly classified samples so that later learners pay more attention to them.

---

## Gradient Boosting

Gradient Boosting trains each new tree to predict the **remaining errors (residuals)** of the previous trees.

---

## XGBoost

XGBoost (Extreme Gradient Boosting) is an optimized implementation of Gradient Boosting.

It introduces several improvements, including:

- Faster training
- Regularization
- Better handling of missing values
- Parallel processing

It is widely used in Machine Learning competitions and many real-world applications.

---

## LightGBM

LightGBM is another optimized Gradient Boosting implementation designed for speed and efficiency on large datasets.

---

# Common Hyperparameters

| Hyperparameter | Purpose |
|---------------|---------|
| n_estimators | Number of trees. |
| learning_rate | Contribution of each tree to the final model. |
| max_depth | Maximum depth of each tree. |
| subsample | Fraction of training data used by each tree. |

---

# Advantages

- Often achieves very high predictive accuracy.
- Learns complex relationships.
- Works well for structured (tabular) data.
- Widely used in industry and data science competitions.

---

# Limitations

- Training is slower than Bagging because trees are built sequentially.
- More sensitive to noisy data and outliers.
- Hyperparameter tuning is often important.
- Individual models are difficult to interpret.

---

# Common Applications

Boosting is commonly used in:

- Credit scoring
- Fraud detection
- Customer churn prediction
- Recommendation systems
- Medical diagnosis
- Ranking and search systems

---

# Memory Hook

> **Random Forest asks many experts for independent opinions.**
>
> **Boosting hires one expert at a time, and each expert fixes the previous expert's mistakes.**

---

# Common Mistakes

- Confusing Boosting with Bagging.
- Assuming all ensemble methods work the same way.
- Using a very large learning rate.
- Ignoring the risk of overfitting when too many trees are added.

---

# Frequently Asked Questions

### Is Boosting a single algorithm?

No.

Boosting is a family of ensemble techniques that includes AdaBoost, Gradient Boosting, XGBoost, LightGBM and others.

---

### Why is Boosting often more accurate than Random Forest?

Because each new learner focuses on correcting the errors that remain after previous learners have been trained.

---

### Why is Boosting slower?

Because every tree depends on the trees trained before it, preventing fully independent training.

---

# 30-Second Revision

- Supervised ensemble learning technique.
- Trains models sequentially.
- Each learner corrects previous errors.
- Often achieves very high accuracy.
- Common algorithms: AdaBoost, Gradient Boosting, XGBoost, LightGBM.
- More accurate than a single tree, but slower to train.

---

# Looking Ahead

So far, every supervised algorithm has learned from **labelled data**.

But what happens when no labels are available?

The next chapter introduces **K-Nearest Neighbors (KNN)**, a simple yet powerful algorithm that makes predictions by looking at the most similar examples in the training data.