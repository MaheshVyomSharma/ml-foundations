# 05. Decision Trees

> *"The right question is often more valuable than the right answer."*

---

# Why Should I Care?

Many real-world problems cannot be solved using a single straight line or linear decision boundary.

Consider questions like:

- Will a customer repay a loan?
- Should a patient undergo further medical tests?
- Is a transaction fraudulent?

These decisions are often made by asking a sequence of simple questions.

Decision Trees follow the same idea.

Instead of fitting equations, they learn a series of decision rules that gradually separate the data into different groups.

---

# Five-Minute Story

Imagine visiting a doctor.

The doctor doesn't immediately diagnose your illness.

Instead, they ask a sequence of questions.

- Do you have a fever?
- Yes.

↓

- Do you have a cough?
- Yes.

↓

- Have your symptoms lasted more than three days?
- No.

↓

Possible viral infection.

Each question narrows the possibilities.

A Decision Tree works in exactly the same way.

Instead of medical questions, it learns the questions automatically from the training data.

---

# Learning Objectives

After reading this chapter, you should be able to:

- Explain what a Decision Tree is.
- Understand how a Decision Tree makes predictions.
- Describe how a tree is constructed.
- Explain why Decision Trees can overfit.
- Identify the advantages and limitations of Decision Trees.

---

# What is a Decision Tree?

A **Decision Tree** is a **supervised learning algorithm** that predicts an outcome by repeatedly splitting the data into smaller groups using a sequence of decision rules.

Each split is chosen to separate the data as effectively as possible.

Decision Trees can be used for both:

- Classification
- Regression

In this chapter, we focus primarily on **classification**.

---

# Problem It Solves

Linear models assume that the relationship between features and the target is approximately linear.

Many real-world datasets do not satisfy this assumption.

Decision Trees overcome this limitation by learning **non-linear decision boundaries** through a sequence of simple questions.

---

# Key Terminology

| Term | Meaning |
|------|---------|
| Root Node | The starting point of the tree. |
| Decision Node | A node where a question is asked. |
| Leaf Node | The final prediction produced by the tree. |
| Split | Dividing data into smaller groups based on a feature. |
| Branch | A path connecting two nodes. |
| Depth | The number of levels in the tree. |

---

# Intuition

Imagine playing the game **Twenty Questions**.

The objective is to identify an object by asking questions that eliminate as many possibilities as possible.

For example:

- Is it an animal?
- Does it fly?
- Is it larger than a pigeon?

Each answer removes many incorrect possibilities.

Decision Trees learn these questions automatically from data.

---

# Mental Picture

Visualize a flowchart.

Every box asks a simple question.

Every arrow represents an answer.

Following the arrows eventually leads to a final decision.

That flowchart is a Decision Tree.

---

# How a Decision Tree Works

The training process follows these steps:

1. Start with the entire training dataset.
2. Find the feature that best separates the data.
3. Split the data into smaller groups.
4. Repeat the process for each group.
5. Stop when no useful split remains or a stopping condition is reached.

```text
Entire Dataset
       │
       ▼
 Best Split?
   /       \
 Yes        No
 │           │
Split     Prediction
 │
 ▼
Repeat
```

---

# Choosing the Best Split

At every decision node, the algorithm evaluates many possible questions.

It selects the question that produces the **purest** child groups.

A pure group contains mostly samples from a single class.

Common measures used to evaluate a split include:

- Gini Impurity
- Entropy (Information Gain)

Both aim to produce child nodes that are more homogeneous than the parent node.

---

# Stopping the Tree

A Decision Tree does not grow forever.

Training usually stops when one or more conditions are met, such as:

- Maximum tree depth reached.
- Too few samples remain in a node.
- All samples belong to the same class.
- Further splits provide little improvement.

These conditions help prevent excessive growth.

---

# Overfitting

A very deep tree may memorize the training data instead of learning general patterns.

Such a tree performs well on training data but poorly on unseen data.

This is known as **overfitting**.

Common ways to reduce overfitting include:

- Limiting tree depth.
- Increasing the minimum samples required to split.
- Pruning unnecessary branches.

---

# Common Hyperparameters

| Hyperparameter | Purpose |
|---------------|---------|
| max_depth | Maximum depth of the tree. |
| min_samples_split | Minimum samples required before splitting a node. |
| min_samples_leaf | Minimum samples allowed in a leaf node. |
| criterion | Method used to evaluate split quality (Gini or Entropy). |

---

# Advantages

- Easy to understand and interpret.
- Handles non-linear relationships.
- Requires little data preparation.
- Does not require feature scaling.
- Works for both classification and regression.

---

# Limitations

- Can overfit easily.
- Small changes in data may produce different trees.
- Individual trees may have lower predictive accuracy than ensemble methods.
- Very deep trees become difficult to interpret.

---

# Common Applications

Decision Trees are widely used in:

- Credit approval
- Medical diagnosis
- Customer churn prediction
- Fraud detection
- Risk assessment
- Recommendation systems

---

# Memory Hook

> **Decision Trees learn by asking questions.**

Think of them as an intelligent game of **Twenty Questions**, where every answer narrows the search until only one prediction remains.

---

# Common Mistakes

- Allowing the tree to grow without limits.
- Assuming deeper trees are always better.
- Ignoring overfitting.
- Confusing Decision Trees with Random Forests.

---

# Frequently Asked Questions

### Can Decision Trees perform regression?

Yes.

Decision Trees can predict both numerical values (regression) and categories (classification).

---

### Why don't Decision Trees require feature scaling?

Because they compare feature values using thresholds rather than distances or gradients.

---

### What is pruning?

Pruning is the process of removing unnecessary branches from a Decision Tree to improve its ability to generalize to unseen data.

---

# 30-Second Revision

- Supervised learning algorithm.
- Learns by repeatedly asking questions.
- Works for classification and regression.
- Handles non-linear relationships.
- Uses Gini or Entropy to choose splits.
- Can overfit if allowed to grow too deep.
- Important hyperparameters: max_depth, min_samples_split, min_samples_leaf.

---

# Looking Ahead

A single Decision Tree is easy to understand, but it has one major weakness—it can be unstable and prone to overfitting.

What if, instead of relying on one tree, we trained **hundreds of trees** and combined their predictions?

The next chapter introduces **Random Forest**, an ensemble method that improves accuracy and robustness by allowing many Decision Trees to vote on the final prediction.