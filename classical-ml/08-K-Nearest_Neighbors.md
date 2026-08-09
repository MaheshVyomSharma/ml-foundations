# 08. K-Nearest Neighbors (KNN)

> *"Tell me who your closest neighbours are, and I'll tell you who you are."*

---

# Why Should I Care?

Many Machine Learning algorithms spend time learning an explicit mathematical model from the training data.

K-Nearest Neighbors (KNN) takes a completely different approach.

Instead of learning a model beforehand, it waits until a prediction is required.

When a new sample arrives, KNN simply looks for the most similar examples it has already seen and lets them decide the prediction.

---

# Five-Minute Story

Imagine you move into a new neighbourhood.

You know nothing about the area.

You want to know whether it's a good place to live.

Instead of reading reports, you speak to the people living nearest to your house.

If most of your neighbours describe the neighbourhood as safe, you're likely to reach the same conclusion.

If most complain about frequent problems, you'll probably think twice.

KNN follows exactly this principle.

Instead of asking nearby people, it asks the **nearest data points**.

---

# Learning Objectives

After reading this chapter, you should be able to:

- Explain how KNN makes predictions.
- Understand the meaning of **K**.
- Describe how distance is used to identify neighbours.
- Explain why feature scaling is important for KNN.
- Identify the strengths and limitations of KNN.

---

# What is K-Nearest Neighbors?

**K-Nearest Neighbors (KNN)** is a **supervised learning algorithm** that predicts the output for a new sample by examining the **K most similar samples** in the training dataset.

For classification, the neighbours vote for the class.

For regression, the neighbours' target values are averaged.

---

# Problem It Solves

Some datasets do not follow a simple mathematical relationship.

Instead of trying to learn an equation, KNN assumes:

> **Samples that are close to each other are likely to have similar outputs.**

This simple assumption is often surprisingly effective.

---

# Key Terminology

| Term | Meaning |
|------|---------|
| Neighbour | A training sample close to the new sample. |
| K | Number of neighbours considered. |
| Distance Metric | Method used to measure similarity between samples. |
| Query Point | The new sample whose output must be predicted. |
| Majority Vote | Selecting the class chosen by most neighbours. |

---

# Intuition

Suppose a fruit has unknown species.

Nearby fruits in the dataset are:

- Apple
- Apple
- Apple
- Pear
- Apple

Most nearby fruits are apples.

KNN predicts that the unknown fruit is also an apple.

No equations are learned.

The prediction comes entirely from nearby examples.

---

# Mental Picture

Imagine dropping a pin onto a map.

Draw a circle around the pin until it includes **K neighbours**.

Those neighbours determine the prediction.

---

# How KNN Works

The prediction process follows these steps:

1. Choose a value for **K**.
2. Calculate the distance from the query point to every training sample.
3. Identify the K nearest neighbours.
4. Combine their outputs.
5. Return the prediction.

```text
Training Data
      │
      ▼
Calculate Distances
      │
      ▼
Find K Nearest Neighbours
      │
      ▼
Majority Vote
(Classification)

or

Average
(Regression)
```

---

# Choosing the Value of K

The choice of **K** influences the behaviour of the model.

### Small K

- Sensitive to noise.
- Can overfit.

### Large K

- Produces smoother predictions.
- May overlook important local patterns.
- Can underfit.

Selecting an appropriate value of **K** is therefore important.

---

# Distance Metrics

KNN depends entirely on measuring similarity.

Common distance measures include:

- Euclidean Distance
- Manhattan Distance
- Minkowski Distance

The choice of distance metric depends on the problem and the nature of the data.

---

# Why Feature Scaling Matters

Suppose a dataset contains:

- Age (20–60)
- Annual Income (₹2,00,000–₹50,00,000)

Without scaling, income dominates the distance calculation simply because its numerical values are much larger.

As a result, age contributes very little to the prediction.

Feature scaling ensures that all features contribute more fairly to the distance calculation.

> **Remember:** KNN is a distance-based algorithm, so feature scaling is usually essential.

---

# Common Hyperparameters

| Hyperparameter | Purpose |
|---------------|---------|
| n_neighbors | Number of neighbours (K). |
| metric | Distance measure used. |
| weights | Equal weighting or distance-based weighting of neighbours. |

---

# Advantages

- Simple to understand.
- No explicit training phase.
- Naturally handles non-linear decision boundaries.
- Works for both classification and regression.

---

# Limitations

- Prediction becomes slow for large datasets.
- Requires feature scaling.
- Sensitive to irrelevant features.
- Sensitive to noisy data.
- Performance decreases in very high-dimensional datasets.

---

# Common Applications

KNN is commonly used in:

- Recommendation systems
- Image classification
- Pattern recognition
- Medical diagnosis
- Document classification

---

# Memory Hook

> **KNN asks its nearest neighbours before making a decision.**

No equations.

No decision trees.

Just nearby examples.

---

# Common Mistakes

- Forgetting to scale features.
- Choosing K arbitrarily.
- Assuming KNN has a traditional training phase.
- Using KNN on extremely large datasets without considering prediction time.

---

# Frequently Asked Questions

### Is KNN a lazy learning algorithm?

Yes.

KNN stores the training data and delays most of its computation until a prediction is requested.

---

### Can KNN perform regression?

Yes.

Instead of majority voting, it predicts the average value of the nearest neighbours.

---

### Why is KNN slow during prediction?

Because it compares the query point with many or all training samples before making a prediction.

---

# 30-Second Revision

- Supervised learning algorithm.
- Predicts using the K nearest neighbours.
- Uses distance to measure similarity.
- Classification uses majority voting.
- Regression uses averaging.
- Feature scaling is usually required.
- Prediction becomes slower as the dataset grows.

---

# Looking Ahead

KNN predicts labels using nearby examples.

But what if our dataset has **no labels at all**?

Can a Machine Learning algorithm still discover meaningful structure?

The next chapter introduces **K-Means Clustering**, which automatically groups similar samples without using labelled data.