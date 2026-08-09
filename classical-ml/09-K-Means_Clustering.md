# 09. K-Means Clustering

> *"Sometimes the data already contains the answers—we just have to discover the groups."*

---

# Why Should I Care?

So far, every algorithm in this handbook has learned from **labelled data**.

For example:

- House → Price
- Email → Spam
- Customer → Will Buy

But what if no labels exist?

Can a machine still discover useful patterns?

Yes.

This is the purpose of **Clustering**.

K-Means is one of the simplest and most widely used clustering algorithms.

---

# Five-Minute Story

Imagine entering a party where nobody is wearing name tags.

You know nothing about the guests.

After observing for a while, you notice that people naturally begin forming groups.

Some are discussing sports.

Some are talking about technology.

Others gather around the food counter.

Nobody instructed them where to stand.

The groups formed naturally because of similarities.

K-Means works in exactly the same way.

It discovers natural groups within data without being told what those groups are.

---

# Learning Objectives

After reading this chapter, you should be able to:

- Explain what clustering is.
- Understand how K-Means discovers groups.
- Explain the meaning of **K**.
- Describe how cluster centroids are updated.
- Identify the strengths and limitations of K-Means.

---

# What is K-Means Clustering?

**K-Means** is an **unsupervised learning algorithm** that partitions a dataset into **K clusters** based on similarity.

Each sample belongs to the cluster whose **centroid** is nearest to it.

Unlike supervised learning, K-Means does not require labelled data.

---

# Problem It Solves

Suppose a company has thousands of customer records but knows nothing about customer types.

Instead of manually examining every customer, K-Means automatically groups similar customers together.

These groups can then be analysed for business insights.

---

# Key Terminology

| Term | Meaning |
|------|---------|
| Cluster | A group of similar samples. |
| Centroid | The centre of a cluster. |
| K | Number of clusters to create. |
| Assignment Step | Assigning every sample to its nearest centroid. |
| Update Step | Recalculating each centroid after assignments. |

---

# Intuition

Imagine placing **K magnets** on a table covered with iron filings.

Each filing is attracted to its nearest magnet.

Once the filings gather, the magnets are moved to the centre of their groups.

Again the filings rearrange themselves.

This process repeats until the groups stop changing.

K-Means follows the same principle.

---

# Mental Picture

Imagine several circles drawn on a map.

Each house belongs to the nearest circle.

After everyone chooses their nearest circle, the centre of each circle moves to better represent its group.

The process repeats until the circles stabilize.

---

# How K-Means Works

The algorithm follows these steps:

1. Choose the number of clusters (**K**).
2. Randomly initialize K centroids.
3. Assign every sample to its nearest centroid.
4. Recalculate each centroid.
5. Repeat the assignment and update steps until the centroids stop changing significantly.

```text
Choose K
     │
     ▼
Initialize Centroids
     │
     ▼
Assign Samples
     │
     ▼
Update Centroids
     │
     ▼
Converged?
     │
 Yes ─────► Stop
     │
 No
     │
     └──────────────► Repeat
```

---

# Why Are Centroids Updated?

After the first assignment, the initial centroids are rarely in the best locations.

Moving each centroid to the centre of its assigned samples improves the clustering.

Repeated updates gradually produce better cluster centres.

---

# Choosing the Value of K

One of the biggest challenges in K-Means is deciding how many clusters should be created.

Several methods help estimate a suitable value of K.

Common approaches include:

- Elbow Method
- Silhouette Score

These techniques help balance simplicity and clustering quality.

---

# Feature Scaling

Like KNN, K-Means depends heavily on distance calculations.

Features with much larger numerical values can dominate the clustering process.

Feature scaling is therefore usually recommended before applying K-Means.

---

# Common Hyperparameters

| Hyperparameter | Purpose |
|---------------|---------|
| n_clusters | Number of clusters (K). |
| init | Method used to initialize centroids. |
| max_iter | Maximum number of iterations. |
| random_state | Ensures reproducible results. |

---

# Advantages

- Simple to understand.
- Fast on large datasets.
- Easy to implement.
- Works well when clusters are compact and well separated.
- Widely used for customer segmentation.

---

# Limitations

- Number of clusters must be chosen beforehand.
- Sensitive to initialization.
- Sensitive to outliers.
- Assumes clusters are approximately spherical.
- Performance decreases when clusters have very different sizes or densities.

---

# Common Applications

K-Means is widely used in:

- Customer segmentation
- Market analysis
- Image compression
- Document clustering
- Recommendation systems
- Anomaly detection (as a preprocessing step)

---

# Memory Hook

> **K-Means groups similar data by repeatedly finding the centre of each group.**

Think of people naturally forming circles at a social gathering.

---

# Common Mistakes

- Choosing K arbitrarily.
- Forgetting feature scaling.
- Assuming K-Means always finds the optimal clustering.
- Ignoring the effect of random initialization.
- Using K-Means for highly irregular cluster shapes.

---

# Frequently Asked Questions

### Is K-Means supervised or unsupervised?

Unsupervised.

The algorithm receives no labels and discovers groups automatically.

---

### Why is it called K-Means?

"K" represents the number of clusters.

"Means" refers to the centroid, which is calculated as the mean of the samples assigned to a cluster.

---

### Does K-Means always produce the same clusters?

Not necessarily.

Different random initializations may lead to different solutions.

Using an appropriate initialization method and random seed improves consistency.

---

# 30-Second Revision

- Unsupervised learning algorithm.
- Groups similar samples into K clusters.
- Uses centroids to represent clusters.
- Repeatedly performs assignment and update steps.
- Requires choosing K beforehand.
- Usually requires feature scaling.
- Common methods for selecting K: Elbow Method and Silhouette Score.

---

# Looking Ahead

K-Means assigns every sample to exactly one cluster.

But what if we want to understand **how clusters are related**?

What if we want to see groups gradually merge into larger groups?

The next chapter introduces **Hierarchical Clustering**, which builds a tree-like hierarchy of clusters instead of creating all clusters at once.