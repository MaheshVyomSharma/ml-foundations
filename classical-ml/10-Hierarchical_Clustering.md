# 10. Hierarchical Clustering

> *"Some relationships are best understood by seeing how they grow."*

---

## 1. Why Should I Care?

K-Means divides data into a fixed number of clusters.

But what if you don't know how many clusters should exist?

Or what if you want to understand **how different groups are related to one another**?

Hierarchical Clustering answers these questions by building a hierarchy of clusters instead of creating them all at once.

---

## 2. Five-Minute Story

Imagine organizing a large family reunion.

You first separate everyone into individual people.

Then you begin grouping them:

- Brothers and sisters
- Immediate families
- Cousins
- Entire branches of the family
- The complete family tree

Notice that you never decided beforehand how many groups there should be.

Instead, the groups formed naturally as you moved from individuals to larger families.

Hierarchical Clustering works in a very similar way.

It gradually builds a tree showing how clusters are formed.

---

## 3. Learning Objectives

After reading this chapter, you should be able to:

- Explain the idea behind Hierarchical Clustering.
- Distinguish between agglomerative and divisive clustering.
- Understand the purpose of a dendrogram.
- Explain how clusters are merged.
- Identify the advantages and limitations of Hierarchical Clustering.

---

## 4. What is Hierarchical Clustering?

**Hierarchical Clustering** is an **unsupervised learning algorithm** that groups similar samples by building a hierarchy of clusters.

The result is a tree-like structure called a **dendrogram**, which shows how clusters are related.

Unlike K-Means, Hierarchical Clustering does not require the number of clusters to be specified before training.

---

## 5. Problem It Solves

Sometimes we do not know how many natural groups exist in the data.

Rather than forcing the data into a fixed number of clusters, Hierarchical Clustering allows us to explore the relationships between groups before deciding where to divide them.

---

## 6. Key Terminology

| Term | Meaning |
|------|---------|
| Cluster | A group of similar samples. |
| Dendrogram | A tree diagram showing how clusters are merged or divided. |
| Linkage | Rule used to measure the distance between clusters. |
| Agglomerative | Starts with individual samples and gradually merges them. |
| Divisive | Starts with one large cluster and repeatedly splits it. |

---

## 7. Intuition

Suppose every student in a classroom initially stands alone.

Students with the most similar interests pair up.

Small groups then combine with other similar groups.

Eventually, the entire class becomes one large group.

This gradual merging is the idea behind **Agglomerative Hierarchical Clustering**, the most commonly used form of hierarchical clustering.

---

## 8. Mental Picture

Imagine building a family tree upside down.

At the bottom, every person stands alone.

Moving upward, branches merge together.

At the very top, everyone belongs to one large family.

A dendrogram represents exactly this process.

---

## 9. Types of Hierarchical Clustering

## 10. Agglomerative (Bottom-Up)

Agglomerative clustering begins with every sample in its own cluster.

The algorithm repeatedly merges the two closest clusters until only one cluster remains.

This is the approach most commonly implemented in Machine Learning libraries.

---

## 11. Divisive (Top-Down)

Divisive clustering begins with all samples in a single cluster.

The algorithm repeatedly divides the cluster into smaller groups until each sample forms its own cluster or another stopping condition is reached.

Although conceptually simple, divisive clustering is used less frequently because it is computationally more expensive.

---

## 12. How Agglomerative Clustering Works

The algorithm follows these steps:

1. Place every sample in its own cluster.
2. Compute distances between clusters.
3. Merge the two closest clusters.
4. Update the distances.
5. Repeat until only one cluster remains.

```text
Every Sample
Own Cluster
      │
      ▼
Find Closest Clusters
      │
      ▼
Merge
      │
      ▼
Update Distances
      │
      ▼
Repeat
```

---

## 13. Linkage Methods

When deciding which clusters should merge, the algorithm must define the distance between clusters.

Common linkage methods include:

| Linkage | Description |
|---------|-------------|
| Single Linkage | Uses the closest pair of samples. |
| Complete Linkage | Uses the farthest pair of samples. |
| Average Linkage | Uses the average distance between all pairs of samples. |
| Ward Linkage | Merges clusters that produce the smallest increase in within-cluster variance. |

Different linkage methods can produce different cluster structures.

---

## 14. The Dendrogram

A **dendrogram** is the primary output of Hierarchical Clustering.

It shows:

- How clusters merge.
- The order in which they merge.
- The distance at which they merge.

To obtain the final clusters, imagine drawing a horizontal line across the dendrogram.

Every branch intersected by that line becomes a separate cluster.

---

## 15. Feature Scaling

Like K-Means and KNN, Hierarchical Clustering relies on distance calculations.

Features measured on very different scales can dominate the clustering process.

Feature scaling is therefore usually recommended.

---

## 16. Common Hyperparameters

| Hyperparameter | Purpose |
|---------------|---------|
| n_clusters | Desired number of final clusters. |
| linkage | Method used to merge clusters. |
| metric | Distance measure used to compare samples. |

---

## 17. Advantages

- Does not require specifying K before building the hierarchy.
- Produces an informative dendrogram.
- Can reveal hierarchical relationships within the data.
- Easy to visualize for smaller datasets.

---

## 18. Limitations

- Computationally expensive for large datasets.
- Sensitive to noise and outliers.
- Different linkage methods may produce different results.
- Difficult to update when new data arrives.

---

## 19. Common Applications

Hierarchical Clustering is commonly used in:

- Biological taxonomy
- Gene expression analysis
- Document clustering
- Customer segmentation
- Social network analysis

---

## 20. Memory Hook

> **K-Means creates groups.**
>
> **Hierarchical Clustering creates a family tree of groups.**

---

## 21. Common Mistakes

- Assuming the dendrogram automatically determines the correct number of clusters.
- Ignoring feature scaling.
- Using Hierarchical Clustering on extremely large datasets without considering computational cost.
- Assuming different linkage methods always produce similar clusters.

---

## 22. Frequently Asked Questions

### 22.1. Does Hierarchical Clustering require K beforehand?

No.

The hierarchy is built first.

The desired number of clusters can be chosen later by cutting the dendrogram.

---

### 22.2. What is a dendrogram?

A dendrogram is a tree diagram that shows how clusters are formed through successive merging or splitting.

---

### 22.3. Which type is more common in practice?

Agglomerative Hierarchical Clustering is by far the most commonly used approach.

---

## 23. 30-Second Revision

- Unsupervised learning algorithm.
- Builds a hierarchy of clusters.
- Produces a dendrogram.
- Two approaches: Agglomerative and Divisive.
- Common linkage methods: Single, Complete, Average, Ward.
- Usually requires feature scaling.
- Suitable for exploring relationships between clusters.

---

## 24. Looking Ahead

KNN, K-Means and Hierarchical Clustering all depend on one fundamental idea:

> **How do we measure similarity?**

The next chapter introduces **Similarity Measures**, where we explore distance metrics such as Euclidean, Manhattan, Cosine Similarity and Jaccard Similarity, and understand when each should be used.
