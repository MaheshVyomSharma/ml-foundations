# 11. Similarity Measures

> *"Before deciding whether two things belong together, we must first decide how to measure their similarity."*

---

## 1. Why Should I Care?

Several Machine Learning algorithms depend on comparing data points.

For example:

- **KNN** finds the nearest neighbours.
- **K-Means** assigns samples to the nearest centroid.
- **Hierarchical Clustering** repeatedly merges the closest clusters.

All of these algorithms rely on one fundamental question:

> **How do we measure how similar or different two samples are?**

Similarity measures answer this question.

Choosing an appropriate measure can significantly affect the quality of a Machine Learning model.

---

## 2. Five-Minute Story

Suppose you want to choose a roommate.

How should you compare two people?

You could compare:

- Age
- Food preferences
- Sleeping habits
- Music taste
- Cleanliness

Different comparisons may lead to different conclusions.

Likewise, Machine Learning must decide what "similar" means before it can group or classify data.

---

## 3. Learning Objectives

After reading this chapter, you should be able to:

- Explain why similarity measures are important.
- Distinguish between distance and similarity.
- Describe common distance metrics.
- Identify when each similarity measure is appropriate.
- Select an appropriate measure for a given problem.

---

## 4. What are Similarity Measures?

**Similarity measures** are mathematical methods used to quantify how alike or different two samples are.

Some measures produce a **distance**, where smaller values indicate greater similarity.

Others produce a **similarity score**, where larger values indicate greater similarity.

---

## 5. Problem They Solve

Many Machine Learning algorithms make decisions based on how close two samples are.

Without a method for measuring closeness, algorithms such as KNN and clustering cannot determine which samples belong together.

Similarity measures provide a consistent way to compare samples.

---

## 6. Key Terminology

| Term | Meaning |
|------|---------|
| Distance | Numerical measure of how far apart two samples are. |
| Similarity | Numerical measure of how alike two samples are. |
| Feature Space | The multidimensional space formed by all features. |
| Vector | Numerical representation of a sample in feature space. |
| Distance Metric | Mathematical formula used to measure distance. |

---

## 7. Intuition

Imagine two cities connected by roads.

There are many ways to measure the distance between them.

- Straight-line distance.
- Driving distance.
- Walking distance.

Each serves a different purpose.

Similarly, Machine Learning offers different ways to compare data depending on the problem.

---

## 8. Mental Picture

Imagine plotting every sample as a point on a map.

The closer two points are, the more similar they are likely to be.

Different similarity measures simply define different ways of measuring the distance between those points.

---

## 9. Euclidean Distance

Euclidean Distance measures the straight-line distance between two points.

It is the most commonly used distance metric.

### 9.1. Best suited for

- Continuous numerical features
- KNN
- K-Means
- Low- to moderate-dimensional datasets

### 9.2. Advantages

- Simple
- Intuitive
- Widely used

### 9.3. Limitations

- Sensitive to feature scaling.
- Less effective in very high-dimensional spaces.

---

## 10. Manhattan Distance

Manhattan Distance measures the distance travelled along horizontal and vertical paths, similar to navigating city streets laid out in a grid.

### 10.1. Best suited for

- Grid-like movement
- High-dimensional numerical data
- Data containing outliers

### 10.2. Advantages

- Less affected by outliers than Euclidean Distance.
- Easy to compute.

### 10.3. Limitations

- May not represent true straight-line distance.

---

## 11. Minkowski Distance

Minkowski Distance is a generalized distance measure.

It includes:

- Manhattan Distance
- Euclidean Distance

as special cases.

By adjusting a parameter, different distance behaviours can be obtained.

---

## 12. Cosine Similarity

Cosine Similarity measures the angle between two vectors rather than the distance between them.

Two vectors pointing in the same direction have a similarity close to **1**, even if their magnitudes differ.

### 12.1. Best suited for

- Text analysis
- Document similarity
- Natural Language Processing
- High-dimensional sparse data

### 12.2. Advantages

- Ignores vector magnitude.
- Excellent for comparing text documents.

### 12.3. Limitations

- Does not consider absolute distance.

---

## 13. Jaccard Similarity

Jaccard Similarity compares two sets by measuring how many elements they share.

It is defined as:

> **Intersection ÷ Union**

### 13.1. Best suited for

- Binary features
- Set comparisons
- Recommendation systems
- Market basket analysis

### 13.2. Advantages

- Simple interpretation.
- Ideal for presence-or-absence data.

### 13.3. Limitations

- Ignores feature frequency.
- Not suitable for continuous numerical data.

---

## 14. Choosing the Right Measure

| Data Type | Recommended Measure |
|-----------|---------------------|
| Numerical | Euclidean Distance |
| Numerical with outliers | Manhattan Distance |
| Generalized numerical | Minkowski Distance |
| Text documents | Cosine Similarity |
| Binary or set data | Jaccard Similarity |

---

## 15. Feature Scaling

Distance-based measures such as Euclidean and Manhattan Distance are sensitive to differences in feature scale.

Scaling the features before applying these measures is usually recommended.

Cosine Similarity is generally less affected because it depends on vector direction rather than magnitude.

---

## 16. Advantages

- Provides a consistent way to compare samples.
- Supports many Machine Learning algorithms.
- Flexible for different data types.
- Essential for distance-based learning.

---

## 17. Limitations

- No single measure is best for every problem.
- Poor feature scaling can distort distance calculations.
- Performance depends on the characteristics of the dataset.

---

## 18. Common Applications

Similarity measures are widely used in:

- KNN
- K-Means
- Hierarchical Clustering
- Recommendation systems
- Search engines
- Text mining
- Natural Language Processing
- Image retrieval

---

## 19. Memory Hook

> **Distance tells you how far apart two samples are.**
>
> **Similarity tells you how alike they are.**

Remember:

- **Euclidean → Straight line**
- **Manhattan → City blocks**
- **Cosine → Angle**
- **Jaccard → Shared items**

---

## 20. Common Mistakes

- Forgetting feature scaling for distance-based algorithms.
- Using Euclidean Distance for text documents.
- Confusing distance with similarity.
- Assuming one metric works well for every dataset.

---

## 21. Frequently Asked Questions

### 21.1. Why are there so many similarity measures?

Different datasets have different characteristics.

A measure that works well for numerical data may perform poorly on text or binary data.

---

### 21.2. Which similarity measure is used in KNN?

Most implementations use Euclidean Distance by default, although other distance metrics can also be used.

---

### 21.3. Why is Cosine Similarity popular in NLP?

Because documents with similar topics often have similar **directions** in vector space, even if their lengths differ.

---

## 22. 30-Second Revision

- Similarity measures compare samples.
- Distance and similarity are related but not identical.
- Euclidean → Straight-line distance.
- Manhattan → Grid distance.
- Minkowski → Generalized distance.
- Cosine → Angle between vectors.
- Jaccard → Shared elements between sets.
- Choosing the right measure depends on the data and the problem.

---

## 23. Looking Ahead

The next chapter shifts focus from **building models** to **evaluating them**.

A model is only useful if we can measure how well it performs.

In the next chapter, we study **Model Evaluation**, including the Confusion Matrix, Accuracy, Precision, Recall, F1 Score, ROC-AUC, regression metrics, and techniques for selecting reliable models.
