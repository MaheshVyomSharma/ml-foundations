# 16. Convolutional Neural Networks

## 1.1 Why Convolutional Neural Networks?

A standard fully connected neural network treats its input as a collection of numerical values.

For an image, we could flatten all pixels into one long vector:

```text
Image
28 × 28 pixels
      ↓
Flatten
      ↓
784 values
      ↓
Dense Neural Network
```

This works mathematically, but it discards an important property of images:

> Nearby pixels are spatially related.

Pixels that form an edge, texture, eye, wheel, or other visual feature occur in local spatial patterns.

A **Convolutional Neural Network (CNN)** is designed to exploit this spatial structure.

---

## 1.2 What Is a CNN?

A **Convolutional Neural Network** is a neural-network architecture that uses convolution operations to detect local patterns in structured grid-like data.

CNNs are particularly associated with images.

Typical applications include:

- image classification
- object detection
- face recognition
- medical imaging
- segmentation
- video analysis

CNNs can also be applied to other structured data such as one-dimensional signals.

---

## 1.3 Why Dense Networks Are Inefficient for Images

Suppose an RGB image has dimensions:

```math
224 \times 224 \times 3
```

The number of input values is:

```math
224 \times 224 \times 3
=
150528
```

If the first dense layer contains 1000 neurons, the number of weights alone would be:

```math
150528 \times 1000
=
150528000
```

That is more than 150 million weights in just the first layer.

CNNs dramatically reduce this parameter count by:

- connecting neurons only to local regions
- sharing the same weights across different image locations

These two ideas are fundamental.

---

## 1.4 Local Connectivity

In a dense layer, every neuron receives every input.

```text
Every Input
   ↓
Every Neuron
```

In a convolutional layer, a neuron looks only at a small local region of the input.

For example:

```text
Full Image

┌───────────────┐
│               │
│    ┌───┐      │
│    │3×3│      │
│    └───┘      │
│               │
└───────────────┘
```

The small region is called the neuron's **receptive field**.

This reflects the intuition that local visual patterns can often be detected from nearby pixels.

---

## 1.5 What Is a Convolution?

A convolution applies a small matrix called a **kernel** or **filter** across an input.

Suppose the input is:

```math
X
```

and the filter is:

```math
K
```

The filter moves across the input spatially.

At each position:

1. select a local patch
2. multiply corresponding values
3. sum the products
4. produce one output value

The resulting collection of outputs is called a **feature map**.

---

## 1.6 Kernel or Filter

A **kernel** is a small matrix of trainable weights.

For example:

```math
K
=
\begin{bmatrix}
k_{11} & k_{12} & k_{13} \\
k_{21} & k_{22} & k_{23} \\
k_{31} & k_{32} & k_{33}
\end{bmatrix}
```

This is a:

```math
3 \times 3
```

kernel.

Unlike manually designed image-processing filters, CNN kernels are normally **learned automatically during training**.

---

## 1.7 A Simple Convolution Calculation

Suppose a local image patch is:

```math
X
=
\begin{bmatrix}
1 & 2 & 0 \\
0 & 1 & 3 \\
2 & 1 & 1
\end{bmatrix}
```

and the kernel is:

```math
K
=
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}
```

At this position, the output is computed by element-wise multiplication followed by summation:

```math
(1)(1)
+
(2)(0)
+
(0)(-1)
+
(0)(1)
+
(1)(0)
+
(3)(-1)
+
(2)(1)
+
(1)(0)
+
(1)(-1)
```

which gives:

```math
-1
```

The same kernel is then moved to another location and the process is repeated.

---

## 1.8 Sliding the Kernel

Conceptually:

```text
Input Image

┌─────────────────┐
│ ┌───┐           │
│ │ K │           │
│ └───┘           │
│                 │
│                 │
└─────────────────┘

        ↓ move

┌─────────────────┐
│   ┌───┐         │
│   │ K │         │
│   └───┘         │
│                 │
│                 │
└─────────────────┘
```

The same filter is reused at each spatial position.

This is called **weight sharing**.

---

## 1.9 Weight Sharing

Suppose a kernel contains:

```math
3 \times 3 = 9
```

weights.

Those same 9 weights are used everywhere the kernel slides.

Without weight sharing, each location would require its own parameters.

Instead:

```text
One Filter
→ Same Weights
→ Many Image Locations
```

This drastically reduces parameter count.

It also gives the CNN an important property:

> A feature learned in one part of the image can be detected elsewhere.

---

## 1.10 Translation Awareness

Suppose a filter learns to detect a vertical edge.

That edge may appear:

```text
Left Side
Centre
Right Side
Top
Bottom
```

Because the same filter moves across the image, it can detect that pattern wherever it occurs.

This makes CNNs naturally suited to visual data where objects can appear at different locations.

---

## 1.11 Feature Map

The output produced by applying one filter across the input is called a **feature map** or **activation map**.

Conceptually:

```text
Input Image
    ↓
3×3 Filter
    ↓
Slide Across Image
    ↓
Feature Map
```

A feature map indicates where a particular learned pattern appears strongly.

For example:

```text
Filter 1
→ vertical edges

Filter 2
→ horizontal edges

Filter 3
→ texture

Filter 4
→ curved pattern
```

The exact features are learned from data.

---

## 1.12 Multiple Filters

A convolutional layer normally contains many filters.

Suppose a layer has:

```math
64
```

filters.

Then it produces:

```math
64
```

feature maps.

Conceptually:

```text
Input
 ↓
Filter 1 → Feature Map 1
Filter 2 → Feature Map 2
Filter 3 → Feature Map 3
...
Filter 64 → Feature Map 64
```

Each filter can learn to detect a different pattern.

---

## 1.13 Channels

A grayscale image contains one channel:

```text
Height × Width × 1
```

An RGB image contains three channels:

```text
Height × Width × 3
```

corresponding to:

```text
Red
Green
Blue
```

A convolutional filter must extend through the full depth of the input.

For an RGB image, a `3 × 3` filter actually has dimensions:

```math
3 \times 3 \times 3
```

---

## 1.14 Kernel Depth

Suppose the input has:

```math
C_{\text{in}}
```

channels.

A filter with spatial size:

```math
K_h \times K_w
```

has dimensions:

```math
K_h \times K_w \times C_{\text{in}}
```

If:

```math
K_h=K_w=3
```

and:

```math
C_{\text{in}}=64
```

one filter contains:

```math
3 \times 3 \times 64
=
576
```

weights, plus usually one bias.

---

## 1.15 Output Channels

The number of filters determines the number of output channels.

If the input has:

```text
32 channels
```

and the layer uses:

```text
64 filters
```

then the output contains:

```text
64 channels
```

Thus:

```text
Input Channels
→ determine filter depth

Number of Filters
→ determine output channels
```

This distinction is important.

---

## 1.16 Stride

**Stride** determines how far the filter moves between positions.

With:

```math
\text{stride}=1
```

the filter moves one pixel at a time.

```text
Position 1
→ move 1 pixel
→ Position 2
```

With:

```math
\text{stride}=2
```

the filter moves two pixels at a time.

Larger strides reduce the spatial dimensions of the output.

---

## 1.17 Effect of Stride

Suppose a filter moves across an image.

### Stride 1

```text
□ → □ → □ → □ → □
```

Many positions are evaluated.

### Stride 2

```text
□   →   □   →   □
```

Fewer positions are evaluated.

Therefore:

```text
Larger Stride
→ Smaller Feature Map
→ Less Computation
→ More Aggressive Downsampling
```

---

## 1.18 Padding

A filter cannot naturally be centred on pixels at the very edge of an image without extending outside the image.

**Padding** adds extra values around the border.

A common choice is zero-padding:

```text
Original

1 2 3
4 5 6
7 8 9
```

After one layer of zero-padding:

```text
0 0 0 0 0
0 1 2 3 0
0 4 5 6 0
0 7 8 9 0
0 0 0 0 0
```

Padding allows filters to process edge regions more fully.

---

## 1.19 Valid Padding

With **valid padding**, no padding is added.

```text
Input
 ↓
Convolution
 ↓
Smaller Output
```

Because the kernel must fit entirely inside the image, the spatial size decreases.

For example, with:

```math
5 \times 5
```

input and:

```math
3 \times 3
```

kernel using stride 1, the output is:

```math
3 \times 3
```

---

## 1.20 Same Padding

With **same padding**, enough padding is added so that, for stride 1, the output retains approximately the same spatial dimensions as the input.

Conceptually:

```text
Input
28 × 28
   ↓
3 × 3 Convolution
Same Padding
   ↓
Output
28 × 28
```

Same padding is widely used in deep CNNs because it allows many convolutional layers to be stacked without rapidly shrinking the feature maps.

---

## 1.21 Output Size Formula

For one spatial dimension, the output size of a convolution is:

```math
O
=
\left\lfloor
\frac{N+2P-K}{S}
\right\rfloor
+
1
```

where:

- `N` = input size
- `P` = padding
- `K` = kernel size
- `S` = stride
- `O` = output size

This formula applies separately to height and width.

---

## 1.22 Output Size Example

Suppose:

```math
N=32
```

```math
K=3
```

```math
P=1
```

```math
S=1
```

Then:

```math
O
=
\frac{32+2(1)-3}{1}+1
```

which gives:

```math
32
```

Thus a `3 × 3` convolution with padding `1` and stride `1` preserves the spatial dimension.

---

## 1.23 Another Output Size Example

Suppose:

```math
N=32
```

```math
K=3
```

```math
P=1
```

and:

```math
S=2
```

Then:

```math
O
=
\left\lfloor
\frac{32+2-3}{2}
\right\rfloor
+
1
```

which gives:

```math
16
```

approximately halving the spatial dimension.

---

## 1.24 Convolutional Layer Parameters

Suppose a convolutional layer has:

```math
C_{\text{in}}
```

input channels,

```math
C_{\text{out}}
```

filters,

and kernel dimensions:

```math
K_h \times K_w
```

Each filter contains:

```math
K_h K_w C_{\text{in}}
```

weights.

With one bias per filter, total trainable parameters are:

```math
C_{\text{out}}
\left(
K_hK_wC_{\text{in}}+1
\right)
```

Importantly, parameter count does **not** depend directly on image width or height.

---

## 1.25 Parameter Count Example

Suppose:

```math
C_{\text{in}}=3
```

```math
C_{\text{out}}=32
```

and:

```math
K_h=K_w=3
```

Then:

```math
32
\left(
3 \times 3 \times 3 + 1
\right)
```

gives:

```math
32(28)=896
```

trainable parameters.

This is dramatically smaller than connecting every image pixel to every output neuron.

---

## 1.26 Activation Functions in CNNs

After convolution, an activation function is usually applied.

For example:

```math
A
=
\mathrm{ReLU}(Z)
```

A typical block is therefore:

```text
Input
 ↓
Convolution
 ↓
ReLU
 ↓
Feature Map
```

Without a non-linear activation, stacked convolutional layers would still behave as compositions of linear transformations.

The same activation-function principles from ordinary neural networks therefore continue to apply.

---

## 1.27 Pooling

**Pooling** reduces the spatial dimensions of feature maps.

A common form is **max pooling**.

Suppose a region contains:

```math
\begin{bmatrix}
1 & 3 \\
2 & 4
\end{bmatrix}
```

Max pooling returns:

```math
4
```

because it selects the maximum value.

---

## 1.28 Max Pooling

A `2 × 2` max-pooling operation examines small regions:

```text
1 3 | 2 1
2 4 | 0 5
----+----
7 2 | 3 6
1 8 | 4 2
```

Taking the maximum from each `2 × 2` block gives:

```text
4 5
8 6
```

The spatial dimensions are reduced.

---

## 1.29 Why Pooling?

Pooling can provide several benefits:

- reduces feature-map dimensions
- reduces computation
- reduces memory usage
- increases effective receptive field
- adds some tolerance to small spatial shifts

Conceptually:

```text
Large Feature Map
       ↓
Pooling
       ↓
Smaller Feature Map
       ↓
Retain Strong Features
```

---

## 1.30 Pooling Has No Trainable Weights

Unlike convolution:

```text
Convolution
→ Learns Filters
```

pooling usually has:

```text
No Trainable Parameters
```

Max pooling simply selects maximum values.

Average pooling calculates averages.

Thus pooling is a fixed operation rather than a learned transformation.

---

## 1.31 Average Pooling

**Average pooling** replaces a local region with its average.

For:

```math
\begin{bmatrix}
1 & 3 \\
2 & 4
\end{bmatrix}
```

the average is:

```math
\frac{1+3+2+4}{4}
=
2.5
```

Max pooling emphasizes the strongest activation.

Average pooling preserves the average response.

Both have applications.

---

## 1.32 Global Average Pooling

**Global Average Pooling (GAP)** reduces each complete feature map to a single value.

Suppose the final convolutional output has:

```text
7 × 7 × 256
```

dimensions.

Global average pooling averages each `7 × 7` feature map, producing:

```text
1 × 1 × 256
```

or simply:

```text
256 values
```

This can replace large fully connected layers and substantially reduce parameter count.

---

## 1.33 Convolution vs Pooling

The distinction is:

```text
Convolution
→ Learn Features

Pooling
→ Reduce Spatial Resolution
```

Convolution contains trainable weights.

Pooling normally does not.

A CNN commonly alternates these operations.

---

## 1.34 Hierarchical Feature Learning

CNN layers often learn increasingly abstract features.

Conceptually:

```text
Raw Pixels
    ↓
Edges
    ↓
Corners / Textures
    ↓
Shapes
    ↓
Object Parts
    ↓
Objects
```

Early convolutional layers often respond to simple patterns.

Deeper layers combine earlier features into more complex representations.

This is an example of **hierarchical representation learning**.

---

## 1.35 First Layers

Early layers operate directly on pixels.

They may learn features such as:

```text
Vertical Edges
Horizontal Edges
Colour Transitions
Simple Textures
```

These patterns are reusable across many objects.

For example, a vertical edge may belong to:

```text
A Car
A Building
A Person
A Chair
```

The early network does not need to know which one yet.

---

## 1.36 Deeper Layers

Later layers receive combinations of earlier feature maps.

They may therefore represent:

```text
Curves
Patterns
Shapes
Object Parts
```

Eventually, high-level representations may correspond strongly to semantic concepts.

Thus:

```text
Layer Depth ↑
→ Feature Abstraction ↑
```

This hierarchical learning is one of the major strengths of CNNs.

---

## 1.37 Receptive Field

The **receptive field** of a neuron is the region of the original input that can influence it.

A neuron in the first convolutional layer with a:

```math
3 \times 3
```

kernel directly sees only a `3 × 3` patch.

However, a neuron in a deeper layer receives information from neurons that themselves looked at neighbouring regions.

Therefore, its effective receptive field becomes larger.

---

## 1.38 Receptive Field Growth

Consider repeated `3 × 3` convolutions.

```text
Layer 1
→ sees small local region

Layer 2
→ combines several Layer 1 regions

Layer 3
→ combines even larger region
```

Thus deeper layers can detect patterns covering increasingly large portions of the image.

This allows the network to progress from local edges toward entire objects.

---

## 1.39 Why Use Small Kernels?

Modern CNNs often use kernels such as:

```math
3 \times 3
```

rather than very large filters.

Stacking multiple small kernels provides:

- fewer parameters
- multiple non-linear transformations
- progressively growing receptive fields

For example, two successive `3 × 3` convolutions can achieve a receptive-field effect similar to a larger kernel while introducing an extra activation function.

---

## 1.40 Basic CNN Architecture

A simple image classifier might look like:

```text
Image
 ↓
Conv(32, 3×3)
 ↓
ReLU
 ↓
MaxPool
 ↓
Conv(64, 3×3)
 ↓
ReLU
 ↓
MaxPool
 ↓
Conv(128, 3×3)
 ↓
ReLU
 ↓
Global Average Pooling
 ↓
Dense Output
 ↓
Prediction
```

The convolutional layers learn features.

Pooling reduces spatial resolution.

The final layer performs the classification.

---

## 1.41 Flattening

Traditional CNN architectures often convert the final feature maps into a vector using **flattening**.

For example:

```text
Feature Maps
7 × 7 × 128
      ↓
Flatten
      ↓
6272 values
      ↓
Dense Layer
```

The calculation is:

```math
7 \times 7 \times 128
=
6272
```

The resulting vector can be passed into an ordinary dense neural-network layer.

---

## 1.42 Why Flattening Can Create Many Parameters

Suppose:

```text
7 × 7 × 512
```

feature maps are flattened.

This produces:

```math
7 \times 7 \times 512
=
25088
```

values.

Connecting these to a dense layer containing 4096 neurons requires:

```math
25088 \times 4096
```

weights.

That is more than 100 million parameters.

This is why modern architectures often prefer global average pooling over very large dense heads.

---

## 1.43 CNN Classification Output

After feature extraction, the output layer follows the same rules already learned.

### Binary Classification

```text
1 Output Neuron
+
Sigmoid
```

### Multiclass Classification

```text
K Output Neurons
+
Softmax
```

Therefore, CNNs change the **feature extraction architecture**, not the fundamental output/loss principles.

---

## 1.44 CNN Loss Functions

The same loss functions remain applicable.

For binary classification:

```text
Sigmoid
+
Binary Cross-Entropy
```

For multiclass classification:

```text
Softmax
+
Categorical Cross-Entropy
```

Backpropagation then propagates gradients through:

```text
Output Layer
 ↓
Dense / Pooling Layers
 ↓
Convolutional Layers
 ↓
Learned Filters
```

CNN filters are trained using the same gradient-based learning principles as ordinary weights.

---

## 1.45 How Filters Learn

CNN kernels do not begin as useful edge or texture detectors.

They begin with initialized weights.

During training:

```text
Random Filter
      ↓
Forward Propagation
      ↓
Prediction
      ↓
Loss
      ↓
Backpropagation
      ↓
Filter Gradient
      ↓
Optimizer Update
      ↓
Improved Filter
```

Over many iterations, filters become useful for the task.

There is no separate algorithm manually teaching the network what an edge is.

---

## 1.46 Parameter Sharing Is the Key Efficiency

A filter might contain only:

```math
3 \times 3 \times 3=27
```

weights.

Yet it can be applied to thousands of locations in an image.

This means:

```text
Few Parameters
→ Reused Across Many Positions
```

This combination of:

```text
Local Connectivity
+
Weight Sharing
```

is the fundamental reason CNNs are so parameter-efficient for images.

---

## 1.47 Translation Equivariance

Convolution has an important property known as **translation equivariance**.

Roughly:

> If an input feature shifts spatially, its resulting feature-map activation also shifts correspondingly.

For example:

```text
Vertical Edge on Left
→ Strong Activation on Left

Move Edge Right
→ Strong Activation Moves Right
```

This is more precise than saying convolution itself is completely translation invariant.

Pooling and later aggregation can introduce some tolerance to exact position.

---

## 1.48 Translation Equivariance vs Invariance

These terms should not be confused.

### Equivariance

```text
Input Moves
→ Output Feature Moves Correspondingly
```

### Invariance

```text
Input Moves
→ Final Prediction Remains Essentially the Same
```

Convolution primarily provides translation equivariance.

Classification architectures aim to develop increasing tolerance or invariance to irrelevant spatial shifts.

---

## 1.49 CNNs vs Dense Neural Networks

| Dense Network | CNN |
|---|---|
| Every neuron connects to every input | Local connections |
| Separate weights for connections | Shared filter weights |
| Ignores spatial structure when flattened | Preserves spatial structure |
| Parameter-heavy for large images | Much more parameter-efficient |
| General-purpose | Particularly suited to grid-like data |

A CNN is still a neural network.

It simply uses a more appropriate connectivity pattern for spatial data.

---

## 1.50 CNN Hyperparameters

Important CNN hyperparameters include:

```text
Number of Convolutional Layers
Number of Filters
Kernel Size
Stride
Padding
Pooling Size
Activation Function
Learning Rate
Batch Size
Regularization
```

For example:

```text
Conv(
    filters = 64,
    kernel = 3 × 3,
    stride = 1,
    padding = same
)
```

contains several architectural hyperparameters.

The kernel **values**, however, are learned parameters.

---

## 1.51 Filters vs Hyperparameters

This distinction is important.

Suppose we specify:

```text
64 filters
3 × 3 kernel size
```

The values:

```text
64
and
3 × 3
```

are architectural hyperparameters.

But the individual numbers inside every kernel:

```math
k_{11},k_{12},\ldots
```

are trainable parameters learned through backpropagation.

---

## 1.52 Data Augmentation and CNNs

CNN image models commonly use data augmentation.

Examples include:

```text
Horizontal Flip
Small Rotation
Crop
Translation
Zoom
Brightness Variation
```

This exposes the network to plausible variations of the training images.

Thus:

```text
Original Dataset
      ↓
Augmentation
      ↓
More Effective Training Variation
      ↓
Reduced Overfitting
```

The chosen transformation must preserve the target meaning.

---

## 1.53 When Augmentation Can Be Wrong

Not every transformation is valid.

Suppose the task distinguishes:

```text
6
from
9
```

Rotating an image by 180 degrees may change the label.

Similarly, horizontal flipping may be inappropriate for some medical or text-related tasks.

Therefore:

```text
Data Augmentation
must preserve
semantic meaning
```

Augmentation should reflect realistic variation in the application domain.

---

## 1.54 Transfer Learning

CNNs are often used through **transfer learning**.

A model is first trained on a large dataset.

Its learned feature extractor is then reused for another task.

Conceptually:

```text
Large Source Dataset
       ↓
Pretrained CNN
       ↓
Learned General Visual Features
       ↓
Adapt to New Dataset
```

The early layers may already detect useful general patterns such as edges and textures.

---

## 1.55 Feature Extraction

One transfer-learning strategy is to freeze the pretrained convolutional layers.

```text
Pretrained CNN
     ↓
Frozen Feature Extractor
     ↓
New Classification Head
     ↓
Train New Head
```

The existing filters are kept fixed.

Only the newly added layers are trained.

This is useful when the new dataset is relatively small.

---

## 1.56 Fine-Tuning

Another strategy is **fine-tuning**.

After training the new output head, some pretrained layers are unfrozen.

```text
Pretrained Network
      ↓
Unfreeze Selected Layers
      ↓
Train with Small Learning Rate
      ↓
Adapt Features to New Task
```

Fine-tuning modifies existing learned representations for the new problem.

---

## 1.57 Feature Extraction vs Fine-Tuning

```text
Feature Extraction
→ Keep pretrained layers fixed
→ Train new head

Fine-Tuning
→ Update some pretrained layers
→ Adapt representations
```

Fine-tuning is more flexible but can require:

- more data
- more computation
- careful learning rates

Both are forms of transfer learning.

---

## 1.58 Modern CNN Architecture Ideas

Several influential CNN architectures introduced important ideas.

Examples include:

```text
LeNet
AlexNet
VGG
Inception
ResNet
EfficientNet
```

For revision purposes, the key conceptual milestones are:

```text
Early CNNs
→ Convolution + Pooling

VGG-style Networks
→ Deep stacks of small kernels

Inception
→ Multiple convolution scales

ResNet
→ Residual / Skip Connections

EfficientNet
→ Systematic scaling of network dimensions
```

The architecture names matter less than understanding the ideas behind them.

---

## 1.59 ResNet Connection

We already encountered residual connections:

```math
y
=
F(x)+x
```

CNNs made this idea especially famous through **Residual Networks (ResNets)**.

Conceptually:

```text
x ───────────────┐
│                ↓
│             Addition
│                ↓
└→ Conv Layers → F(x)
```

Residual paths help gradients flow through very deep convolutional networks.

---

## 1.60 CNN Training Is Still Ordinary Neural-Network Training

Despite the new architecture, the training loop is unchanged:

```text
Image Batch
    ↓
Convolutional Forward Pass
    ↓
Prediction
    ↓
Loss
    ↓
Backpropagation
    ↓
Gradients
    ↓
Optimizer
    ↓
Updated Filters and Weights
```

The major difference is the structure of the layers.

All previously learned concepts remain relevant:

- activations
- loss functions
- gradient descent
- backpropagation
- initialization
- optimizers
- regularization
- dropout
- normalization
- hyperparameter tuning

---

## 1.61 The Big Picture

A CNN performs two broad stages.

### Feature Extraction

```text
Image
 ↓
Convolution
 ↓
Activation
 ↓
Pooling / Downsampling
 ↓
More Convolutions
 ↓
High-Level Feature Maps
```

### Prediction

```text
Learned Features
 ↓
Flatten or Global Pooling
 ↓
Output Layer
 ↓
Prediction
```

Modern CNNs often blur this distinction architecturally, but it remains a useful mental model.

---

## 1.62 Key Takeaways

- CNNs are neural networks designed to exploit spatial structure.
- They are especially useful for image data.
- Dense networks become parameter-heavy when applied directly to large images.
- CNNs use local connectivity and weight sharing.
- A **kernel** or **filter** is a small matrix of trainable weights.
- The filter slides across the input and performs local weighted calculations.
- One filter produces one feature map.
- Multiple filters learn different features.
- Filter depth matches the number of input channels.
- The number of filters determines the number of output channels.
- Stride controls how far a kernel moves between positions.
- Larger strides reduce output spatial dimensions.
- Padding allows filters to process edge regions and control output size.
- Valid padding adds no border values.
- Same padding typically preserves dimensions for stride 1.
- The convolution output dimension is:

```math
O
=
\left\lfloor
\frac{N+2P-K}{S}
\right\rfloor
+
1
```

- Convolutional parameter count depends on kernel size, input channels, and number of filters rather than image width and height.
- ReLU or another non-linear activation is usually applied after convolution.
- Pooling reduces spatial dimensions.
- Max pooling selects the strongest activation in each local region.
- Pooling generally has no trainable parameters.
- CNNs learn hierarchical features, from simple local patterns to complex representations.
- Deeper neurons have larger effective receptive fields.
- Small kernels such as `3 × 3` are widely used.
- Flattening converts feature maps into vectors but can create many dense-layer parameters.
- Global average pooling can reduce this parameter burden.
- CNN output activations and loss functions follow the same rules as other neural networks.
- Filters are learned through backpropagation.
- Convolution is translation equivariant rather than strictly translation invariant.
- Kernel size and number of filters are hyperparameters; kernel values are learned parameters.
- Data augmentation is commonly used to reduce CNN overfitting.
- Transfer learning reuses representations learned from another dataset.
- Feature extraction freezes pretrained layers.
- Fine-tuning updates some pretrained layers for the new task.
- CNNs still use the same fundamental forward-loss-backprop-optimizer training loop.

### Memory Hook

```text
CNN
= Neural Network
+ Spatial Structure

Two Core Ideas:

Local Connectivity
+
Weight Sharing


Kernel / Filter
→ Small Learned Matrix

Filter Slides
→ Local Pattern Detection

One Filter
→ One Feature Map

Many Filters
→ Many Learned Features


Stride
→ How Far to Move

Padding
→ What to Do at the Border

Pooling
→ Reduce Spatial Size


Early CNN Layers
→ Edges / Textures

Middle Layers
→ Shapes / Parts

Deep Layers
→ High-Level Features


CNN Flow:

Image
 ↓
Convolution
 ↓
ReLU
 ↓
Pooling
 ↓
More Convolution
 ↓
Learned Features
 ↓
Classifier
 ↓
Prediction


Most Important Idea:

CNN does not learn
a separate detector
for every image location.

It learns one detector
and reuses it everywhere.
```
