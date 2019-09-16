# Introduction

This is a repository of Jingtun ZHANG’s 2019 summer intern @University of California Santa Barbara

>   Paper reading note can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/tree/master/paper-reading-note)
>
>   Studying note can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/tree/master/Studying%20Note)
>
>   Weekly report can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/tree/master/weekly-report)

# Graph Neural Network Survey

## Background

### Origin GNN

**Target problem**: learn a state embedding $$ \mathbf{h}_v\in \mathbb{R}^{s}$$ for each node

**Traditional procedure**:

>   $$\mathbf{h}_{v} = f(\mathbf{x}_v,edge\_attr_v,\mathbf{h}_u,\mathbf{x}_u)$$
>   $$u$$ means neighbors of $$v$$, $$f$$ is local parameterized transition function
>   $$\mathbf{o}_v = g(\mathbf{h}_v, \mathbf{x}_v)$$
>  $$g$$ is local output function

**Typical loss**:

> $$loss = \sum_{i = 1}^{p}(target_i - output_i)$$

Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/Studying%20Note/README.md)

### From spectral to propagation: Graph Convolutional Network (GCN)

**$g_\theta(\Lambda)$ can be well-approximated by a truncated expansion in terms of Chebyshev polynomials $T_k(x)$ up to $K^{th}$ order:** 

$g_{\theta'}(\Lambda) \approx \sum^{K}_{k=0}\theta^{'}_{k} T_k(\widetilde{\Lambda})$
1.  $\widetilde{\Lambda} = \frac{2}{\lambda_{max}} \Lambda - I_N$, $\lambda_{max}$ denotes the largest eigenvalue of $L$
2.  $\theta' \in \mathbb{R}^{K}$: vector of Chebyshev coefficients
3.  $Chebyshev \quad polynomial: T_0(x) = 1, \quad T_1(x) = x, \quad T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)$
4.  SO: $g_{\theta '} \star x \approx U \sum^{K}_{k=0}\theta^{'}_{k} T_k(\widetilde{\Lambda})U^{\top}x = \sum^{K}_{k=0}\theta^{'}_{k} T_k(\widetilde{L})x$
1.   $\widetilde{L} = \frac{2}{\lambda_{max}}L - I_N$ 
2.  $K^{th}-order$ polynomial in the Laplacian: it depends only on nodes that are at maximum K steps away from the cantral node
3.  with $K = 1$ and $\lambda_{max} \approx 2$:
$g_{\theta '} \star x \approx \theta_{0}^{'}x+\theta^{'}_{1}(L-I_N)x = \theta^{'}_{0}x - \theta^{'}_{1}D^{-\frac{1}{2}}AD^{-\frac{1}{2}}x$
with $\theta = \theta^{'}_{0} = -\theta^{'}_{1}$ and renormalization trick $I_N + D^{-\frac{1}{2}}AD^{-\frac{1}{2}} \rightarrow \widetilde{D}^{-\frac{1}{2}} \widetilde{A} \widetilde{D}^{-\frac{1}{2}}$
from C input channelsand F filters:
$Z = \widetilde{D}^{-\frac{1}{2}} \widetilde{A} \widetilde{D}^{-\frac{1}{2}}X\Theta$: complexity:$\mathcal{O}(|\varepsilon|FC)$
1. $\Theta \in \mathbb{R}^{C \times F}$: matrix of filter parameters
2. $Z\in \mathbb{R}^{N \times F}$: convolved signal matrix

**Layer-wise propagation rule:**
$H^{(l+1)} = \sigma( \widetilde{D}^{-\frac{1}{2}} \widetilde{A} \widetilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})$
$\widetilde{A} = A + I_N$: adjacency matrix with added self connection
$\widetilde{D}_{ii} = \sum_j \widetilde{A}_{ij}$:
$W^{(l)}$: layer-specific trainable weight matrix
$\sigma(\cdot)$:  activation function
$H^{(l)} \in \mathbb{R}^{N \times D}$: the matirx of activations in the $l^{th}$ layer, $H^{(0)} = X$

Details can be found at [Github link]([https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/Studying%20Note/GCN.md](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/Studying Note/GCN.md)

### Why GNN



## Models



## General Model Project: PyTorch Geometric

Github Project [rusty1s/pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) implements many important GNN models with general GNN model Message Passing Neural Network, and builds an end-to-end graph data loading to testing model architecture. Detail studying note can be found at [Github Link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/Studying Note/learn-pytorch-geometric.md)

![1568652726168](figures/1568652726168.png)

I modified this project for the following research: Code can be found at [Github Link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/tree/master/pytorch_geometric)

+   Profiling of GNN models

    Complexity analysis of MPNN network

    Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/weekly-report/weeklyreport0722-0728.pdf)
![ppi_plot_cpu](figures/ppi_plot_cpu.png)

+   Visualization of GNN Pooling function

    Details can be found at [Github link]()

## Hierarchically Aggregated computation Graphs (HAGs)

### Paper reading and Key idea

Represent common neighbors across different nodes using aggregation hierarchies, which eliminates redundant computation and unnecessary data transfers in both GNN training and inference.

![1568650351172](figures/1568650351172.png)

Problems:

+   Maybe Not Optimal, But No Better Idea

+   Did not release code or details the heap maintenance method

+   More Discussion:

    1.  Understanding of the model: Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/weekly-report/weeklyreport0722-0728.pdf)

        ![HAG](figures/HAG.png)

    2.  

### Reimplementation

Details can be found at [Github link]()

# Motion-Vector based video Object Detection

## Tasks

### Proof of mathmatical principal of DFF

We try to proof the existence of a linear transformation that maps the feature map of key frame to the feature map of non-key frames based on the motion information:

>   Given a convolution operation $\mathbb{C}$, $\forall$ frame $\mathcal{A}$ and $\mathcal{B}$, as well as the corresponding feature maps $\mathcal{A'}$ and $\mathcal{B'}$, $\exists$ a linear transformation $\mathcal{T} = \mathcal{C}^{-1}\cdot \mathcal{M}_{\mathcal{A} \to \mathcal{B}} \cdot \mathcal{C}$, such that $\mathcal{B'} = \mathcal{A'} \cdot \mathcal{T} + \mathcal{\delta'}$, where $\mathcal{\delta'} = \mathcal{\delta}C$, where $\mathcal{M}_{\mathcal{A} \to \mathcal{B}}$ and $\mathcal{\delta}$ are motion and error information extracted from motion vector and residual map respectively.

And the error term of residual map will not explode after a sequence of convolution operations:

>   Given a convolution operation $\mathbb{C}$ with unit normality and an error information $\delta \sim \mathcal{N}(0, \sigma^2)$, the error information $\delta'$ after convolution operation enjoys convolution-invariance, *i.e.*, $\delta' = \delta C \sim \mathcal{N}(0, \sigma^2)$.

Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/weekly-report/Rivulet_Proof.pdf)

### Motion Vector Feature Flow Version3



### Motion Vector Feature Flow Version4



## Discuss



# Quantum Computing Learning

## First stage: From bits to qubits: Basical Concepts and Algorithm of Quantum Computing



## ~~Second stage: Great idea evolution and Important Works~~

## ~~Third stage: On-going Front Problem and Research~~

## ~~Fourth stage: Research directions~~

