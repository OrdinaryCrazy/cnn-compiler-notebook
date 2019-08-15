# GCN

# SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS (ICLR2017)

paper: [<https://arxiv.org/pdf/1609.02907.pdf>]

code: [<https://github.com/tkipf/gcn>]

## Important Concepts:



## Key Idea:

1.     graph edges need not necessarily encode node similarity, but could contain additional information

2.     layer-wise propagation rule:

      $$H^{(l+1)} = \sigma( \widetilde{D}^{-\frac{1}{2}} \widetilde{A} \widetilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})$$

      $$\widetilde{A} = A + I_N$$: adjacency matrix with added self connection

      $$\widetilde{D}_{ii} = \sum_j \widetilde{A}_{ij}$$:

      $$W^{(l)}$$: layer-specific trainable weight matrix

      $$\sigma(\cdot)$$

## Problem definition:

1.      Graph-based semi-supervised learning: classifying nodes in a graph, where labels are only available for a small subset of nodes
2.     

## Precessed Work:

1.    explicit graph-based regularization 

## Metrix gain:

1.   first-order approximation of spectral graph convolutions 

## Challenge:

