# Table of Contents
------------------

   * [Introduction](#introduction)
   * [Graph Neural Network Survey](#graph-neural-network-survey)
      * [Background](#background)
         * [Origin GNN](#origin-gnn)
         * [From spectral to propagation: Graph Convolutional Network (GCN)](#from-spectral-to-propagation-graph-convolutional-network-gcn)
         * [Why GNN from CNN](#why-gnn-from-cnn)
      * [Models](#models)
      * [System View Optimization](#system-view-optimization)
      * [General Model Project: PyTorch Geometric](#general-model-project-pytorch-geometric)
      * [Hierarchically Aggregated computation Graphs (HAGs)](#hierarchically-aggregated-computation-graphs-hags)
         * [Paper reading and Key idea](#paper-reading-and-key-idea)
         * [Reimplementation](#reimplementation)
   * [Motion-Vector based video Object Detection](#motion-vector-based-video-object-detection)
      * [Tasks](#tasks)
         * [Proof of mathmatical principal of DFF](#proof-of-mathmatical-principal-of-dff)
         * [Motion Vector Feature Flow Version3](#motion-vector-feature-flow-version3)
            * [Idea](#idea)
            * [Result and Discussion](#result-and-discussion)
         * [Motion Vector Feature Flow Version4](#motion-vector-feature-flow-version4)
            * [Idea](#idea-1)
            * [Result and Discussion](#result-and-discussion-1)
         * [Motion Vector Output Flow Step-Performance Curve](#motion-vector-output-flow-step-performance-curve)
   * [Quantum Computing Learning](#quantum-computing-learning)
      * [First stage: From bits to qubits: Basical Concepts and Algorithm of Quantum Computing](#first-stage-from-bits-to-qubits-basical-concepts-and-algorithm-of-quantum-computing)
         * [What Is a QPU?](#what-is-a-qpu)
         * [Native QPU Instructions](#native-qpu-instructions)
         * [Simulator Limitations](#simulator-limitations)
         * [QPU Versus GPU:](#qpu-versus-gpu)
      * [<del>Second stage: Great idea evolution and Important Works</del>](#second-stage-great-idea-evolution-and-important-works)
      * [<del>Third stage: On-going Front Problem and Research</del>](#third-stage-on-going-front-problem-and-research)
      * [<del>Fourth stage: Research directions</del>](#fourth-stage-research-directions)

# Introduction

This is a repository of Jingtun ZHANG’s 2019 summer intern @[University of California Santa Barbara](https://ucsb.edu)

>   Paper reading note can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/tree/master/paper-reading-note)
>
>   Studying note can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/tree/master/Studying%20Note)
>
>   Weekly report can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/tree/master/weekly-report)

# Graph Neural Network Survey

## Background

### Origin GNN

**Target problem**: learn a state embedding <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/tree/master/svgs/63caef3d1bb7532c8580e97fc28abc9c.svg?invert_in_darkmode" align=middle width=56.48009069999999pt height=22.831056599999986pt/> for each node

**Traditional procedure**:

<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/89aef3ba8f13b8e51f9a99bc072c4b9a.svg?invert_in_darkmode" align=middle width=214.20555915pt height=24.65753399999998pt/>
<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode" align=middle width=9.41027339999999pt height=14.15524440000002pt/> means neighbors of <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode" align=middle width=8.55786029999999pt height=14.15524440000002pt/>, <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.81741584999999pt height=22.831056599999986pt/> is local parameterized transition function
<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/5745b8cddc670b3bdfecd8210ba13b1f.svg?invert_in_darkmode" align=middle width=103.80117989999998pt height=24.65753399999998pt/>
<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.430376349999989pt height=14.15524440000002pt/> is local output function

**Typical loss**:

<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/1fa048d44b9c9cc7ca1cd80c4daa2e2a.svg?invert_in_darkmode" align=middle width=225.26571869999998pt height=26.438629799999987pt/>

Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/Studying%20Note/README.md)

### From spectral to propagation: Graph Convolutional Network (GCN)

**<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/50a3cbebd51755abf8b2287dab8f1efe.svg?invert_in_darkmode" align=middle width=39.47872664999999pt height=24.65753399999998pt/> can be well-approximated by a truncated expansion in terms of Chebyshev polynomials <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/2131ee0175003aca3f6b8f1e659a351d.svg?invert_in_darkmode" align=middle width=39.87455174999999pt height=24.65753399999998pt/> up to <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/5fadc8645b0ff23decb30259f696b6a2.svg?invert_in_darkmode" align=middle width=27.798818849999993pt height=27.91243950000002pt/> order:** 

<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/9fa1022a95f840b3916cb818e7bed3a1.svg?invert_in_darkmode" align=middle width=168.36582344999997pt height=32.256008400000006pt/>
1.  <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/fa6df9d90570fd504dcba896dbf0fb28.svg?invert_in_darkmode" align=middle width=119.24148389999998pt height=32.054807400000016pt/>, <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/06fbd0c7c3b3f5de7a0b935888e8b0c3.svg?invert_in_darkmode" align=middle width=35.83868804999999pt height=22.831056599999986pt/> denotes the largest eigenvalue of <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.18724254999999pt height=22.465723500000017pt/>
2.  <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/f234b7926c6a80b34416db37111b64b1.svg?invert_in_darkmode" align=middle width=56.60038724999999pt height=27.6567522pt/>: vector of Chebyshev coefficients
3.  <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/d6522fefa1dec7cec00aba886ebea891.svg?invert_in_darkmode" align=middle width=591.40406655pt height=24.65753399999998pt/>
4.  SO: <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/0599ff9e7cf092b3f73bdfbf097a2f26.svg?invert_in_darkmode" align=middle width=351.95466899999997pt height=32.256008400000006pt/>
1.   <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/b582508ba651e0dc4c5b663cc123a39a.svg?invert_in_darkmode" align=middle width=118.78490249999999pt height=32.054807400000016pt/> 
2.  <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/3627311143d273b6ea342a0e645b80a1.svg?invert_in_darkmode" align=middle width=88.6359639pt height=27.91243950000002pt/> polynomial in the Laplacian: it depends only on nodes that are at maximum K steps away from the cantral node
3.  with <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/6548debba4b05edacf375ef470d3292a.svg?invert_in_darkmode" align=middle width=45.273840149999984pt height=22.465723500000017pt/> and <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/dfc733c40c91c427c0c960c97c6205b0.svg?invert_in_darkmode" align=middle width=66.79741364999998pt height=22.831056599999986pt/>:
<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/5dfcf8757e7375a257af133ce1545daf.svg?invert_in_darkmode" align=middle width=372.0959318999999pt height=31.359338999999984pt/>
with <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/a0a69fd2e98614bf40e59d78ba2e2d5c.svg?invert_in_darkmode" align=middle width=94.15501259999999pt height=30.984656999999984pt/> and renormalization trick <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/3cbc026cc64f019912a421ecd6a298ef.svg?invert_in_darkmode" align=middle width=226.0231314pt height=32.054807400000016pt/>
from C input channelsand F filters:
<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/1d4d30f9114314b32f161fa960c32e0b.svg?invert_in_darkmode" align=middle width=143.73983085pt height=32.054807400000016pt/>: complexity:<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/5278f3634a18926270ed93a90fcd28c4.svg?invert_in_darkmode" align=middle width=68.90538434999999pt height=24.65753399999998pt/>
1. <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/92ab3bf1bb9b9f6fa1951617e9f56342.svg?invert_in_darkmode" align=middle width=75.36350909999999pt height=27.6567522pt/>: matrix of filter parameters
2. <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/e9e0d82387f94195d4af2aeb8d7acb0e.svg?invert_in_darkmode" align=middle width=76.38698925pt height=27.6567522pt/>: convolved signal matrix

**Layer-wise propagation rule:**

<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/01fee8deccbe5360b008b71c69e026ff.svg?invert_in_darkmode" align=middle width=236.8280046pt height=32.054807400000016pt/>

<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/a17dc0e8a7c63fe5b251ce74eb3c746d.svg?invert_in_darkmode" align=middle width=85.53862514999999pt height=32.054807400000016pt/>: adjacency matrix with added self connection

<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/f4a6201c59f29303999fbfca325460f1.svg?invert_in_darkmode" align=middle width=95.75280329999998pt height=32.054807400000016pt/>:

<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/e7e30f89a8c77e163e112c244ba28bf2.svg?invert_in_darkmode" align=middle width=32.30607764999999pt height=29.190975000000005pt/>: layer-specific trainable weight matrix

<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/2507d59b6868c8cdd57220d1830f4c22.svg?invert_in_darkmode" align=middle width=27.334540199999985pt height=24.65753399999998pt/>:  activation function

<img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/816be29d211852d3e93fc18c54747364.svg?invert_in_darkmode" align=middle width=95.30526719999997pt height=29.190975000000005pt/>: the matirx of activations in the <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/0503ad389cbf5fa79e5f6a939670db56.svg?invert_in_darkmode" align=middle width=17.89016294999999pt height=27.91243950000002pt/> layer, <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/790e13a1b7243c0f1033727c2e5f1317.svg?invert_in_darkmode" align=middle width=69.47477129999999pt height=29.190975000000005pt/>

Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/Studying%20Note/GCN.md)

###  Why GNN from CNN

1.   Graphs are the most typical locally connected structure
2.   Share weights reduce the computational cost compared with traditional spectral graph theory
3.   Multilayer structure is the key to deal with hierarchical patterns, which captures the featuofres of various sizes
4.   CNNs or RNNs need a specific order, but there is no natural order of nodes in graph, GNNs output is input order invarient
5.   Human intelligencce is most based on the graph, GNNs can do information propagation guided by the graph structure
6.   GNNs explores to generate the graph from non-structural data

## Models

+   GDyNet and CGCNN model: Application of GNN in materials. Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/weekly-report/weeklyreport0722-0728.pdf)

## System View Optimization

+   Tigr

    **T**ransform **i**rregular **g**raphs into more **r**egular ones such that the graphs can be processed more efficiently on GPU-like architectures while guaranteeing correctness.
    ![1568654140133](figures/1568654140133.png)
    
    Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/Studying%20Note/tigr.md)
    
+ Fast train GNN on dense hardware

    Permute nodes to expose low bandwidth structure and express GNN propagation in terms of application of dense matrix multiply primitive.

    Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/paper-reading-note/fast-training%20of%20sparse%20gnn%20on%20dense%20hardware.pdf)
## General Model Project: PyTorch Geometric

Github Project [rusty1s/pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) implements many important GNN models with general GNN model Message Passing Neural Network, and builds an end-to-end graph data loading to testing model architecture. Detail studying note can be found at [Github Link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/Studying%20Note/learn-pytorch-geometric.md)

![1568652726168](figures/1568652726168.png)

I modified this project for the following research: Code can be found at [Github Link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/tree/master/pytorch_geometric)

+   Profiling of GNN models

    Complexity analysis of MPNN network

    Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/weekly-report/weeklyreport0722-0728.pdf)
![ppi_plot_cpu](figures/ppi_plot_cpu.png)

+   Visualization of Pooling effectiveness

    In topology domain: **Less nodes**
    
    ![pool](figures/pool.png)
    
    In embedding domain: **Maintenance of group structure and similarity**
    
    ![pool2](figures/pool2.png)
    
    Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/GNN-jupyter-code/topk-pooling%20visualization.ipynb)

## Hierarchically Aggregated computation Graphs (HAGs)

### Paper reading and Key idea

Represent common neighbors across different nodes using aggregation hierarchies, which eliminates redundant computation and unnecessary data transfers in both GNN training and inference.

![1568650351172](figures/1568650351172.png)

Problems:

+   Maybe Not Optimal, But No Better Idea

+   Did not release code or details the heap maintenance method

+   More Discussion:

    1.  Understanding of the model: 

        Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/weekly-report/weeklyreport0722-0728.pdf)

        ![HAG](figures/HAG.png)
    
    2.  Detailed description about max redundancy computation: 
    
        Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/weekly-report/weeklyreport0729-0804.pdf)
    
    ![computation and topology](figures/computation and topology.png)

### Reimplementation

Release [HAG model Version 0.0](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/GNN-jupyter-code/HAG.py)

More details can be found at [Tutorial](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/weekly-report/weeklyreport0729-0804.pdf) and [Github build](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/tree/master/HAG)

# Motion-Vector based video Object Detection

## Tasks

### Proof of mathmatical principal of DFF

We try to proof the existence of a linear transformation that maps the feature map of key frame to the feature map of non-key frames based on the motion information:

>   Given a convolution operation <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/81324f07e9ffb7920321df72cc0bee1b.svg?invert_in_darkmode" align=middle width=11.87217899999999pt height=22.648391699999998pt/>, <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/ecea226b5977d1a327732124dccb8969.svg?invert_in_darkmode" align=middle width=9.132448049999992pt height=22.831056599999986pt/> frame <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/7651ba0e8e29ee7537841a819041a172.svg?invert_in_darkmode" align=middle width=13.12555859999999pt height=22.465723500000017pt/> and <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/99fe4d8255dd7318412c8dbe107b71ce.svg?invert_in_darkmode" align=middle width=11.296807499999991pt height=22.465723500000017pt/>, as well as the corresponding feature maps <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/d3ee904bcccbc041c735c11c53596b7c.svg?invert_in_darkmode" align=middle width=16.91551949999999pt height=24.7161288pt/> and <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/8918902f73415dae95eefe679b9e8f05.svg?invert_in_darkmode" align=middle width=15.086763449999989pt height=24.7161288pt/>, <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/42353da95c0a3784bd8339b6e4fb1260.svg?invert_in_darkmode" align=middle width=9.132448049999992pt height=22.831056599999986pt/> a linear transformation <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/f1e615fa8a876c9e41c37d55ec72ed7c.svg?invert_in_darkmode" align=middle width=148.84801634999997pt height=26.76175259999998pt/>, such that <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/df78221dc20a3236f170e0b158d96127.svg?invert_in_darkmode" align=middle width=112.37606654999999pt height=24.7161288pt/>, where <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/0452ab9e585765adc9c60d577453dceb.svg?invert_in_darkmode" align=middle width=55.31028689999999pt height=24.7161288pt/>, where <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/6c392bc6c8ee4343b9bdc921e6bc37ae.svg?invert_in_darkmode" align=middle width=52.356475049999986pt height=22.465723500000017pt/> and <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/ff90e345247bbbad70477ebe6026551e.svg?invert_in_darkmode" align=middle width=7.928075099999989pt height=22.831056599999986pt/> are motion and error information extracted from motion vector and residual map respectively.

And the error term of residual map will not explode after a sequence of convolution operations:

>   Given a convolution operation <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/81324f07e9ffb7920321df72cc0bee1b.svg?invert_in_darkmode" align=middle width=11.87217899999999pt height=22.648391699999998pt/> with unit normality and an error information <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/7a98a54c119420db4ee6aa0469136a1c.svg?invert_in_darkmode" align=middle width=91.42342604999999pt height=26.76175259999998pt/>, the error information <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/f7d66faa7becf05fe34ca40a39f45edd.svg?invert_in_darkmode" align=middle width=11.718034349999991pt height=24.7161288pt/> after convolution operation enjoys convolution-invariance, *i.e.*, <img src="https://github.com/OrdinaryCrazy/cnn-compiler-notebook/master/svgs/719c94ef7e88447b1c4addeb8d874305.svg?invert_in_darkmode" align=middle width=138.80563949999998pt height=26.76175259999998pt/>.

Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/weekly-report/Rivulet_Proof.pdf)

### Motion Vector Feature Flow Version3

#### Idea

Rather than just scale the motion vector by 1x1 Convolutional Layer, we try to build a more complicated MV_Net try to improve the quality of motion vector used at feature map level. MV_Net is piced as following:

![](./figures/mvnet.png)

#### Result and Discussion

We get MAP@5 = 0.6225 with above MV_Net architecture, some points we discussed in the design:

*   MVFF-Object-Detection task is sensitive to the information loss in integer-times scale and width-height-same-ratio scale of movtion vector in pooling process, so we need firstly use interpolation (non-integer-times) scale to scale the movtion vector to a integer-times of
    feature map shape (16\*feat-map-width, 16*feat-map-height)

    Details can be found at [Github link](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/weekly-report/weeklyreport0819-0825.pdf)

Code find at [MVFF-Version3](https://github.com/OrdinaryCrazy/mvff-sideversions) 

File organization same as [Deep Feature Flow for Video Recognition](https://github.com/msracver/Deep-Feature-Flow)

### Motion Vector Feature Flow Version4

#### Idea

Use DMC-Net like structure to fine-tune the motion vector, try to gain more motion information form residual data under optical flow guidence.

![1566861769179](figures/1566861769179.png)
![1566862456389](figures/1566862456389.png)

DMC-Net details can be found at [Github Link]([https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/Studying%20Note/DMC-Net.md](https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/Studying Note/DMC-Net.md))

#### Result and Discussion

Result of MVFF-Version4-without optical flow guidence: MAP@5 = 0.5091.

Maybe we need extract optical flow from the dataset first.

Code find at [MVFF-Version4](https://github.com/OrdinaryCrazy/mvff-sideversions/tree/master/Version4) 

### Motion Vector Output Flow Step-Performance Curve

We tried different steps of MVOF to approximate the result of DFF and accelerate it, trying to analyse motion vector propagation at output level.

![mvof](figures/mvof.png)

# Quantum Computing Learning

## First stage: From bits to qubits: Basical Concepts and Algorithm of Quantum Computing

### What Is a QPU? 

*   QPU (Quantum Processing Unit) is a co-processor 

*   QPU has ability to dramatically extend the kinds of problems that are tractable within computing. 

*   The CPU issues the QPU co-processor commands only to initiate tasks suited to its capabilities.

### Native QPU Instructions 

*   Conventional high-level languages are commonly used to control lower-level QPU instructions. 

*   Essential QPU instruction set 

### Simulator Limitations 

*   One measure of the power of a QPU is the number of qubits it has available to operate on. 

*   Each qubit added to simulation will double the memory required to run the simulation, cutting its speed in half.

### QPU Versus GPU: 

Some Common Characteristics What it’s like to program a QPU:
*   It is very rare that a program will run entirely on a QPU. Usually, a program runnning on a CPU will issue QPU instructions, and later retrieve the results.
*   Some tasks are very suited to the QPU, and others are not.
*   The QPU runs on a separate clock from the CPU, and usually has its own dedicated hardware interfaces to external devices (such as optical outputs).
*   A typical QPU has its own special RAM, which the CPU cannot efficiently access.
*   A simple QPU will be one chip accessed by a laptop, or even perhaps eventually an area within another chip. A more advanced QPU is a large and expensive addon, and always requires special cooling.
*   When a computation is done, a projection of the result is returned to the CPU, and most of the QPU’s internal working data is discarded.
*   QPU debugging can be very tricky, requiring special tools and techniques, Stepping through a program can be difficult, and often the best approcah is to make changes to the program and observe their effect on the output.
*   Optimizations that speed up one QPU may slow down another.

## ~~Second stage: Great idea evolution and Important Works~~

## ~~Third stage: On-going Front Problem and Research~~

## ~~Fourth stage: Research directions~~

