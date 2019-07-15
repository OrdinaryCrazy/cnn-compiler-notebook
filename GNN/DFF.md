# DFF: Deep Feature Flow for Video Recognition

*Question*: per-frame evaluation is too slow and unaffordable

*Solution*: run expensive convolutional sub-network only on sparse **key frames（关键帧）** and propagates their deep feature maps to other frames via a **flow field（流域）**.

*   Decompose neutral network $$\mathcal{N}$$ into **feature network** and **task network**
*   Run $$\mathcal{N_{feat}}$$ only on key-frame and propagate featuremap to following non-key frame

## Deep Feature Flow

*   feature propagation function (特征传播函数)

$$
f_i =\mathcal{W}(f_k, M_{i \to k}, S_{i \to k})
$$

*   where $$M_{i \to k}$$ is a two dimensional flow field
*   

