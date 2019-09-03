# Weekly Report 2019.08.19-2019.08.25

>   Jingtun ZHANG

**WHERE WE ARE**:

<img src="./figures/summer_intern.png" width="280px" height="750px" />

<img src="./figures/summer_intern2.png" width="500px" height="300px" />

## Work and Progress
1.    Paper-reading: DMC-Net [note][1](in writing)

2.    Coding and Debugging of MVFF-ObjectDetection-Version3

      **More Discussion:**

      1.   Simply process Movtion-Vector by a not so deep CNN (4~5 layers) and Pooling to feature-map size will not work, Object-Detection will only work on key frames, bounding box regression will not work well on non-key frames' feature map get by Conv-Pooling processed Movtion-Vector (Map = 0.08 *on small dataset*)
      2.   Simply change CNN structure or initialization method will not slove this problem (Map = 0.08)
      3.   Only one layer Pooling or using Conv to replace Pooling will not work either (Map = 0.08)
      4.   Interpolation+CNN without Pooling method will work --> Conv is in function (Map = 1.00)
      5.   Firstly interpolation Movtion-Vector to a integer-times of feature-map shape and then process it by Conv+Pooling will work (Map = 1.00)
      6.   Change Movtion-Vector loding size to make the width and height to have the same scale ratio, and then Conv-Pooling without interpolation method result will be improved by 2~3 times (Map = 0.15 ~ 0.25)

      **Conclusion:**

      ​	*MVFF-Object-Detection task is sensitive to the information loss in integer-times scale and width-height-same-ratio scale of movtion vector in pooling process, so we need firstly use interpolation (non-integer-times) scale to scale the movtion vector to a integer-times of feature map shape (16\*feat-map-width, 16\*feat-map-height)*


## This week plan

1.     paper reading for idea:
       1.      Quantum Computing
       3.      GNN models
2.     MVFF-Version4 coding

---
[1]: https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/GNN/DMC-Net.md
[2]: https://github.com/OrdinaryCrazy/cnn-compiler-notebook/blob/master/GNN/GCN.md