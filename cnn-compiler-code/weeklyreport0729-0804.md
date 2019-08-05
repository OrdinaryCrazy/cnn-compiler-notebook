Weekly Report 2019.07.22-2019.07.28

>   Jingtun ZHANG

**WHERE WE ARE**:

<img src="./figures/summer_intern.png" width="600px" height="1200px" />

## Work and Progress
1.   Reimplementation of [HAG][4]:
     
     1.   release [HAG model Version 0.0][1]
     
         system configuration:
     
         *   operating system: Ubuntu 19.04
         *   python:
         *   packets:
     
         model:
     
         `class HAG(x, edge_index, ha_proportion=0.25, redundancy_threshold=1, aggr='add', flow='source_to_target')`:
     
         attributes:
     
         *   ``
         *   ``
     
         methods:
     
         *   ``
     
     2.   detailed description about max redundancy computation:
     
         *   object:
     
         
     
         *   graph adjacent matrix method (implemented by scipy.sparse API, have not profiling now)
             *   common targets number = number of two steps


## This week plan

1.     paper reading for idea:
2.     optimization of HAG Code

---
[4]: https://arxiv.org/pdf/1906.03707.pdf