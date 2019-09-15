import torch_geometric as pyg
import numpy as np
import torch
import heapq
import math
from scipy.sparse import coo_matrix
from torch_geometric.utils import scatter_

class HAG(object):
    '''
        Hierarchically Aggregated Computation Graphs
        Version: 0.0
        Programmer: Jingtun ZHANG
        Last modified: Jingtun ZHANG 20190804
    '''
    def __init__(self, x, edge_index, ha_proportion=0.25, redundancy_threshold=1, 
                 aggr='add', flow='source_to_target'):
        '''
            input:
                x:  [torch.tensor shape (N, d)]
                    origin nodes 
                edge_index: [torch.tensor COO format shape ( 2 * E, 2)]
                    origin edges 
                ha_proportion: [scalar 0~1]
                    proportion of aggregated nodes number to origin nodes number 
                redundancy_threshold: [scalar]
                    minimum redundancy threshold aggregated node generation 
                aggr: [str 'add' or 'mean' or 'max']
                    aggregation scheme to use
                flow: [str 'source_to_target' or 'target_to_source']
                    flow direction of message passing
            output:
                None
        '''
        self.h = x
        self.x = x
        self.V = x.shape[0]
        # coo format edge index
        ############################################################
        # ATTENTION: HERE IS COMPUTATION GRAPH (DIRECTED GRAPH)    #
        #            NOT TOPOLOGY GRAPH                            #
        # [ [ target_i_0, target_i_1, ... , target_i_2E ]          #
        #   [ source_i_0, source_i_1, ... , source_i_2E ] ]        #
        ############################################################
        self.edge_index = edge_index
        self.capacity = np.ceil(x.shape[0] * ha_proportion)
        # aggregated nodes [torch.tensor shape (N_A, d)]
        # self.ha = torch.FloatTensor(0, x.shape[1])
        self.redundancy_threshold = redundancy_threshold
        # in Version 0.0 only consider 'add'
        self.aggr = aggr
        # in Version 0.0 only consider 'source_to_targets'
        self.flow = flow
    def graph_to_hag(self):
        '''
            build HAG at preprocessing stage
            input:
                None
            output:
                None
        '''
        while self.h.shape[0] - self.V < self.capacity :
            v_i, v_j, max_r = self.max_redundancy()
            log = 'redundancy:{:4d}, {:4d} / {:4d} '
            print(log.format(int(max_r), int(self.h.shape[0]-self.V),int(self.capacity)),end = "\r")
            if max_r > self.redundancy_threshold :
                newPoint = self.h[v_i] + self.h[v_j]
                newPointIndex = self.h.shape[0]
                # insert new point
                newPoint = newPoint.view(1,self.h.shape[1])         
                self.h = torch.cat([self.h, newPoint], 0)
                self.edge_index = torch.cat([
                    self.edge_index.t(),
                    torch.tensor([
                      # [v_i, newPointIndex],
                        [newPointIndex, v_i],
                      # [v_j, newPointIndex],
                        [newPointIndex, v_j]
                    ], dtype=torch.long)
                ]).t().contiguous()
                
                for i in range(self.h.shape[0] - 1) :
                    # common neighbor judge
                    v_i_con = ( (self.edge_index.t()[:,0] == i  )
                               &(self.edge_index.t()[:,1] == v_i) ).nonzero().shape[0] == 1
                    v_j_con = ( (self.edge_index.t()[:,0] == i)
                               &(self.edge_index.t()[:,1] == v_j) ).nonzero().shape[0] == 1
                    if v_i_con and v_j_con :
                        #-------------------------------------------------------------
                        # index = ( (self.edge_index.t()[:,0] == v_i)
                        #          &(self.edge_index.t()[:,1] == i) ).nonzero().item()
                        # self.edge_index = torch.cat([
                        #     self.edge_index.t()[:index],
                        #     self.edge_index.t()[index+1:]
                        # ]).t().contiguous()
                        #-------------------------------------------------------------
                        # index = ( (self.edge_index.t()[:,0] == v_j)
                        #          &(self.edge_index.t()[:,1] == i) ).nonzero().item()
                        # self.edge_index = torch.cat([
                        #     self.edge_index.t()[:index],
                        #     self.edge_index.t()[index+1:]
                        # ]).t().contiguous()
                        #-------------------------------------------------------------
                        index = ( (self.edge_index.t()[:,0] == i  )
                                 &(self.edge_index.t()[:,1] == v_i)).nonzero().item()
                        self.edge_index = torch.cat([
                            self.edge_index.t()[:index],
                            self.edge_index.t()[index+1:]
                        ]).t().contiguous()
                        #-------------------------------------------------------------
                        index = ( (self.edge_index.t()[:,0] == i  )
                                 &(self.edge_index.t()[:,1] == v_j)).nonzero().item()
                        self.edge_index = torch.cat([
                            self.edge_index.t()[:index],
                            self.edge_index.t()[index+1:]
                        ]).t().contiguous()
                        #-------------------------------------------------------------
                        self.edge_index = torch.cat([
                            self.edge_index.t(), 
                        #     torch.tensor([ [ i, newPointIndex ], [ newPointIndex, i ] ],
                        #                  dtype=torch.long)]).t().contiguous()
                            torch.tensor([[i,newPointIndex]],dtype=torch.long)]).t().contiguous()
                        #-------------------------------------------------------------
            else:
                break
        print("hag building finished")
        return

    def hag_aggregate(self):
        '''
            compute embedding of aggregate node every iteration 
            input:
                None
            output:
                None
        '''
        iter_times = int(np.ceil(math.log2(self.capacity)))
        out = self.h.clone()
        for i in range(iter_times):
            out = scatter_(self.aggr, out, self.edge_index[0])
            out = torch.cat(self.h[:V], out[V:])
        self.h = out
        self.x = self.h[:V]

#     def hag_aggregate_grad(self):
#         '''
#             compute the gradients of hag_aggregate for back propagation
#             not implement in Version 0.0
#         '''

    def max_redundancy(self):
        '''
            find max redundancy node pair and return it
            input:
                None
            output: 
                v_i [scalar], v_j [scalar], max_r [scalar]
        '''
        all_nodes_num = self.h.shape[0]
        coo_edge_index = coo_matrix(( np.ones(self.edge_index.shape[1]), 
                                     (self.edge_index[0], self.edge_index[1])), 
                                     shape=(all_nodes_num, all_nodes_num))
        csr_edge_index = coo_edge_index.tocsr()
        common_neighbor_num = csr_edge_index.transpose().dot(csr_edge_index)
        common_neighbor_num.setdiag(0)
        max_num_flat_index = common_neighbor_num.argmax()
        v_i, v_j = (int(max_num_flat_index/all_nodes_num), int(max_num_flat_index%all_nodes_num))
        max_r = common_neighbor_num.max()
        
        return v_i, v_j, max_r
