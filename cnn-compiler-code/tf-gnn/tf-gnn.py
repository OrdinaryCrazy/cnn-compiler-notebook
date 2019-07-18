
import  os.path as osp

import  torch
import  torch.nn.functional         as F
from    torch_geometric.datasets    import Planetoid
from    torch_geometric.datasets    import PPI
# The citation network datasets "Cora", "CiteSeer" and "PubMed"
import  torch_geometric.transforms  as T
from    torch_geometric.nn          import GCNConv, ChebConv  # noqa

import time

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)

dataset = Planetoid(path, dataset, T.NormalizeFeatures())
# Rotates all points so that the eigenvectors overlie the axes of the Cartesian coordinate system. 
# If the data additionally holds normals saved in `data.norm these` will be also rotated.

data = dataset[0]

##############################################################################
# HAG HERE
##############################################################################
def maxRedundancy(x, edge_index):
    max_i = 0
    max_j = 0
    max_redundancy = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            redun = redundancyCompute(i, j, x, edge_index)
            print(str(i) + ", " + str(j) + ", " + str(redun))
            if redun > max_redundancy :
                max_redundancy = redun
                print(redun)
                max_i = i
                max_j = j
    return max_i, max_j

def redundancyCompute(v_i, v_j, x, edge_index):
    redun = 0
    for k in range(x.shape[0]):
        if (((edge_index.t()[:,0] == v_i)&(edge_index.t()[:,1] == k))&((edge_index.t()[:,0] == v_j)&(edge_index.t()[:,1] == k))).nonzero().shape[0] > 0 :
            redun += 1
    return redun

redundancy_threshold = 1.0
V = data.x.shape[0]
V_A_MAX = int(data.x.shape[0] / 4)
V_A = 0
# print(((data.edge_index.t()[:,0] == 2582)&(data.edge_index.t()[:,1] == 0)).nonzero().item())
# print(((data.edge_index.t()[:,0] == 0)&(data.edge_index.t()[:,1] == 2582)).nonzero().item())
# print(data.is_directed())
# input()

# while V_A < V_A_MAX :
#     v_i, v_j = maxRedundancy(data.x, data.edge_index)
#     if redundancyCompute(v_i, v_j, data.x, data.edge_index) > redundancy_threshold :

#         newPoint = data.x[v_i] + data.x[v_j]
#         newPointIndex = data.x.shape[0]

#         data.x = torch.cat([data.x, newPoint],0)
#         data.edge_index = torch.cat([data.edge_index.t(), torch.tensor([    [ v_i, newPointIndex ], 
#                                                                             [ newPointIndex, v_i ],
#                                                                             [ v_j, newPointIndex ],
#                                                                             [ newPointIndex, v_j ]
#                                                                         ],dtype=torch.long)],0).t().contiguous()
#         for i in range(V):
#             v_i_con = ((data.edge_index.t()[:,0] == v_i)&(data.edge_index.t()[:,1] == i)).nonzero().shape[0] == 1
#             v_j_con = ((data.edge_index.t()[:,0] == v_j)&(data.edge_index.t()[:,1] == i)).nonzero().shape[0] == 1
#             if v_i_con and v_j_con :
#                 index = ((data.edge_index.t()[:,0] == v_i)&(data.edge_index.t()[:,1] == i)).nonzero().item()
#                 data.edge_index = torch.cat([data.edge_index.t()[:index],data.edge_index.t()[index+1:]],0).t().contiguous()
#                 index = ((data.edge_index.t()[:,0] == v_j)&(data.edge_index.t()[:,1] == i)).nonzero().item()
#                 data.edge_index = torch.cat([data.edge_index.t()[:index],data.edge_index.t()[index+1:]],0).t().contiguous()
#                 index = ((data.edge_index.t()[:,0] == i)&(data.edge_index.t()[:,1] == v_i)).nonzero().item()
#                 data.edge_index = torch.cat([data.edge_index.t()[:index],data.edge_index.t()[index+1:]],0).t().contiguous()
#                 index = ((data.edge_index.t()[:,0] == i)&(data.edge_index.t()[:,1] == v_j)).nonzero().item()
#                 data.edge_index = torch.cat([data.edge_index.t()[:index],data.edge_index.t()[index+1:]],0).t().contiguous()

#                 data.edge_index = torch.cat([data.edge_index.t(), torch.tensor([ [ i, newPointIndex ] ],dtype=torch.long)],0).t().contiguous()
#     else:
#         break
#     V_A += 1
#     print("adding: %d"%(V_A))
##############################################################################

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        # During training, randomly zeroes some of the elements of the input tensor 
        # with probability p using samples from a Bernoulli distribution.
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    # Sets gradients of all model parameters to zero.
    # 将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）

    predict = model()[data.train_mask]
    label = data.y[data.train_mask]
    F.nll_loss(predict, label).backward()
    # The negative log likelihood loss.

    optimizer.step()
    # 更新所有参数


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
time_sum = 0.0
for epoch in range(1, 201):

    start_time = time.time()
    train()
    end_time = time.time()
    
    time_sum += (end_time - start_time)

    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

print(  "d = %d, V = %d, n = %d, average training time: %f ms"
        %(dataset.num_features, data.edge_index.shape[1], data.x.shape[0], (time_sum/200)*1e3)
        )