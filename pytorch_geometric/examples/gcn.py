import  os.path as osp

import  torch
import  torch.nn.functional         as F
from    torch_geometric.datasets    import Planetoid
from    torch_geometric.datasets    import PPI
from    torch_geometric.data        import DataLoader
import  torch_geometric.transforms  as T
from    torch_geometric.nn          import GCNConv, ChebConv  # noqa

# dataset = 'Cora'
# dataset = 'CiteSeer'
dataset = 'PubMed'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
# train_dataset = PPI(path, split='train')
# val_dataset = PPI(path, split='val')
# test_dataset = PPI(path, split='test')
# train_batch_size = 256
# train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

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
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model, data = Net().to(device), data.to(device)
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    # optimizer.zero_grad()
    # pred = model()
    # print("aggregate time: %.16f, mapping time: %.16f "%(   model.conv1.aggregateTime + model.conv2.aggregateTime, 
    #                                                         model.conv1.mappingTime + model.conv2.mappingTime))
    # F.nll_loss(pred[data.train_mask], data.y[data.train_mask]).backward()
    # optimizer.step()
    total_loss = 0
    for data in train_loader:
        num_graphs = data.num_graphs
        data.batch = None
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset), model.conv1.aggregateTime + model.conv2.aggregateTime, model.conv1.mappingTime + model.conv2.mappingTime


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

from pandas import DataFrame
import time
gcn_Cora_profile = DataFrame(columns=[  'epoch','node_num','edge_num','feature_dimension', 'train_batch_size',
                                        'train_time','agg_time','map_time'])

best_val_acc = test_acc = 0
for epoch in range(1, 201):

    start_time = time.time()
    loss, aggtime, maptime = train()
    print(loss)
    end_time = time.time()

    # print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
    #     epoch, loss, val_f1, test_f1))
    # train_acc, val_acc, tmp_test_acc = test()
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     test_acc = tmp_test_acc
    # log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(epoch, train_acc, best_val_acc, test_acc))
    result = DataFrame([[   epoch, data.x.shape[0], data.edge_index.shape[1], dataset.num_features,train_batch_size, end_time - start_time, aggtime, maptime]],
                columns=[  'epoch','node_num','edge_num','feature_dimension','train_batch_size','train_time','agg_time','map_time'])
    # print(result)
    gcn_Cora_profile = gcn_Cora_profile.append(result)
print(gcn_Cora_profile)
# gcn_Cora_profile.to_excel("./gcn_Cora_profile.xlsx")
# gcn_Cora_profile.to_excel("./gcn_CiteSeer_profile.xlsx")
# gcn_Cora_profile.to_excel("./gcn_PubMed_profile.xlsx")
gcn_Cora_profile.to_excel("./gcn_PPI_profile.xlsx")
