import torch
from torch.utils.data import Dataset
from nas_101_database import NASBenchDataBase
from copy import deepcopy
import tqdm
from torch_geometric.data import Data


PRIMITIVES_101 = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'output', 'none']

def conver_cell2graph(adj, ops):
    
    node_attr = torch.nn.functional.one_hot(ops, len(PRIMITIVES_101)).float()
    
    source_nodes = []
    target_nodes = []
    for index_0 in range(adj.shape[0]):
        for index_1 in range(adj.shape[1]):
            if adj[index_0, index_1] == 1:
                source_nodes.append(index_0)
                target_nodes.append(index_1)
    
    source_nodes = torch.LongTensor(source_nodes).unsqueeze(0)
    target_nodes = torch.LongTensor(target_nodes).unsqueeze(0)

    edge_index = torch.cat([source_nodes, target_nodes], dim = 0)
    
    return node_attr, edge_index

class NAS_Bench_101_Dataset(Dataset):
    def __init__(self,):
        
        database = NASBenchDataBase('data/nasbench_only108_with_vertex_flops_and_params.json')
        
        self.arch_list = []
        self.valid_acc = []
        self.test_acc = []
        self.hash_list = []
        
        print('loading...')
        
        for index in tqdm.tqdm(range(database.size)):
            
            arch_info = database.query_by_index(index)
            arch_info = deepcopy(arch_info)
            
            arch_hash = arch_info['unique_hash']
            
            self.hash_list.append(arch_hash)
            
            adj = arch_info['module_adjacency']
            ops = arch_info['module_operations']
            valid_acc = arch_info['avg_validation_accuracy']
            test_acc = arch_info['avg_test_accuracy']
            
            adj = torch.tensor(adj)
            adj = adj.t() + adj
            
            if adj.shape[0] != 7:
                pad_num = 7 - adj.shape[0]
                pad = torch.nn.ZeroPad2d(padding=(0, pad_num, 0, pad_num))
                adj = pad(adj)
                for i in range(pad_num):
                    ops.append('none')  
             
            ops = [PRIMITIVES_101.index(primitive) for primitive in ops]
            
            ops = torch.tensor(ops)
            
            node_attr, edge_index = conver_cell2graph(adj, ops)

            arch_adj = adj.float().unsqueeze(0)
            
            arch_ops = ops.unsqueeze(0)
            
            arch_graph = Data(
                x = node_attr, edge_index = edge_index, adj = arch_adj,
                ops = arch_ops,
                valid_acc = torch.tensor([valid_acc]),
                test_acc = torch.tensor([test_acc]),
                arch_index = torch.tensor([index])
            )
            
            self.arch_list.append(arch_graph)
            self.valid_acc.append(torch.tensor([valid_acc]))
            self.test_acc.append(torch.tensor([test_acc]))
                                    
    def conver_arch2_adjandops(self, adj, ops):
        adj = adj.squeeze().detach().cpu().reshape(7,7).int().numpy().tolist()
        
        ops = ops.squeeze().detach().cpu()
        
        values, indices = torch.max(ops, dim = 1)
        ops = ['input']
        
        for index in indices:
            ops.append(PRIMITIVES_101[index])
            
        return adj, ops
            
            
    def __getitem__(self, idx):
        return self.arch_list[idx]
        
    def __len__(self):
        return len(self.arch_list)
            
# dataset = NAS_Bench_101_Dataset()
# torch.save(dataset, 'dataset/nas_bench_101.pth')
            
        