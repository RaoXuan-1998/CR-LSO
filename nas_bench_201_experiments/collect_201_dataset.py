from nas_201_database import NASBench201DataBase
import torch
import torch.nn as nn
from torch_geometric.data import Data
import random
from torch.utils.data import Dataset

PRIMITIVES_201 = ['avg_pool_3x3', 'nor_conv_1x1', 'skip_connect', 'nor_conv_3x3', 'none']

def arch2list(arch_str):
    node_strs = arch_str.split('+')
    genotypes = []
    for i, node_str in enumerate(node_strs):
      inputs = list(filter(lambda x: x != '', node_str.split('|')))
      for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
      inputs = ( xi.split('~') for xi in inputs )
      input_infos = tuple( (op, int(IDX)) for (op, IDX) in inputs)
      genotypes.append( input_infos )
    return genotypes

def conver_cell2tensor(arch_list):
    node_attr = nn.functional.one_hot(torch.tensor([0, 1, 2, 3]), 4)
    edge_attr = nn.functional.one_hot(torch.tensor([0, 1, 2, 3, 4]), 5)
    edge_list = []
    for node_index in range(3):
        node_geno = arch_list[node_index]
        for _, (primitive, input_node) in enumerate(node_geno):
            edge_list.append(
                (input_node, node_index + 1, primitive))   
            
    def edge2architecture(edge_list):
        edge_tensor_list = []
        for edge_index, edge in enumerate(edge_list):
            edge_tensor = torch.cat(
                [node_attr[edge[0]], node_attr[edge[1]],
                edge_attr[PRIMITIVES_201.index(edge[2])]]
                ).unsqueeze(0).float()
            
            edge_tensor_list.append(edge_tensor)        
        architecture = torch.cat(edge_tensor_list, dim = 0)
        return architecture
    architecture = edge2architecture(edge_list)
    return architecture  

# This is for training the encoder
def conver_cell2graph(arch_list):
    node_attr = nn.functional.one_hot(torch.tensor([0, 1, 2, 3]), 4).float()
    
    source_nodes = torch.LongTensor([0,0,1,0,1,2]).unsqueeze(0)
    target_nodes = torch.LongTensor([1,2,2,3,3,3]).unsqueeze(0)
    
    edge_index = torch.cat([
        torch.cat([source_nodes, target_nodes], dim = 1),
        torch.cat([target_nodes, source_nodes], dim = 1)], dim = 0)
    
    edge_primitives = []
    
    for node_index in range(3):
        node_geno = arch_list[node_index]
        for _, (primitive, input_node) in enumerate(node_geno):
            edge_primitives.append(PRIMITIVES_201.index(primitive))
        
    edge_primitives = torch.LongTensor(edge_primitives)
    cell_tensor = conver_cell2tensor(arch_list).unsqueeze(0)

    edge_attr = nn.functional.one_hot(edge_primitives, len(PRIMITIVES_201)).float()
    edge_attr = torch.cat([edge_attr, edge_attr], dim = 0)
    
    return edge_index, node_attr, edge_attr, cell_tensor

# raw dataset
class NAS_Bench_201_Dataset(Dataset):
    def __init__(self,):
        self.genotypes = []
        
        database = NASBench201DataBase('data/nasbench201_with_edge_flops_and_params.json')
        
        self.arch_str_list = []
        self.cifar10_acc = []
        self.cifar100_acc = []
        self.imagenet_acc = []
        
        self.arch_graph_list = []
        
        self.arch_rank = []
             
        for index in range(len(database.archs)):
            
            arch_info = database.query_by_id('{}'.format(index))
            
            arch_str = arch_info['arch_str']
            self.arch_str_list.append(arch_str)
            
            arch_list = arch2list(arch_str)
            
            self.arch_rank.append((arch_info['cifar10_rank'],
                                   arch_info['cifar100_rank'],
                                   arch_info['imagenet16_rank'])
                                  )
            
            valid_acc_cifar10 = arch_info['cifar10_val_acc']
            test_acc_cifar10 = arch_info['cifar10_test_acc']
            self.cifar10_acc.append((valid_acc_cifar10, test_acc_cifar10))
            
            valid_acc_cifar100 = arch_info['cifar100_val_acc']
            test_acc_cifar100 = arch_info['cifar100_test_acc']
            self.cifar100_acc.append((valid_acc_cifar100, test_acc_cifar100))
            
            valid_acc_imagenet = arch_info['imagenet16_val_acc']
            test_acc_imagenet = arch_info['imagenet16_test_acc']
            
            self.imagenet_acc.append((valid_acc_imagenet, test_acc_imagenet))
            
            edge_index, node_attr, edge_attr, cell_tensor = conver_cell2graph(arch_list)
            
            arch_graph = Data(
                x = node_attr, edge_index = edge_index, edge_attr = edge_attr, tensor = cell_tensor,
                valid_acc_cifar10 = torch.tensor([valid_acc_cifar10]).float(),
                test_acc_cifar10 = torch.tensor([test_acc_cifar10]).float(),
                valid_acc_cifar100 = torch.tensor([valid_acc_cifar100]).float(),
                test_acc_cifar100 = torch.tensor([test_acc_cifar100]).float(),
                valid_acc_imagenet = torch.tensor([valid_acc_imagenet]).float(),
                test_acc_imagenet = torch.tensor([test_acc_imagenet]).float(),
                arch_str = arch_str,
                )
            
            self.arch_graph_list.append(arch_graph)
            
    def __getitem__(self, idx):
        return self.arch_graph_list[idx]
    
    def __len__(self):
        return len(self.arch_graph_list)
    
    def str2index(self, arch_str):
        return self.arch_str_list.index(arch_str)
    
# dataset = NAS_Bench_201_Dataset()
# torch.save(dataset, 'dataset/nas_201_dataset.pth')
# dataset = torch.load('dataset/nas_201_dataset.pth')


def random_sample_a_genotype():
    arch_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(
                PRIMITIVES_201[random.randint(0,4)],
                PRIMITIVES_201[random.randint(0,4)],
                PRIMITIVES_201[random.randint(0,4)],
                PRIMITIVES_201[random.randint(0,4)],
                PRIMITIVES_201[random.randint(0,4)],
                PRIMITIVES_201[random.randint(0,4)],
            )
    return arch_str




