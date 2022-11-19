import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from collect_201_dataset import PRIMITIVES_201


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim = 128, output_dim = 1, hidden_layer = 2):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim),                                        
                                         nn.LeakyReLU()))    
        for l in range(hidden_layer):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias = False),
                nn.LeakyReLU()))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        h = x
        for l in range(len(self.layers)):
            h = self.layers[l](h)
        return h

class Resid_MLP(nn.Module):
    def __init__(self, idim, odim):
        super(Resid_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LeakyReLU())
        self.resid = nn.Linear(idim, odim)
    
    def forward(self, x):
        return self.mlp(x) + self.resid(x)
    
class ICNN(nn.Module):
    def __init__(self, input_dim, hidden_dim = 128, output_dim = 1, hidden_layer = 3):
        super(ICNN, self).__init__()
        self.affine_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                          nn.LeakyReLU())
        self.convx_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.hidden_layer = hidden_layer
        for l in range(hidden_layer):
            self.convx_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias = False),
                nn.LeakyReLU()))
            self.skip_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias = False),
                nn.LeakyReLU()))
        self.convx_layers.append(nn.Linear(hidden_dim, output_dim, bias = False))
        self.skip_layers.append(nn.Linear(input_dim, output_dim, bias = False))
        
        self.keys = list(self.convx_layers.state_dict().keys())
        
    def forward(self, x):
        h = self.affine_layer(x)
        for l in range(self.hidden_layer):
            h = self.convx_layers[l](h) + self.skip_layers[l](x)
        h = self.convx_layers[-1](h) + self.skip_layers[-1](x)
        return h

    def constraint_weights(self):        
        for key in self.keys:
           self.convx_layers.state_dict()[key].data.clamp_(min = 0.0, max = None)

class NodeConv(MessagePassing):
    def __init__(self, xdim, edim, out_channels):
        super(NodeConv, self).__init__(aggr = 'add', flow = 'source_to_target')

        self.residual_mlp = nn.Sequential(
            nn.Linear(xdim, xdim),
            nn.Linear(xdim, out_channels)
            )
        
        self.kernel = nn.Sequential(
            nn.Linear(2*xdim + edim, out_channels, bias = False),
            nn.LeakyReLU()
            )
           
    def forward(self, x, edge_index, edge_attr):
        residual_term = self.residual_mlp(x)
        nolinear_term = self.propagate(x = x, edge_index = edge_index, edge_attr = edge_attr)
        return residual_term + nolinear_term
    
    def message(self, x_j, x_i, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim = 1)
        return self.kernel(tmp)

class GNN_Predictor(nn.Module):
    def __init__(self, xdim = 4, edim = 5, hdim = 64, zdim = 64,
                 layer_num = 4):
        super(GNN_Predictor, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(NodeConv(xdim, edim, hdim))
        
        for layer in range(layer_num - 1):
            self.convs.append(NodeConv(hdim, edim, hdim))
            
        self.fc3 = nn.Linear(4*hdim, zdim)
        
        self.mlp = MLP(zdim, hdim, hidden_layer = 3)
        
    def encode(self, arch):
        batch = len(arch.tensor)
        x = arch.x
        edge_index = arch.edge_index
        edge_attr = arch.edge_attr
        for _, conv in enumerate(self.convs):
            x = conv(x = x, edge_index = edge_index, edge_attr = edge_attr)
        
        hidden_graph = x.view(batch, -1)
        mu = self.fc3(hidden_graph)
        
        return mu
    
    
    def forward(self, arch):
        z = self.encode(arch)
        return self.mlp(z)
    

class ArchGVAE(nn.Module):
    def __init__(self, xdim = 4, edim = 5, hdim = 128, zdim = 32,
                 node_num = 4, edge_type = 5, edge_num = 6, layers = 3):
        super(ArchGVAE, self).__init__()
        # encoder
        self.node_num = node_num
        self.edge_num = edge_num
        self.edge_type = edge_type
        self.convs = nn.ModuleList()
        self.convs.append(NodeConv(xdim, edim, hdim))
        for layer in range(layers - 1):
            self.convs.append(NodeConv(hdim, edim, hdim))

        self.fc3 = nn.Linear(hdim, zdim)
        self.fc4 = nn.Linear(hdim, zdim)
        
        # decoder
        odim = (2*node_num + edge_type)*edge_num
        self.fc5 = nn.Linear(zdim, hdim)
        
        self.decoder = nn.Sequential(
            Resid_MLP(hdim, hdim),
            Resid_MLP(hdim, odim))
        
        self.icnn = ICNN(zdim, 128, hidden_layer = 4)
        
        self.tanh = nn.Tanh()
        
    def encode(self, graph):
        batch = len(graph.tensor)
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        
        for _, conv in enumerate(self.convs):
            x = conv(x = x, edge_index = edge_index, edge_attr = edge_attr)
            
        hidden_graph = x.view(batch, self.node_num, -1)
        hidden_graph = torch.sum(hidden_graph, dim = 1)
        
        mu, logvar = self.fc3(hidden_graph), self.fc4(hidden_graph)
        
        return mu, logvar

    def reparameterize(self, mu, logvar, eps_scale = 0.01):
        if self.training:            
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu            
 
    def decode(self, z):
        Hg = self.tanh(self.fc5(z))
        pred_arch = self.decoder(Hg)
        pred_arch = pred_arch.reshape(pred_arch.size(0), self.edge_num, -1)

        return pred_arch[:,:,:self.node_num], pred_arch[:,:,self.node_num:2*self.node_num], pred_arch[:,:,2*self.node_num:]
     
    def loss(self, mu, logvar, arch_tensor, beta = 0.005):
        # G_true: [batch_size * edge_num * operator_num]      
        z = self.reparameterize(mu, logvar)
        _, true_input_node = torch.max(arch_tensor[:, :, :self.node_num], 2)
        _, true_output_node = torch.max(arch_tensor[:, :, self.node_num: 2*self.node_num], 2)
        _, true_edge_type = torch.max(arch_tensor[:, :, 2*self.node_num :], 2)        
        
        input_node_predict, output_node_predict, edge_type_predict = self.decode(z)
        
        res = F.cross_entropy(edge_type_predict.transpose(1, 2), true_edge_type, reduction = 'mean') + \
                F.cross_entropy(input_node_predict.transpose(1, 2), true_input_node, reduction = 'mean') + \
                F.cross_entropy(output_node_predict.transpose(1, 2), true_output_node, reduction = 'mean')     
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return res + beta*kld, res, kld, z  
    
    def get_tensor(self, z, one_hot = False):
        i, o, e = self.decode(z)
        i = torch.softmax(i, dim = 2)
        o = torch.softmax(o, dim = 2)
        e = torch.softmax(e, dim = 2)
        if not one_hot:
            g = torch.cat([i,o,e], dim = 2)
        else:
            i_indices = torch.argmax(i, dim = 2)
            o_indices = torch.argmax(o, dim = 2)
            e_indices = torch.argmax(e, dim = 2)
            i = torch.nn.functional.one_hot(i_indices, self.node_num)
            o = torch.nn.functional.one_hot(o_indices, self.node_num)
            e = torch.nn.functional.one_hot(e_indices, self.edge_type)
            g = torch.cat([i,o,e], dim = 2)  
        
        arch_tensor = g
        return arch_tensor
      
    def conver_tensor2arch(self, arch_tensor):
        # do not contain batch dim
        # cell_tensor: [edge_num, 2*node_num + edge_type]
        arch_tensor = arch_tensor.squeeze()
        edge_type = arch_tensor[:,2*self.node_num:]
        edge_type = torch.argmax(edge_type, dim = 1)
        arch_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(
                PRIMITIVES_201[edge_type[0]],
                PRIMITIVES_201[edge_type[1]],
                PRIMITIVES_201[edge_type[2]],
                PRIMITIVES_201[edge_type[3]],
                PRIMITIVES_201[edge_type[4]],
                PRIMITIVES_201[edge_type[5]]
                )
        
        return arch_str

