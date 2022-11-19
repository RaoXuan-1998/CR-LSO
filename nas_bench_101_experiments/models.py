import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from collect_101_dataset import PRIMITIVES_101

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
    def __init__(self, xdim, out_channels):
        super(NodeConv, self).__init__(aggr = 'add', flow = 'source_to_target')

        self.residual_mlp = nn.Sequential(
            nn.Linear(xdim, xdim),
            nn.Linear(xdim, out_channels)
            )
        
        self.kernel = nn.Sequential(
            nn.Linear(2*xdim, out_channels, bias = False),
            nn.LeakyReLU()
            )
           
    def forward(self, x, edge_index):
        residual_term = self.residual_mlp(x)
        nolinear_term = self.propagate(x = x, edge_index = edge_index)
        return residual_term + nolinear_term
    
    def message(self, x_j, x_i):
        tmp = torch.cat([x_i, x_j], dim = 1)
        return self.kernel(tmp)
    
class GNN_Predictor(nn.Module):
    def __init__(self, xdim = 6, hdim = 64, zdim = 64,
                 layer_num = 4):
        super(GNN_Predictor, self).__init__()
        
        # encoder
        self.convs = nn.ModuleList()
        self.convs.append(NodeConv(xdim, hdim))
        
        for layer in range(layer_num - 1):
            self.convs.append(NodeConv(hdim, hdim))
            
        self.fc3 = nn.Linear(7*hdim, zdim)
        
        self.mlp = MLP(zdim, hdim, hidden_layer = 3)
        
    def encode(self, arch):
        batch = len(arch.adj)
        x = arch.x
        edge_index = arch.edge_index

        for _, conv in enumerate(self.convs):
            x = conv(x = x, edge_index = edge_index)
        
        hidden_graph = x.view(batch, -1)
        mu = self.fc3(hidden_graph)
        return mu
    
    def forward(self, arch):
        z = self.encode(arch)
        return self.mlp(z)    
    
class ArchGVAE(nn.Module):
    def __init__(self, xdim = 6, hdim = 128, zdim = 56,
                 node_num = 7, node_type = 6, layers = 3):
        super(ArchGVAE, self).__init__()
        # encoder
        self.node_num = node_num
        self.node_type = node_type
        self.convs = nn.ModuleList()
        self.convs.append(NodeConv(xdim, hdim))
        for layer in range(layers - 1):
            self.convs.append(NodeConv(hdim, hdim))

        self.fc3 = nn.Linear(hdim, zdim)
        self.fc4 = nn.Linear(hdim, zdim)
        
        # decoder
        adj_dim = node_num*node_num
        ops_dim = node_num*node_type
        
        self.fc5 = nn.Linear(zdim, hdim)
        
        self.decoder = nn.Sequential(
            Resid_MLP(hdim, hdim),
            Resid_MLP(hdim, hdim))
        
        self.fc6 = Resid_MLP(hdim, adj_dim)
        self.fc7 = Resid_MLP(hdim, ops_dim)
        
        self.icnn = ICNN(zdim, 256, hidden_layer = 4)
        
        self.tanh = nn.Tanh()
        
        self.sigmoid = nn.Sigmoid()
        
    def encode(self, graph):
        batch = len(graph.adj)
        x = graph.x
        edge_index = graph.edge_index
       
        for _, conv in enumerate(self.convs):
            x = conv(x = x, edge_index = edge_index)
            
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
        Hg = self.decoder(Hg)
        
        pred_adj = self.fc6(Hg)
        pred_ops = self.fc7(Hg)
        
        pred_adj = pred_adj.reshape(pred_adj.shape[0], self.node_num, self.node_num)
        pred_ops = pred_ops.reshape(pred_ops.shape[0], self.node_num, self.node_type)

        return self.sigmoid(pred_adj), pred_ops
     
    def loss(self, mu, logvar, adj, ops, beta = 0.1):
        # G_true: [batch_size * edge_num * operator_num]      
        z = self.reparameterize(mu, logvar)
        
        BCE = nn.BCELoss(reduction = 'mean')
        
        pred_adj, pred_ops = self.decode(z)
        
        res = BCE(pred_adj, adj) + \
                  F.cross_entropy(pred_ops.transpose(1, 2), ops, reduction = 'mean')
                  
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return res + beta*kld, res, kld, z
    
    def cross_entropy(t, p, eps = 1e-6):
        return -torch.sum((t*torch.log(p + eps) + (1 - t)*torch.log(p - eps)))/(p.shape[0]*p.shape[1]*p.shape[2])
    
    def get_tensor(self, z):
        
        adj, ops = self.decode(z)
      
        return adj, ops
    
    def conver_tensor2arch(self, adj, ops):
        adj = adj.squeeze().detach()
        ops = ops.squeeze().detach()
        
        adj[adj > 0.5] = 1.0         

        _, pred_ops = torch.max(ops, dim = 1) 
        
        ops_str = [PRIMITIVES_101[index] for index in pred_ops]
        
        try:
            ops_str.remove('none')
        except:
            pass
        
        save_node_num = len(ops_str)
        adj = adj[0:save_node_num, 0:save_node_num].cpu().int().numpy()
        
        for i in range(adj.shape[0]):
            for j in range(adj.shape[0]):
                if i > j:
                    adj[i][j] = 0
                    
        return adj, ops_str
    