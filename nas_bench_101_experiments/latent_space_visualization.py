from models import ICNN, ArchGVAE, GNN_Predictor
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from collect_101_dataset import NAS_Bench_101_Dataset
import torch.nn as nn

import matplotlib.pyplot as plt
# In[]
dataset = torch.load('dataset/nas_bench_101.pth')

# In[]
loader = DataLoader(
         dataset,
         batch_size = 20000)

gvae = torch.load('gvae/gvae_semisupervised_noICNN_4236.pth')


x_list = []
y_list = []
with torch.no_grad():
    for _, arch in enumerate(loader):
        arch = arch.cuda()
        mu, logvar = gvae.encode(arch)
        acc = arch.test_acc
        x_list.append(mu)
        y_list.append(acc)

x_list = torch.cat(x_list, dim = 0)
y_list = torch.cat(y_list, dim = 0)

import numpy as np

# In[]
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from torch.utils.data import DataLoader, Dataset
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as pylab

def plot_surface(x1, x2, y, title, save = None, xmax = 3.0, zlim = True,
                 ):
    fig = plt.figure(figsize = (6,6), dpi = 90)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim = 131, elev = 35)
    p = ax.plot_surface(x1, x2, y, cmap='Spectral_r', alpha = 0.9)
   
    ax.grid(True)
    x_major_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(y_major_locator)
    
    z_major_locator = MultipleLocator(0.3)
    ax.zaxis.set_major_locator(z_major_locator)
      
    # fig.tight_layout()
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('Accuracy')
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-xmax, xmax)

    # cbar.set_label('Values')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')
    plt.title(title)
    if save is not None:
        plt.savefig(save, dpi = 400)
        
def plot_scatter(x1, x2, y, title, save = None, xmax = 3.0, zlim = True,
                 ):
    fig = plt.figure(figsize = (6,6), dpi = 90)
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()
    # ax.view_init(azim = -30, elev = 35)
    
    values, indices = y.sort()
    
    p = ax.scatter(x1[indices], x2[indices],c = list(range(len(y))), cmap='Spectral', alpha = 0.8, s = 1.0)
   
    ax.grid(True)
    x_major_locator = MultipleLocator(0.3)
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(0.3)
    ax.yaxis.set_major_locator(y_major_locator)
    
    # z_major_locator = MultipleLocator(0.3)
    # ax.zaxis.set_major_locator(z_major_locator)
      
    # fig.tight_layout()
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    # ax.set_zlabel('Accuracy')
    ax.set_xlim(-0.30, 0.35)
    ax.set_ylim(-0.30, 0.35)

    # cbar.set_label('Values')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    # ax.zaxis.pane.set_edgecolor('k')
    plt.title(title)
    if save is not None:
        plt.savefig(save, dpi = 400)
# In[PCA]

from sklearn.decomposition import KernelPCA
kpca = KernelPCA(2, kernel = 'linear')
indices = torch.randint(0,len(x_list),[40000])
all_x = kpca.fit_transform(x_list[indices].cpu().numpy())

y = y_list[indices].cpu()

plot_scatter(all_x[:,0], all_x[:,1], y, title= 'Convexity regularized space', xmax = 0.4)

# In[]z
x_list = torch.tensor(all_x).cuda()

class Train_Dataset(Dataset):
    def __init__(self, x_list, y_list):
        super().__init__()
        self.x_list = x_list
        self.y_list = y_list
            
    def __getitem__(self, idx):
        return self.x_list[idx], self.y_list[idx]
            
    def __len__(self):
        return len(self.x_list)

train_dataset = Train_Dataset(x_list, y_list)
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)

class MLP(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 256, output_dim = 1, hidden_layer = 4):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim),                                        
                                         nn.LeakyReLU()))    
        for l in range(hidden_layer):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias = False),
                nn.ReLU()))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        h = x
        for l in range(len(self.layers)):
            h = self.layers[l](h)
        return h

net = MLP(input_dim = 2).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4, betas = (0.0,0.5), weight_decay = 0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = float(50), eta_min = 0.0)
mse = nn.MSELoss(reduction = 'mean')
import tqdm 
for epoch in tqdm.tqdm(range(50)):
    for step, (x, y) in enumerate(train_loader):
        p = net(x[:]).squeeze()
        loss = mse(0.01*y,p)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(loss)
    scheduler.step()

net = net.cuda()

scope = torch.arange(-1.6, 1.6, 0.01).cuda()
x1, x2 = torch.meshgrid([scope, scope])
inputs = torch.cat([x1.reshape(-1).unsqueeze(1), x2.reshape(-1).unsqueeze(1)], dim = 1)
with torch.no_grad():
    pred = net(inputs)
pred = pred.reshape(x1.shape)

plot_surface(x1.cpu(), x2.cpu(), pred.cpu(), title = 'convexity regularized space', xmax = 1.6)

# In[]
from sklearn.manifold import TSNE
all_x = TSNE(n_components = 2).fit_transform(x_list.cpu().numpy())
# In[]
x_list = torch.tensor(all_x).cuda()

class Train_Dataset(Dataset):
    def __init__(self, x_list, y_list):
        super().__init__()
        self.x_list = x_list
        self.y_list = y_list
            
    def __getitem__(self, idx):
        return self.x_list[idx], self.y_list[idx]
            
    def __len__(self):
        return len(self.x_list)

train_dataset = Train_Dataset(x_list, y_list)
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)

class MLP(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 1024, output_dim = 1, hidden_layer = 3):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim),                                        
                                         nn.LeakyReLU()))    
        for l in range(hidden_layer):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias = False),
                nn.ReLU()))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        h = x
        for l in range(len(self.layers)):
            h = self.layers[l](h)
        return h

net = MLP(input_dim = 2).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4, betas = (0.0,0.5), weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = float(50), eta_min = 0.0)
mse = nn.MSELoss(reduction = 'mean')
import tqdm 
for epoch in tqdm.tqdm(range(50)):
    for step, (x, y) in enumerate(train_loader):
        p = net(x[:]).squeeze()
        loss = mse(0.01*y,p)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(loss)
    scheduler.step()

net = net.cuda()

scope = torch.arange(-5, 5, 0.05).cuda()
x1, x2 = torch.meshgrid([scope, scope])
inputs = torch.cat([x1.reshape(-1).unsqueeze(1), x2.reshape(-1).unsqueeze(1)], dim = 1)
with torch.no_grad():
    pred = net(inputs)
pred = pred.reshape(x1.shape)

plot_surface(x1.cpu(), x2.cpu(), pred.cpu(), title = 'convexity regularizedc space')

# In[]
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from torch.utils.data import DataLoader, Dataset
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as pylab

def plot_surface(x1, x2, y, title, save = None, xmax = 5.0, zlim = True,
                 ):
    fig = plt.figure(figsize = (4,4), dpi = 500)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim = 131, elev = 35)
    p = ax.plot_surface(x1, x2, y, cmap='Spectral', alpha = 1.0)
   
    ax.grid(True)
    x_major_locator = MultipleLocator(2.0)
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(2.0)
    ax.yaxis.set_major_locator(y_major_locator)
    
    z_major_locator = MultipleLocator(10.0)
    ax.zaxis.set_major_locator(z_major_locator)
      
    # fig.tight_layout()
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('Performance')
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-xmax, xmax)

    # cbar.set_label('Values')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')
    plt.title(title)
    if save is not None:
        plt.savefig(save, dpi = 400)

# In[]
from sklearn.manifold import TSNE

all_x = TSNE(n_components = 2).fit_transform(x_list.cpu().numpy())

# In[]
values, indices = y_list.sort()
indices = indices.cpu().numpy()
y = y_list.cpu().numpy()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(all_x[:,0][indices], all_x[:,1][indices], y[indices], c = list(range(len(indices))), s = 3.0, marker="o", 
           cmap='viridis_r')


# In[]
x1 = x_list[:,0].cpu().numpy()
x2 = x_list[:,1].cpu().numpy()

# ax.view_init(azim = - 120, elev = 20)

values, indices = y_list.sort()

y = y_list.cpu().numpy()
indices = indices.cpu().numpy()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x1[indices], x2[indices], y[indices], c = y[indices], s = 3.0, marker="o", 
           cmap='viridis_r')


# p = ax.plot_surface(x1, x2, y_list, cmap='Spectral_r', alpha = 0.95)