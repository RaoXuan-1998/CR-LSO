from models import ICNN, ArchGVAE, GNN_Predictor
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from collect_101_dataset import NAS_Bench_101_Dataset
import torch.nn as nn
import numpy as np
import tqdm
import matplotlib.pyplot as plt

# In[]
dataset = torch.load('dataset/nas_bench_101.pth')

# In[]
loader = DataLoader(
         dataset,
         batch_size = 40000)

gvae = torch.load('gvae/gvae_4236.pth')


x_list = []
y_list = []

with torch.no_grad():
    for _, arch in enumerate(loader):
        arch = arch.cuda()
        mu, logvar = gvae.encode(arch)
        acc = arch.test_acc
        x_list.append(mu)
        y_list.append(acc)

x_list = torch.cat(x_list, dim = 0).cpu()
y_list = torch.cat(y_list, dim = 0).cpu()


# In[]
num = 2000
values, indices = y_list.topk(k = num)

x = x_list[indices]
y = y_list[indices]

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

# In[]
# top-100 similarity
shape = x.shape[0]
simlarity_matrix = torch.zeros(shape, shape)

for index in tqdm.tqdm(range(shape)):
    simlarity_matrix[index] = cos(x, x[index])
    
simlarity_matrix = simlarity_matrix.numpy()

fig, axs = plt.subplots(1,3, figsize = (12,4))

ax = axs[0]

labels = ['1', '50', '100']
ax.set_xticks(np.linspace(0, shape -1, 3), labels = labels)
ax.set_yticks(np.linspace(0, shape -1, 3), labels = labels)

# plt.xlabel('Ranking', position = 'top')
# plt.ylabel('Ranking')

ax.xaxis.set_ticks_position('top')

im = ax.imshow(simlarity_matrix, cmap = 'viridis_r',
               interpolation = 'gaussian')

# fig.colorbar(im, ax = ax, label='Cosine similarity')

# In[]
# bot-100 

values_bot, indices_bot = (- y_list).topk(k = num)

x_bot = x_list[indices_bot]
y_bot = y_list[indices_bot]

simlarity_matrix = torch.zeros(shape, shape)

for index in tqdm.tqdm(range(shape)):
    simlarity_matrix[index] = cos(x, x_bot[index])

ax = axs[1]

ax.xaxis.set_ticks_position('top')

labels_bot = ['-1', '-50', '-100']
ax.set_xticks(np.linspace(0, shape -1, 3), labels = labels)
ax.set_yticks(np.linspace(0, shape -1, 3), labels = labels_bot)

im = ax.imshow(simlarity_matrix, cmap = 'viridis_r',
               interpolation = 'gaussian')

simlarity_matrix = torch.zeros(shape, shape)

for index in tqdm.tqdm(range(shape)):
    simlarity_matrix[index] = cos(x_bot, x_bot[index])
    

ax = axs[2]

ax.xaxis.set_ticks_position('top')

labels_bot = ['-1', '-50', '-100']
ax.set_xticks(np.linspace(0, shape -1, 3), labels = labels_bot)
ax.set_yticks(np.linspace(0, shape -1, 3), labels = labels_bot)

im = ax.imshow(simlarity_matrix, cmap = 'viridis_r',
               interpolation = 'gaussian')  


fig.colorbar(im, ax = axs, label='Cosine similarity',# location = 'bottom', 
              shrink = 0.6, fraction=0.1)
# In[]

simlarity_matrix = torch.zeros(shape, shape)

for i in range(shape):
    for j in range(shape):
        simlarity_matrix[i,j] = CosSimilarity(x_bot[i], x_bot[j])

ax = axs[2]

ax.xaxis.set_ticks_position('top')

labels_bot = ['-1', '-50', '-100']
ax.set_xticks(np.linspace(0, shape -1, 3), labels = labels_bot)
ax.set_yticks(np.linspace(0, shape -1, 3), labels = labels_bot)

im = ax.imshow(simlarity_matrix, cmap = 'viridis_r',
               interpolation = 'gaussian')

fig.colorbar(im, ax = axs, label='Cosine similarity',# location = 'bottom', 
             shrink = 0.6, fraction=0.1)
        
# In[]
simlarity_matrix = torch.zeros(shape, shape)

for i in range(shape):
    for j in range(shape):
        simlarity_matrix[i,j] = CosSimilarity(x_bot[i], x_bot[j])

simlarity_matrix = simlarity_matrix.numpy()

fig, ax = plt.subplots()

# ax.set_xticks(np.arange(20, num + 1, 20), labels = np.arange(-num, 0, 20))
# ax.set_yticks(np.arange(0, num + 1, 20), labels = np.arange(-num, 0, 20))

ax.xaxis.set_ticks_position('top')

im = ax.imshow(simlarity_matrix, cmap = 'viridis_r',
               interpolation = 'hermite')

fig.colorbar(im, ax = ax, label='Cosine similarity')

# In[]

    




