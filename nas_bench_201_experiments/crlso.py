from collect_201_dataset import NAS_Bench_201_Dataset, random_sample_a_genotype, conver_cell2graph
from models import ArchGVAE, GNN_Predictor
import torch.nn as nn
import logging
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import AvgrageMeter, create_exp_dir
import tqdm
from train_gvae_semi_supervised import train_gvae
from nas_201_api import NASBench201API as API
from torch_geometric.data import Data
from copy import deepcopy
from nas_201_database import NASBench201DataBase

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = log_format, datefmt = '%m/%d %I:%M:%S %p')


configs = {
    'nas_bench_201_dataset_path' : 'dataset/nas_201_dataset.pth',
    
    # which dataset to evaluate?
    'dataset' : 'ImageNet',
    # the maximum evaluation number
    'evaluate_num' : 500,

    # hyperparameters of fine-tunning the ICNN 
    'lr' : 1e-4,
    'betas' : (0.0, 0.5),
    'weight_decay' : 0.0,
    'epoch_num' : 50,
    'batch_size' : 32,
    'topk' : 10,
    
    # if do not use a pretrained, train a new one, and save it in 'gvae/gvae.pth'
    'pretrained_gvae' : True, 
    'zdim' : 8,
    
    #clso
    'step_num' : 1,
    'eta' : 0.2,
    'delta_eta' : 0.2,
    'random_num' : 2000, # before searching, how many architectures should be sampled from the latent space ?
    }

configs['gvae_path'] = 'gvae/gvae_{}.pth'.format(configs['dataset'])

class ICNN_Dataset(Dataset):
    def __init__(self, labeled_set):
        super().__init__()
        self.dataset = labeled_set
            
    def __getitem__(self, idx):
        return self.dataset[1][idx], self.dataset[2][idx]
            
    def __len__(self):
        return len(self.dataset[1])

class CRLSO:
    def __init__(self, configs = configs):
        # self.api = API(configs['nas_bench_201_path'])
        self.database = NASBench201DataBase('data/nasbench201_with_edge_flops_and_params.json')
        self.dataset = torch.load(configs['nas_bench_201_dataset_path'])
        self.configs = configs
        
        if configs['pretrained_gvae']:
            pass
        else:
            train_gvae()
        self.gvae = torch.load(configs['gvae_path']).cuda()
        
        self.labeled_set = self.gvae.labeled_set
        
    def main_loop(self, noise = True):
        
        while len(self.labeled_set[1]) < (self.configs['evaluate_num']):
            self.tune_icnn()
        
            values, indices = self.labeled_set[2].topk(self.configs['topk'])
            
            # topk_arch_strs = [self.labeled_set[0][indice] for indice in indices]
            
            topk_latents = [self.labeled_set[1][indice] for indice in indices]
            
            for _, latent in enumerate(topk_latents):
                
                eta = self.configs['eta']
                new_arch = False
                latent_copy = deepcopy(latent)
                while not new_arch:
                    
                    if noise:
                        # Add some noise to explore the search space
                        latent = deepcopy(latent_copy) + 0.5*torch.randn_like(latent_copy)
                    else:
                        latent = deepcopy(latent_copy)
                    
                    latent = nn.Parameter(latent.cuda(), requires_grad = True)
                    optimizer = torch.optim.SGD([latent], lr = eta)
            
                    acc_p = (- self.gvae.icnn(latent) + 1.0)
                    acc_p.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                    with torch.no_grad():
                        arch_tensor = self.gvae.get_tensor(latent.unsqueeze(0))
                        arch_str = self.gvae.conver_tensor2arch(arch_tensor)
                    
                    if arch_str in set(self.labeled_set[0]):
                        eta = eta + self.configs['delta_eta']
                        
                    else:
                        arch_index = self.dataset.str2index(arch_str)
                        
                        if self.configs['dataset'] == 'CIFAR10':
                            acc = self.dataset.cifar10_acc[arch_index][0]
                        elif self.configs['dataset'] == 'CIFAR100':
                            acc = self.dataset.cifar100_acc[arch_index][0]
                        elif self.configs['dataset'] == 'ImageNet':
                            acc = self.dataset.imagenet_acc[arch_index][0]
                    
                        acc = 0.01*acc
                    
                        logging.info(
                            'Obtain an new architecture with acc:%f', acc)
                    
                        self.labeled_set[0].append(arch_str)
                        
                        self.labeled_set[1] = torch.cat(
                            [self.labeled_set[1], deepcopy(latent.detach().cpu()).unsqueeze(0)])
                        
                        self.labeled_set[2] = torch.cat(
                            [self.labeled_set[2], torch.tensor([acc]).float()])
                        
                        new_arch = True
                        
            
    def tune_icnn(self, noise = True):
        mse = nn.MSELoss(reduction = 'mean')
        
        dataset = ICNN_Dataset(self.labeled_set)
        dataloader = DataLoader(
            dataset, batch_size = self.configs['batch_size'], shuffle = True
            )
        
        optimizer = torch.optim.Adam(
            self.gvae.icnn.parameters(),
            lr = self.configs['lr'], betas = self.configs['betas'], weight_decay = 1e-5
            )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = float(self.configs['epoch_num']), eta_min = 1e-5
            )
        
        for epoch in tqdm.tqdm(range(self.configs['epoch_num'])):
            objs = AvgrageMeter()
            mse = nn.MSELoss(reduction = 'mean')
            for step, (latents, acc) in enumerate(dataloader):
                n = len(latents)
                
                if not noise:
                    latents = latents.cuda()
                    acc = acc.cuda()
                else:
                    # add some noise to explore the search space
                    latents = latents.cuda() + 0.05*torch.randn_like(latents.cuda())
                    acc = acc.cuda() + 0.01*torch.randn_like(acc.cuda())
                
                pred_acc = (-self.gvae.icnn(latents) + 1.0).squeeze()
                
                loss = mse(acc, pred_acc)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                objs.update(loss.data.item(), n)
                
                self.gvae.icnn.constraint_weights()
            
            scheduler.step()
        
        logging.info('Finetune the icnn, loss_pred:%e', objs.avg)
        
        
    def obtain_topk_performance(self, topk = 1):
        values, indices = self.labeled_set[2].topk(self.configs['topk'])
        arch_str = self.labeled_set[0][indices[topk]]
        arch_info = self.database.query_by_str(arch_str)
        return arch_info
        
            
if __name__ == '__main__':
    lso = CRLSO()
    lso.main_loop()
    lso.obtain_topk_performance(3)


        
        
        
        
        
        
            