from collect_101_dataset import NAS_Bench_101_Dataset
from nas_101_database import NASBenchDataBase
from models import *
import torch.nn as nn
import logging
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import AvgrageMeter
import tqdm
from train_gvae_semi_supervised import get_gvae
from torch_geometric.data import Data
from copy import deepcopy

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = log_format, datefmt = '%m/%d %I:%M:%S %p')


configs = {
    'nas_bench_101_database_path' : 'data/nasbench_only108_with_vertex_flops_and_params.json',
    'nas_bench_101_dataset_path' : 'dataset/nas_101_dataset.pth',
    
    # the maximum evaluation number
    'evaluate_num' : 900,

    # hyperparameters of fine-tunning the ICNN 
    'lr' : 1e-4,
    'betas' : (0.0, 0.5),
    'weight_decay' : 0.0,
    'epoch_num' : 50,
    'batch_size' : 256,
    'topk' : 10,
    
    # if do not use a pretrained, train a new one, and save it in 'gvae/gvae.pth'
    'pretrained_gvae' : True, 
    'zdim' : 128,
    
    #clso
    'step_num' : 1,
    'eta' : 0.2,
    'delta_eta' : 0.2,
    'random_num' : 2000, # before searching, how many architectures should be sampled from the latent space ?
    }

configs['gvae_path'] = 'gvae/gvae_600.pth'

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
        self.database = NASBenchDataBase(configs['nas_bench_101_database_path'])
        self.configs = configs
        
        if configs['pretrained_gvae']:
            pass
        else:
            get_gvae()
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
                        latent = deepcopy(latent_copy) + 0.05*torch.randn_like(latent_copy)
                    else:
                        latent = deepcopy(latent_copy)
                    
                    latent = nn.Parameter(latent.cuda(), requires_grad = True)
                    optimizer = torch.optim.SGD([latent], lr = eta)
            
                    acc_p = (- self.gvae.icnn(latent) + 1.0)
                    acc_p.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                    with torch.no_grad():
                        arch_adj, arch_ops = self.gvae.get_tensor(deepcopy(latent.unsqueeze(0)))
                        arch_adj, arch_ops = self.gvae.conver_tensor2arch(arch_adj, arch_ops)
                        
                    try:
                        arch_hash = self.database.query_hash_by_matrix(arch_adj, arch_ops)
                        if arch_hash in self.labeled_set[0]:
                            eta = eta + self.configs['delta_eta']
                            
                        else:
                            
                            arch_info = self.database.query_by_matrix(arch_adj, arch_ops)
                            acc = arch_info['avg_validation_accuracy']
                        
                            logging.info(
                                'Obtain an new architecture with acc:%f', acc)
                        
                            self.labeled_set[0].append(arch_hash)
                            
                            self.labeled_set[1] = torch.cat(
                                [self.labeled_set[1], deepcopy(latent.detach().cpu()).unsqueeze(0)])
                            
                            self.labeled_set[2] = torch.cat(
                                [self.labeled_set[2], torch.tensor([acc]).float()])
                            
                            new_arch = True
                    except:
                        pass
                          
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
                    # add some noise to explore the search space.
                    latents = latents.cuda() + 0.05*torch.randn_like(latents.cuda())
                    acc = acc.cuda() # + 0.01*torch.randn_like(acc.cuda())
                
                pred_acc = (-self.gvae.icnn(latents) + 1.0).squeeze()
                
                loss = mse(acc, pred_acc)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                objs.update(loss.data.item(), n)
                
                self.gvae.icnn.constraint_weights()
            
            scheduler.step()
        
        logging.info('Finetune the icnn, loss_pred:%e', objs.avg)
        
    def obtain_topk_performance(self, topk = 0):
        values, indices = self.labeled_set[2].topk(100)
        arch_hash = self.labeled_set[0][indices[topk]]
        arch_info = self.database.query_by_hash(arch_hash)
        return arch_info
      

if __name__ == '__main__':
    lso = CRLSO()
    lso.main_loop()
    
    best_acc = 0.0
    for i in range(50):
        arch_info = lso.obtain_topk_performance(topk = i)
        if arch_info['avg_test_accuracy'] > best_acc:
            best_arch_info = arch_info
            best_acc = arch_info['avg_test_accuracy']
        else:
            pass
    print(best_arch_info)
    
    # acc = 0.9422
    # min_abs = 100
    # arch_info = None
    # for index in range(lso.database.size):
    #     arch_info_ = lso.database.query_by_index(index)
    #     test_acc = arch_info_['avg_test_accuracy']
    #     if abs(acc - test_acc) < min_abs:
    #         arch_info = arch_info_
    #         min_abs = abs(acc - test_acc)
    #         print(test_acc)
        
    
        
        
        
        
        
        
        
            