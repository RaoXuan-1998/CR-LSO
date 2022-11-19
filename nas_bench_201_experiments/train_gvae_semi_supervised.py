from collect_201_dataset import NAS_Bench_201_Dataset
from models import ICNN, ArchGVAE, MLP, GNN_Predictor
import torch.nn as nn
import logging
import sys
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from utils import AvgrageMeter, create_exp_dir
import tqdm
from scipy.stats import pearsonr, kendalltau, spearmanr

gvae_configs = {
    'dataset' : 'ImageNet',
    'batch_size' : 256,
    'lr' : 1e-4,
    'betas' : (0.0, 0.5),
    'weight_decay' : 3e-5, 
    'data_path' : 'dataset/nas_201_dataset.pth',
    'graph_clip' : 5.0,
    'portion_for_semipredictor' : 0.0192,
    'semi_epoch_num' : 200,
    'semi_save_path' : 'semi_predictor',
    'semi_batch_size' : 32,
    'zdim' : 16,
    'hdim' : 256,
    'layer_num' : 3,
    'epoch_num' : 200,
    'save_path' : 'statistics',
    'datasets' : ['CIFAR10', 'CIFAR100', 'ImageNet'],
    'seed' : 0,
    }


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = log_format, datefmt = '%m/%d %I:%M:%S %p')

# fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)

def train_semipredictor(predictor, optimizer, train_loader, dataset = None):

    mse = nn.MSELoss(reduction = 'mean').cuda()
    predictor.train()
    objs_pred = AvgrageMeter()
    for step, arch in enumerate(train_loader): 
        n = len(arch.tensor)
        arch.cuda()
        
        if dataset == 'CIFAR10':
            acc = arch.valid_acc_cifar10
        elif dataset == 'CIFAR100':
            acc = arch.valid_acc_cifar100
        elif dataset == 'ImageNet':
            acc = arch.valid_acc_imagenet
        
        acc = 0.01*acc
        pred_acc = predictor(arch)
        
        loss = mse(acc, pred_acc.squeeze())
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        objs_pred.update(loss.data.item(), n)
    
    return objs_pred.avg

def validate_semipredictor(predictor, valid_loader, dataset = None):
      
    mse = nn.MSELoss(reduction = 'mean').cuda()
    predictor.eval()
    objs_pred = AvgrageMeter()
    
    acc_all = []
    pred_acc_all = []
    
    with torch.no_grad():
        for step, arch in enumerate(valid_loader): 
            n = len(arch.tensor)
            arch.cuda()
             
            if dataset == 'CIFAR10':
                acc = arch.valid_acc_cifar10
            elif dataset == 'CIFAR100':
                acc = arch.valid_acc_cifar100
            elif dataset == 'ImageNet':
                acc = arch.valid_acc_imagenet
                
            acc = 0.01*acc.cuda()
            acc_all.append(acc)
            
            pred_acc = predictor(arch)
            
            pred_acc_all.append(pred_acc.squeeze())
            loss = mse(acc, pred_acc.squeeze())
            
            objs_pred.update(loss.data.item(), n)
            
    acc_all = torch.cat(acc_all, dim = 0).squeeze().detach().cpu().numpy()
    pred_acc_all = torch.cat(pred_acc_all, dim = 0).squeeze().detach().cpu().numpy()
    
    p = pearsonr(acc_all, pred_acc_all)[0]
    tau = kendalltau(acc_all, pred_acc_all)[0]
    s = spearmanr(acc_all, pred_acc_all)[0]

    return objs_pred.avg, p, tau, s       
        
def train_gvae(gvae, predictor, train_loader, optimizer, dataset = None):
    
    mse = nn.MSELoss(reduction = 'mean')
    gvae.train()
    
    objs = AvgrageMeter()
    objs_res = AvgrageMeter()
    objs_kld = AvgrageMeter()
    objs_pred = AvgrageMeter()
    
    # the acc is not available in semisupervised learning
    for step, arch in enumerate(train_loader): 
        n = len(arch.tensor)
        arch = arch.cuda()
        
        # do not use the true labels
        with torch.no_grad():
            acc = predictor(arch).squeeze()
            
        mu, logvar = gvae.encode(arch)
        
        loss, res, kld, z = gvae.loss(mu, logvar, arch.tensor)
        
        acc_p = (- gvae.icnn(z) + 1.0).squeeze()
        
        loss_p = mse(acc_p, acc)
        loss = loss + loss_p
        
        loss.backward()
        nn.utils.clip_grad_norm_(gvae.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()
        gvae.icnn.constraint_weights()
        
        objs.update(loss.data.item(), n) 
        objs_res.update(res.data.item(), n)
        objs_kld.update(kld.data.item(), n)
        objs_pred.update(loss_p.data.item(), n)
        
    return objs.avg, objs_res.avg, objs_kld.avg, objs_pred.avg

def validate_gvae(gvae, valid_loader, dataset = None):
    
    mse = nn.MSELoss(reduction = 'mean')
    gvae.train()
    
    objs = AvgrageMeter()
    objs_res = AvgrageMeter()
    objs_kld = AvgrageMeter()
    objs_pred = AvgrageMeter()
    
    # the acc is not available in semisupervised learning
    with torch.no_grad():
        for step, arch in enumerate(valid_loader): 
            n = len(arch.tensor)
            arch = arch.cuda()
            
            if dataset == 'CIFAR10':
                acc = arch.valid_acc_cifar10
            elif dataset == 'CIFAR100':
                acc = arch.valid_acc_cifar100
            elif dataset == 'ImageNet':
                acc = arch.valid_acc_imagenet
                
            acc = 0.01*acc
                
            mu, logvar = gvae.encode(arch)
            
            loss, res, kld, z = gvae.loss(mu, logvar, arch.tensor)
            
            acc_p = (- gvae.icnn(z) + 1.0).squeeze()
            
            loss_p = mse(acc_p, acc)
            loss = loss + loss_p
            
            objs.update(loss.data.item(), n) 
            objs_res.update(res.data.item(), n)
            objs_kld.update(kld.data.item(), n)
            objs_pred.update(loss_p.data.item(), n)
        
    return objs.avg, objs_res.avg, objs_kld.avg, objs_pred.avg


def get_gvae(configs = gvae_configs):
    # Train a semi-supervised predictor
    
    dataset = torch.load(configs['data_path'])
    
    torch.manual_seed(configs['seed'])
    torch.cuda.manual_seed(configs['seed'])
    np.random.seed(configs['seed'])
    
    data_num = len(dataset)
    indices = list(range(data_num))
    split = int(np.floor(configs['portion_for_semipredictor'] * data_num))

    train_loader = DataLoader(
        dataset, batch_size = configs['semi_batch_size'],
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        )
    
    valid_loader = DataLoader(
        dataset, batch_size = 1048,
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        )

    semipredictor = GNN_Predictor().cuda()
    
    optimizer = torch.optim.Adam(
        semipredictor.parameters(),
        lr = 1e-4, betas = configs['betas'], weight_decay = 0.0
        )
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max = float(configs['semi_epoch_num']), eta_min = 1e-5
        )
    
    for epoch in tqdm.tqdm(range(configs['semi_epoch_num'])):
        pred_objs= train_semipredictor(semipredictor, optimizer, train_loader, dataset = configs['dataset'])
        pred_objs_valid, p, tau, s = validate_semipredictor(semipredictor, valid_loader, dataset = configs['dataset'])
        scheduler.step()
        logging.info('Train the semi-predictor, epoch:%d, loss_pred_train:%e loss_pred_valid:%e, p:%.3f, tau:%.3f s:%.3f',
                     epoch, pred_objs, pred_objs_valid, p, tau, s)
        
    torch.save(semipredictor, configs['semi_save_path'] + '/semi_predictor_{}.pth'.format(configs['dataset']))
    
    for parameter in semipredictor.parameters():
        parameter.requires_grad = False
        
    gvae_train_loader = DataLoader(
        dataset, batch_size = configs['batch_size'],
        shuffle = True)
          
    gvae = ArchGVAE(hdim = configs['hdim'], zdim = configs['zdim'], layers = configs['layer_num']).cuda()
    
    optimizer = torch.optim.Adam(
        gvae.parameters(),
        lr = configs['lr'], betas = configs['betas'], weight_decay = configs['weight_decay']
        )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = float(configs['epoch_num']), eta_min = 0.0)
    for epoch in tqdm.tqdm(range(configs['epoch_num'])):
        loss, res, kld, pred = train_gvae(gvae, semipredictor, gvae_train_loader, optimizer, dataset = configs['dataset']) 
        loss_valid, res_valid, kld_valid, pred_valid  = validate_gvae(gvae, valid_loader, dataset = configs['dataset'])
        scheduler.step()
        
        logging.info('Train gvae, epoch:%d, loss_train:%e loss_pred:%e, loss_valid:%e, loss_pred:%e',
                     epoch, loss, pred, loss_valid, pred_valid
                     )
    # save the labeled set for the following cr-lso
    str_list = []
    acc_list = []
    z_list = []
    with torch.no_grad():
        for step, arch in enumerate(train_loader): 
            arch = arch.cuda()
                        
            if configs['dataset'] == 'CIFAR10':
                acc = arch.valid_acc_cifar10
            elif configs['dataset'] == 'CIFAR100':
                acc = arch.valid_acc_cifar100
            elif configs['dataset'] == 'ImageNet':
                acc = arch.valid_acc_imagenet
                
            mu, logvar = gvae.encode(arch)
            
            acc = 0.01*acc
            
            str_list = str_list + arch.arch_str
            acc_list.append(acc.squeeze().cpu())
            z_list.append(mu.squeeze().cpu())
    
 
    acc_list = torch.cat(acc_list, dim = 0)
    z_list = torch.cat(z_list, dim = 0)

    gvae.labeled_set = [str_list, z_list, acc_list]
     
    torch.save(gvae, 'gvae/gvae_semi_16dim_{}.pth'.format(configs['dataset']))
    
if __name__ == '__main__':
    get_gvae()