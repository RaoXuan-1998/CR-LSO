from collect_101_dataset import NAS_Bench_101_Dataset
from models import ArchGVAE, GNN_Predictor
import torch.nn as nn
import logging
import sys
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from utils import AvgrageMeter
import tqdm
from scipy.stats import pearsonr, kendalltau, spearmanr

gvae_configs = {
    
    'portion_for_semipredictor' : 0.01,
    'semi_epoch_num' : 200,
    'semi_save_path' : 'semi_predictor',
    'semi_batch_size' : 32,
    
    'layer_num' : 3,
    'batch_size' : 512,
    'lr' : 1e-4,
    'betas' : (0.0, 0.5),
    'weight_decay' : 0.0,
    'data_path' : 'dataset/nas_bench_101.pth',
    'zdim' : 64,
    'hdim' : 512,
    'epoch_num' : 200,
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
        n = len(arch.adj)
        arch.cuda()
        
        acc = arch.valid_acc.cuda()
        
        pred_acc = predictor(arch)
        
        loss = mse(acc, pred_acc.squeeze())
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        objs_pred.update(loss.data.item(), n)
    
    return objs_pred.avg

def validate_semipredictor(predictor, valid_loader):
      
    mse = nn.MSELoss(reduction = 'mean').cuda()
    predictor.eval()
    objs_pred = AvgrageMeter()
    
    acc_all = []
    pred_acc_all = []
    
    with torch.no_grad():
        for step, arch in enumerate(valid_loader): 
            n = len(arch.adj)
            arch.cuda()
            acc = arch.valid_acc.cuda()
            
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

        
def train_gvae(gvae, predictor, train_loader, optimizer):
    
    mse = nn.MSELoss(reduction = 'mean')
    gvae.train()
    
    objs = AvgrageMeter()
    objs_res = AvgrageMeter()
    objs_kld = AvgrageMeter()
    objs_pred = AvgrageMeter()
    
    # the acc is not available in semisupervised learning
    for step, arch in enumerate(train_loader): 
        n = len(arch.adj)
        arch = arch.cuda()
        
        with torch.no_grad():
            acc = predictor(arch).squeeze()
                   
            mu, logvar = gvae.encode(arch)
        
            loss, res, kld, z = gvae.loss(mu, logvar, arch.adj, arch.ops)
        
        acc_p = (- gvae.icnn(mu) + 1.0).squeeze()
        
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
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:split + 20000])
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
        pred_objs= train_semipredictor(semipredictor, optimizer, train_loader)
        pred_objs_valid, p, tau, s = validate_semipredictor(semipredictor, valid_loader)
        scheduler.step()
        logging.info('Train the semi-predictor, epoch:%d, loss_pred_train:%e loss_pred_valid:%e, p:%.3f, tau:%.3f s:%.3f',
                     epoch, pred_objs, pred_objs_valid, p, tau, s)
        
    torch.save(semipredictor, configs['semi_save_path'] + '/semi_predictor.pth')
    
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
        loss, res, kld, pred = train_gvae(gvae, None, gvae_train_loader, optimizer) 
        scheduler.step()
        
        logging.info('Train gvae, epoch:%d, loss:%e res:%e kld:%e pred:%e',
                     epoch, loss, res, kld, pred
                     )
    
    train_loader = DataLoader(
        dataset, batch_size = 1,
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        )
    
     
    arch_hash_list = []
    acc_list = []
    z_list = []
    
    with torch.no_grad():
        for step, arch in enumerate(train_loader): 
            arch = arch.cuda()
            acc = arch.valid_acc            
            
            mu, logvar = gvae.encode(arch)
            
            arch_hash = dataset.hash_list[arch.arch_index]
            arch_hash_list.append(arch_hash)
            
            acc_list.append(acc.cpu())
            z_list.append(mu.cpu())
 
    acc_list = torch.cat(acc_list, dim = 0)
    z_list = torch.cat(z_list, dim = 0)

    gvae.labeled_set = [arch_hash_list, z_list, acc_list]
    
    torch.save(gvae, 'gvae/gvae_semisupervised_noICNN_4236.pth')
    
if __name__ == '__main__':
    get_gvae()