import torch
import torch.nn as nn
# from networks.resnet import *
# from networks.resnetv2 import *
# from networks.wresnet import *
from networks import resnet
from networks import resnetv2
from networks import wresnet
from metrics import cal_metric, print_results, print_all_results
import time
from collections import OrderedDict
from cal_method import *
from hook import *
import numpy as np

class Methods():
    def __init__(self, opt):
        super(Methods, self).__init__()
        self.opt = opt
        self.device = torch.device('cuda:'+f'{opt.cuda}') if torch.cuda.is_available() else 'cpu'
        device = self.device
        if opt.model_arch == 'resnet':
            self.model = resnet.KNOWN_MODELS[opt.model_name](num_classes=opt.num_classes)
            checkpoint = torch.load(opt.model_path, map_location='cpu')
        elif opt.model_arch == 'wresnet':
            self.model = wresnet.KNOWN_MODELS[opt.model_name](num_classes=opt.num_classes)
            checkpoint = torch.load(opt.model_path, map_location='cpu')
        elif opt.model_arch == 'resnetv2':
            self.model = resnetv2.KNOWN_MODELS[opt.model_name](head_size=opt.num_classes)
            checkpoint = torch.load(opt.model_path, map_location='cpu')
        
        if opt.model_arch == 'resnetv2':
            self.model.load_state_dict_custom(checkpoint['model'])
        elif 'state_dict' in checkpoint.keys():
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.KNOWN_METHODS = OrderedDict([
            ('cal_zero', lambda *a, **kw: cal_zero(*a, **kw)),
            ('cal_grad_value', lambda *a, **kw: cal_grad_value(*a, **kw)),
        ])
        
        
    def cal_score(self, device, dataset, hooks, dataset_name=''):       
        score = torch.tensor([], device=device)
        for data,_ in dataset:
            data = data.to(device)
            score = torch.cat([score, -self.KNOWN_METHODS[self.opt.cal_method](self.model, data, device, hooks)], 0)
        return score

    def get_score(self, id_dataset, ood_dataset, ood_name, device):
        
        if self.opt.hook == 'bn':
            hooks = get_bn_hooks(self.model, self.opt.model_name)
        elif self.opt.hook == 'before_head':
            hooks = get_beforehead_hooks(self.model, self.opt.model_name, self.opt.cal_method, ood_name)
            
        start_time = time.time()
        print('compute in-distribution dataset...') 
        know = self.cal_score(device, id_dataset, hooks, 'id_dataset')
        know = know.cpu().numpy()
        
        print('compute ood dataset '+ood_name+'...')
        novel = self.cal_score(device, ood_dataset, hooks, ood_name)
        novel = np.array(novel.cpu().tolist())
        
        end_time = time.time()
        result = cal_metric(know, novel)
        
        print('process result '+ood_name+' total images num: '+str(len(know) + len(novel))+'...')
        print('Computation cost: '+ str((end_time-start_time)/3600) + ' h')
        
        print_results(result, ood_name, "ours")
        return result
        
        
        
    def get_scores(self, id_dataset, ood_name, ood_datasets):
        device = self.device

        self.model.eval()
        results = []
        for idx_ood in range(len(ood_name)):
            result = self.get_score(id_dataset, ood_datasets[idx_ood], ood_name[idx_ood], device)
            results.append(result)
        
        print_all_results(results, ood_name, "ours")