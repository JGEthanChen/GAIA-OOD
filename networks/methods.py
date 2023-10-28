import torch
import torch.nn as nn
# from networks.resnet import *
# from networks.resnetv2 import *
# from networks.wresnet import *
from networks import resnet
from networks import resnetv2
from networks import wresnet
from metrics import cal_metric, print_all_results
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
            ('cal_score_msp', lambda *a, **kw: cal_score_msp(*a, **kw)),
        ])
        
    def cal_score(self, data, device):
        return self.KNOWN_METHODS[self.opt.cal_method](self.model, data, device, self.hooks)
    
    def get_scores(self, id_dataset, ood_name, ood_datasets):
        device = self.device
        self.hooks = None
        if self.opt.hook == 'bn':
            self.hooks = get_bn_hooks(self.model, self.opt.model_name)
        elif self.opt.hook == 'before_head':
            self.hooks = get_beforehead_hooks(self.model, self.opt.model_name)
        
        self.model.eval()
        start_time = time.time()

        print('compute in-distribution dataset')    
        know = torch.tensor([], device=device)
        for data,_ in id_dataset:
            data = data.to(device)
            know = torch.cat([know, -self.cal_score(data, device)], 0)
        
        print('compute ood datasets')
        novels = []
        for idx_ood in range(len(ood_name)):
            novel = torch.tensor([], device=device)
            print('process '+ ood_name[idx_ood])
            for data,_ in ood_datasets[idx_ood]:
                data = data.to(device)
                novel = torch.cat([novel, -self.cal_score(data, device)], 0)
            novels.append(novel.cpu().tolist())

        
        results = []
        know = know.cpu().numpy()
        end_time = time.time()
        print('Computation cost: '+ str((end_time-start_time)/3600) + ' h')

        for i, novel in enumerate(novels):
            print('Process OOD '+ood_name[i]+' Scores...')
            result = cal_metric(know, np.array(novel))
            results.append(result)

        print_all_results(results, ood_name, "ours")