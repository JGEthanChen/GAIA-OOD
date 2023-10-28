import torch
import torch.nn as nn

# All hooks need data type
class Grad_all_hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.save_grad)
        self.data = torch.Tensor()

    def save_grad(self, module, input, output):
        def _stor_grad(grad):
            self.data = grad.detach()
        output.register_hook(_stor_grad)

    def close(self):
        self.hook.remove()

class Activation_all_hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.save_activations)
        self.data = torch.Tensor()

    def save_activations(self, module, input, output):
        self.data = output.detach()

    def close(self):
        self.hook.remove()

class Grad_feature_hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.save_grad)
        self.data = torch.Tensor()
        self.feature = torch.Tensor()

    def save_grad(self, module, input, output):
        def _stor_grad(grad):
            self.data = grad.detach()
        output.register_hook(_stor_grad)
        self.feature = output.clone()

    def close(self):
        self.hook.remove()
        
def get_conv_hooks(net):
    conv_hooks = []
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            conv_hooks.append(Grad_all_hook(module))
    return conv_hooks

def get_bn_hooks(net, model_name):
    bn_hooks = []
    if model_name == 'BiT-S-R101x1':
        cnt = 0
        for module in net.body.block4.modules():
            if isinstance(module, nn.GroupNorm):
                bn_hooks.append(Grad_all_hook(module))
        print("model bn hook length", len(bn_hooks))
    else:
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_hooks.append(Grad_all_hook(module))
    return bn_hooks

def get_beforehead_hooks(net, model_name):
    beforehead_hooks = []
    module_list = []
    if model_name in['resnet34', 'resnet18', 'resnet50']:
        # for module in net.modules():
            # if isinstance(module, nn.BatchNorm2d):
            #     module_list.append(module)
        for module in net.layer3.modules():
            if isinstance(module, nn.BatchNorm2d):
                module_list.append(module)
        for module in net.layer4.modules():
            if isinstance(module, nn.BatchNorm2d):
                module_list.append(module)
        # module_list.append(net.layer4)
    # elif model_name in ['resnet50']:
    #     for module in net.layer4.modules():
    #         if isinstance(module, nn.BatchNorm2d):
    #             module_list.append(module)
    #     module_list.append(net.layer4)
    elif model_name in ['wrn_40_2']:
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):
                module_list.append(module)
        module_list.append(net.AdaptAvgPool)
    elif model_name in ['vgg', 'vgg16']:
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):
                module_list.append(module)
        module_list.append(net.pool4)
    elif model_name in ['BiT-S-R101x1', 'BiT-M-R152x2']:
        # for module in net.body.modules():
            # if isinstance(module, nn.GroupNorm):
            #     module_list.append(module)
        # for module in net.body.block3.modules():
        #     if isinstance(module, nn.GroupNorm):
        #         module_list.append(module)
        for module in net.body.block4.modules():
            if isinstance(module, nn.GroupNorm):
                module_list.append(module)
        module_list.append(net.before_head.gn)
        module_list.append(net.before_head)
    elif model_name in ['BiT-M-R50x1']:
        for module in net.body.modules():
            if isinstance(module, nn.GroupNorm):
                module_list.append(module)
        module_list.append(net.before_head.gn)
        module_list.append(net.before_head)
    for index, module in enumerate(module_list):
        if index < len(module_list)-1:
            beforehead_hooks.append(Grad_all_hook(module))
        else:
            beforehead_hooks.append(Grad_feature_hook(module))
    return beforehead_hooks