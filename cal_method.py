import torch
import torch.nn as nn
# torch.cuda.set_device(args.cuda)

def get_square(gradients):
    gradients = torch.cat(gradients, dim=1)
    gradients = torch.pow(gradients, 2)
    var = gradients.mean(dim=(-1))
    return var

def cal_score_msp(model, inputs, device=None, hooks=None):
    logits = model(inputs)
    conf, _ = torch.max((logits), dim=-1)
    return -conf.detach()

def cal_zero(net, input, device=None, hooks=None, p=2):
    net.zero_grad()
    y = net(input)
    y.max(dim=1).values.sum().backward()
    # y.sum().backward()
    gradients = [hook.data for hook in hooks]
    gradients = [torch.where(grad != 0, torch.ones_like(grad), torch.zeros_like(grad)) for grad in gradients]
    scores = [grad.mean(dim=(-1, -2)) for grad in gradients]
    square_scores = get_square(scores)
    return square_scores

def cal_grad_value(net, input, device, hooks=None, p=2):
    net.zero_grad()
    y = net(input)
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    
    loss = logsoftmax(y)
    loss.sum().backward(retain_graph=True)
    before_head_grad = hooks[-1].data.mean(dim=(-1, -2))
    output_component = torch.sqrt(torch.abs(before_head_grad).mean(dim=1))
    output_component = output_component.unsqueeze(dim=1)

    loss = net.before_head_data
    loss.sum().backward()
    gradients = [hook.data for hook in hooks]
    gradients = gradients[:-1]
    gradients = [grad.mean(dim=(-1, -2)) for grad in gradients]
    inner_component = torch.abs(torch.cat(gradients, dim=1))
    score = torch.pow(inner_component / output_component, 2).mean(dim=1)
    return score.detach()
