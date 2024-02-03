import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opts import parser
from torch import Tensor

args = parser.parse_args()
if args.dataset == 'cifar100':
    num_classes = 100
elif args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset == 'tinyimagenet':
    num_classes = 200
elif args.dataset =='fashion':
    num_classes=10
else: #iNat18
    num_classes = 8142


# def sb_loss(input_values, ib,labels_onehot,input):
#     """Computes the focal loss"""
#     pt = torch.sum(labels_onehot * F.softmax(input, dim=-1), dim=-1)
#     epsilon = 1
#     loss = (input_values  + epsilon * (1 - pt))*ib
#     return loss.mean()
#
# class SBLoss(nn.Module):
#     def __init__(self, weight=None, alpha=10000.):
#         super(IBLoss, self).__init__()
#         assert alpha > 0
#         self.alpha = alpha
#         self.epsilon = 0.1
#         self.weight = weight
#     def forward(self, input, target, features):
#         grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, num_classes)), 1)
#         labels_onehot = F.one_hot(target, num_classes=100).to(device=input.device,
#                                                                            dtype=input.dtype)
#         ib = grads * features.reshape(-1)  # 两者相乘 (fk-yk)*hl
#         ib = self.alpha / (ib + self.epsilon)  # 变成除法，倒过来s
#         return ib_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib,labels_onehot,input)  # 就是将两个相乘


def sb_loss(input_values, ib, gamma,pt,target,alpha):
    """Computes the ib focal loss"""
    epsilon=1
    gamma = 1
    pt = torch.exp(-input_values)
    FL = input_values * ((1 - pt) ** (gamma))
    loss = (FL + epsilon * torch.pow(1 - pt, gamma + 1))*ib
    return loss.mean()

class SBLoss(nn.Module):
    def __init__(self, weight=None, alpha=10000., gamma=0.):
        super(IB_FocalLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.0001
        self.weight = weight
        self.gamma = gamma

    def forward(self, input, target, features):
        #grads = torch.sum(torch.norm(F.softmax(input, dim=1) - F.one_hot(target, num_classes),p=2,dim=1,keepdim=True))  # N * 1
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, num_classes)), 1)  # N * 1
        ib = grads*(features.reshape(-1))
        ib = self.alpha / (ib + self.epsilon)
        pt = input  # BH[WD]
        return sb_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib, self.gamma,pt,target,self.alpha)

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    #loss = (1 - p) ** gamma * input_values
    loss = (1-p) ** gamma * input_values * 10
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class EQL(nn.Module):
    def __init__(self,cls_fre_per,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 lambda_=0.00177,
                 version="v0_5"
                 ):
        super(EQL, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.lambda_ = lambda_
        self.version = version
        self.freq_info=cls_fre_per


    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]

        target = expand_label(cls_score, label)

        eql_w = 1 - self.exclude_func() * self.threshold_func() * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')

        cls_loss = torch.sum(cls_loss * eql_w) / self.n_i

        return self.loss_weight * cls_loss

    def exclude_func(self):
        # instance-level weight
        bg_ind = self.n_c
        weight = (self.gt_classes != bg_ind).float()
        weight = weight.view(self.n_i, 1).expand(self.n_i, self.n_c)
        return weight

    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight

class Cyclical_FocalLoss(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, gamma_hc=0, eps: float = 0.1, reduction='mean', epochs=200,
                 factor=2):
        super(Cyclical_FocalLoss, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_hc = gamma_hc
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        self.epochs = epochs
        self.factor = factor # factor=2 for cyclical, 1 for modified
#        self.ceps = ceps
#        print("Asymetric_Cyclical_FocalLoss: gamma_pos=", gamma_pos," gamma_neg=",gamma_neg,
#              " eps=",eps, " epochs=", epochs, " factor=",factor)

    def forward(self, inputs, target, epoch):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
#        print("input.size(),target.size()) ",inputs.size(),target.size())
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        if len(list(target.size()))>1:
            target = torch.argmax(target, 1)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # Cyclical
#        eta = abs(1 - self.factor*epoch/(self.epochs-1))
        if self.factor*epoch < self.epochs:
            eta = 1 - self.factor *epoch/(self.epochs-1)
        else:
            eta = (self.factor*epoch/(self.epochs-1) - 1.0)/(self.factor - 1.0)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        positive_w = torch.pow(1 + xs_pos,self.gamma_hc * targets)
        log_preds = log_preds * ((1 - eta)* asymmetric_w + eta * positive_w)

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class LMFLoss(nn.Module):
    def __init__(self, cls_num_list, weight, alpha=1, beta=1, gamma=2, max_m=0.5, s=30):
        super().__init__()
        self.focal_loss = FocalLoss(weight, gamma)
        self.ldam_loss = LDAMLoss(cls_num_list, max_m, weight, s)
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        focal_loss_output = self.focal_loss(output, target)
        ldam_loss_output = self.ldam_loss(output, target)
        total_loss = self.alpha * focal_loss_output + self.beta * ldam_loss_output
        return total_loss