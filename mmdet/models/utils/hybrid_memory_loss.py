# Ge et al. Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID.  # noqa
# Written by Yixiao Ge.

import torch
import torch.nn.functional as F
from torch import autograd, nn

from mmdet.utils import all_gather_tensor

try:
    # PyTorch >= 1.6 supports mixed precision training
    from torch.cuda.amp import custom_fwd, custom_bwd
    class HM(autograd.Function):

        @staticmethod
        @custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            for x, y in zip(inputs, indexes):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None
except:
    class HM(autograd.Function):

        @staticmethod
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            for x, y in zip(inputs, indexes):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(
        inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device)
    )

class HybridMemory(nn.Module):
    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_memory = num_memory

        self.momentum = momentum
        self.temp = temp

        self.idx = torch.zeros(num_memory).long()

        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())
    
    @torch.no_grad()
    def _init_ids(self, ids):
        self.idx.data.copy_(ids.long().to(self.labels.device))

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))
    
    @torch.no_grad()
    def get_cluster_ids(self, indexes):
        return self.labels[indexes].clone()

    def forward(self, results, indexes):
        inputs = results
        inputs = F.normalize(inputs, p=2, dim=1)

        # inputs: B*2048, features: N*2048
        inputs = hm(inputs, indexes, self.features, self.momentum) #B*N, similarity
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums

        targets = self.labels[indexes].clone()
        labels = self.labels.clone() #shape: N, unique label num: u

        sim = torch.zeros(labels.max() + 1, B).float().cuda() #u*B
        sim.index_add_(0, labels, inputs.t().contiguous()) #
        nums = torch.zeros(labels.max() + 1, 1).float().cuda() #many instances belong to a cluster, so calculate the number of instances in a cluster
        nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda()) #u*1
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim) #average features in each cluster, u*B
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous()) #sim: u*B, mask:u*B, masked_sim: B*u
        return F.nll_loss(torch.log(masked_sim + 1e-6), targets)


try:
    # PyTorch >= 1.6 supports mixed precision training
    from torch.cuda.amp import custom_fwd, custom_bwd
    class HMUniqueUpdate(autograd.Function):

        @staticmethod
        @custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            unique = set()
            for x, y in zip(inputs, indexes):
                if y.item() in unique:
                    continue
                else:
                    unique.add(y.item())
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None
except:
    class HMUniqueUpdate(autograd.Function):

        @staticmethod
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            unique = set()
            for x, y in zip(inputs, indexes):
                if y.item() in unique:
                    continue
                else:
                    unique.add(y.item())
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None


def hmuniqueupdate(inputs, indexes, features, momentum=0.5):
    return HMUniqueUpdate.apply(
        inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device)
    )


class HybridMemoryMultiFocalPercent(nn.Module):
    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2, top_percent=0.1):
        super(HybridMemoryMultiFocalPercent, self).__init__()
        self.num_features = num_features
        self.num_memory = num_memory

        self.momentum = momentum
        self.temp = temp

        #for mutli focal
        self.top_percent = top_percent

        self.idx = torch.zeros(num_memory).long()

        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())
    
    @torch.no_grad()
    def _init_ids(self, ids):
        self.idx.data.copy_(ids.long().to(self.labels.device))

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))
    
    @torch.no_grad()
    def get_cluster_ids(self, indexes):
        return self.labels[indexes].clone()

    def forward(self, results, indexes):
        inputs = results
        inputs = F.normalize(inputs, p=2, dim=1)

        # inputs: B*2048, features: N*2048
        inputs = hm(inputs, indexes, self.features, self.momentum) #B*N, similarity
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax_multi_focal(vec, mask, dim=1, targets=None, epsilon=1e-6):
            exps = torch.exp(vec)
            one_hot_pos = torch.nn.functional.one_hot(targets, num_classes=exps.shape[1])
            # assert exps.shape==one_hot_pos.shape
            one_hot_neg = one_hot_pos.new_ones(size=one_hot_pos.shape)
            one_hot_neg = one_hot_neg - one_hot_pos
            masked_exps = exps * mask.float().clone()
            neg_exps = exps.new_zeros(size=exps.shape)
            neg_exps[one_hot_neg>0] = masked_exps[one_hot_neg>0]
            ori_neg_exps = neg_exps
            neg_exps = neg_exps/neg_exps.sum(dim=1, keepdim=True)
            new_exps = masked_exps.new_zeros(size=exps.shape)
            new_exps[one_hot_pos>0] = masked_exps[one_hot_pos>0]
            # topk_values, topk_indexes = neg_exps.topk(dim=1, k=self.topk)

            sorted, indices = torch.sort(neg_exps, dim=1, descending=True)
            sorted_cum_sum = torch.cumsum(sorted, dim=1)
            sorted_cum_diff = (sorted_cum_sum - self.top_percent).abs()
            sorted_cum_min_indices = sorted_cum_diff.argmin(dim=1)
            min_values = sorted[torch.range(0, sorted.shape[0]-1).long(), sorted_cum_min_indices]
            min_values = min_values.unsqueeze(dim=-1) * ori_neg_exps.sum(dim=1, keepdim=True)
            ori_neg_sum = ori_neg_exps.sum(dim=1, keepdim=True)
            ori_neg_exps[ori_neg_exps<min_values] = 0
            # print((ori_neg_exps/ori_neg_sum).sum(dim=1)[:20])

            new_exps[one_hot_neg>0] = ori_neg_exps[one_hot_neg>0]

            masked_exps = new_exps

            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums

        targets = self.labels[indexes].clone()
        labels = self.labels.clone() #shape: N, unique label num: u

        sim = torch.zeros(labels.max() + 1, B).float().cuda() #u*B
        sim.index_add_(0, labels, inputs.t().contiguous()) #
        nums = torch.zeros(labels.max() + 1, 1).float().cuda() #many instances belong to a cluster, so calculate the number of instances in a cluster
        nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda()) #u*1
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim) #average features in each cluster, u*B
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax_multi_focal(sim.t().contiguous(), mask.t().contiguous(), targets=targets) #sim: u*B, mask:u*B, masked_sim: B*u
        return F.nll_loss(torch.log(masked_sim + 1e-6), targets)
