import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Quaduplet2Loss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, bg_weight=0.25):
        super(Quaduplet2Loss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.bg_weight = bg_weight

    def forward(self, inputs, targets):
        """
        Does not calculate noise inputs with label -1
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        #print(inputs.shape, targets.shape)
        inputs_new = []
        bg = []
        targets_new = []
        targets_value = []
        for i in range(len(targets)):
            if targets[i] < 0:
                bg.append(inputs[i])
            else:
                inputs_new.append(inputs[i])
                targets_new.append(targets[i])
                targets_value.append(targets[i].cpu().numpy().item())
        if len(set(targets_value)) < 2:
            tmp_loss = torch.zeros(1)
            tmp_loss = tmp_loss[0]
            tmp_loss = tmp_loss.to(targets.device)
            return tmp_loss
        #print(targets_value)
        inputs_new = torch.stack(inputs_new)
        targets_new = torch.stack(targets_new)
        #print(inputs_new.shape, targets_new.shape)
        n = inputs_new.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs_new, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs_new, inputs_new.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        #print("Triplet ", dist)
        # For each anchor, find the hardest positive and negative
        mask = targets_new.expand(n, n).eq(targets_new.expand(n, n).t())
        #print(mask)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        #dist_ap = torch.cat(dist_ap)
        #dist_an = torch.cat(dist_an)
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        #y = dist_an.data.new()
        #y.resize_as_(dist_an.data)
        #y.fill_(1)
        #y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        bg = torch.stack(bg)
        m = bg.size(0)
        if m > 0:
            #print("m", m, "n", n)
            dist_p = torch.pow(inputs_new, 2).sum(dim=1, keepdim=True).expand(n, m)
            #print("dist_p", dist_p.shape)
            dist_bg = torch.pow(bg, 2).sum(dim=1, keepdim=True)
            #print("dist_bg", dist_bg.shape)
            dist_bg = dist_bg.expand(m, n)
            #print("dist_bg", dist_bg.shape)
            dist_new = dist_p + dist_bg.t()
            dist_new.addmm_(1, -2, inputs_new, bg.t())
            dist_new = dist_new.clamp(min=1e-12).sqrt()  # for numerical stability
            # For each anchor, find the hardest positive and negative
            #mask = targets_new.expand(n, ).eq(targets_new.expand(n, n).t())
            dist_ap, dist_an = [], []
            for i in range(n):
                #dist_ap.append(dist[i][mask[i]].max())
                dist_ap.append(dist[i].max())
                dist_an.append(dist_new[i].min())
            dist_ap = torch.stack(dist_ap)
            dist_an = torch.stack(dist_an)
            y = torch.ones_like(dist_an)
            loss_bg = self.ranking_loss(dist_an, dist_ap, y)
        loss += self.bg_weight*loss_bg

        return loss