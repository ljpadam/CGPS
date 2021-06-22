import os.path as osp
import warnings

from mmcv.runner import Hook
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from mmdet.utils import get_dist_info
from mmdet.utils import all_gather_tensor, synchronize
from mmdet.core.label_generators import LabelGenerator
import mmcv


class ClusterHook(Hook):
    """Evaluation hook.

    Notes:
        If new arguments are added for EvalHook, tools/test.py may be
    effected.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        start (int, optional): Evaluation starting epoch. It enables evaluation
            before the training starts if ``start`` <= the resuming epoch.
            If None, whether to evaluate is merely decided by ``interval``.
            Default: None.
        interval (int): Evaluation interval (by epochs). Default: 1.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self, train_dataloaders, start=None, interval=1, logger=None, cfg=None, epoch_interval=1, **eval_kwargs):
        self.dataloaders = train_dataloaders
        self.datasets = [i.dataset for i in train_dataloaders]
        self.logger = logger
        self.cfg = cfg
        self.label_generator = LabelGenerator(self.cfg, self.dataloaders)
        self.epoch=0
        #cluster every interval epochs
        self.epoch_interval=epoch_interval
    
    def before_train_epoch(self, runner):
        self.logger.info('start clustering for updating, pseudo labels')
        if self.epoch%self.epoch_interval != 0:
            self.epoch += 1
            return
        memory_features = []
        start_ind = 0
        for idx, dataset in enumerate(self.datasets):
            memory_features.append(
                runner.model.module.roi_head.bbox_head.loss_reid
                .features[start_ind : start_ind + dataset.id_num]
                .clone()
                .cpu()
            )
            start_ind += dataset.id_num

        # generate pseudo labels
        if self.cfg.PSEUDO_LABELS.cluster == "dbscan_context" or self.cfg.PSEUDO_LABELS.cluster == "dbscan_context_kmeans":
            pseudo_labels, label_centers = self.label_generator(
            memory_features=memory_features,
            image_inds=runner.model.module.roi_head.bbox_head.loss_reid.idx.clone().cpu()
        )
        elif self.cfg.PSEUDO_LABELS.cluster == "dbscan_context_eps" or self.cfg.PSEUDO_LABELS.cluster == "dbscan_context_eps_all" or self.cfg.PSEUDO_LABELS.cluster == "dbscan_context_eps_all_weight":
            pseudo_labels, label_centers = self.label_generator(
            memory_features=memory_features,
            image_inds=runner.model.module.roi_head.bbox_head.loss_reid.idx.clone().cpu(),
            epoch=self.epoch
        )
        else:
            pseudo_labels, label_centers = self.label_generator(
                memory_features=memory_features
            )

        # update memory labels
        memory_labels = []
        start_pid = 0
        for idx, dataset in enumerate(self.datasets):
            labels = pseudo_labels[idx]
            memory_labels.append(torch.LongTensor(labels) + start_pid)
            start_pid += max(labels) + 1
        memory_labels = torch.cat(memory_labels).view(-1)

        # if self.epoch<3:
        #     memory_labels = torch.range(0, memory_labels.shape[0]-1)

        # mmcv.dump(memory_labels, 'memory_label.pkl')
        # memory_labels = mmcv.load('memory_label.pkl')

        runner.model.module.roi_head.bbox_head.loss_reid._update_label(memory_labels)
        self.logger.info('pseudo label range: '+ str(memory_labels.min())+ str(memory_labels.max()))

        self.logger.info("Finished updating pseudo label")
        
        #with open(str(self.epoch)+'.txt', 'w') as w:
        #    for i in range(memory_labels.shape[0]):
        #        w.write(str(memory_labels[i])+'\n')
        self.epoch+=1

    
    # def before_run(self, runner):
    #     self.logger.info('start feature extraction for hybrid memory initialization')
    #     features = self.extract_features(
    #         runner.model, self.dataloader, self.dataloader.dataset, with_path=False, prefix="Extract: ",
    #     )
    #     assert features.size(0) == self.dataloader.dataset.id_num

    #     runner.model.module.roi_head.bbox_head.loss_reid._update_feature(features)
