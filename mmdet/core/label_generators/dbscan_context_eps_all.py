# Written by Yixiao Ge

import collections

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from .compute_dist import build_dist

#__all__ = ["label_generator_dbscan_context_eps_single", "label_generator_context_eps_dbscan"]

def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray

@torch.no_grad()
def label_generator_dbscan_context_single(cfg, features, dist, eps, **kwargs):
    assert isinstance(dist, np.ndarray)

    # clustering
    min_samples = 4
    use_outliers = True

    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1,)
    labels = cluster.fit_predict(dist)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # cluster labels -> pseudo labels
    # compute cluster centers
    centers = collections.defaultdict(list)
    outliers = 0
    for i, label in enumerate(labels):
        if label == -1:
            if not use_outliers:
                continue
            labels[i] = num_clusters + outliers
            outliers += 1

        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]
    centers = torch.stack(centers, dim=0)
    labels = to_torch(labels).long()
    num_clusters += outliers

    return labels, centers, num_clusters

def list_duplicates(seq):
    tally = collections.defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    dups = [(key,locs) for key,locs in tally.items() if len(locs)>1]
    return dups


@torch.no_grad()
def process_label_with_context(labels, centers, features, inds, num_classes):
    # if the persons in the same image are clustered in the same clust, remove it
    N_p = features.shape[0]
    N_c = centers.shape[0]
    assert num_classes == N_c
    assert N_p == labels.shape[0]
    assert N_p == inds.shape[0]
    unique_inds = set(inds.cpu().numpy())
    #print(unique_inds)
    #print(inds)
    for uid in unique_inds:
        #print("uid", uid)
        b = inds == uid
        tmp_id = b.nonzero()
        #print("tmp_id", tmp_id)
        tmp_labels = labels[tmp_id]
        #print(tmp_labels.squeeze(1), tmp_labels.squeeze(1).shape, list(tmp_labels.squeeze(1).cpu().numpy()))
        dups = list_duplicates(list(tmp_labels.squeeze(1).cpu().numpy()))
        if len(dups) > 0:
            for dup in dups:
                #print(features.shape, centers.shape)
                tmp_center = centers[dup[0]].cpu().numpy()
                #print(tmp_center.shape)
                tmp_features = features[tmp_id[dup[1]].squeeze(1)].cpu().numpy()
                #print(tmp_id[dup[1]].squeeze(1), tmp_features.shape)
                sim = np.dot(tmp_center, tmp_features.transpose())
                #print(sim)
                idx = np.argmax(sim)
                for i in range(len(sim)):
                    if i != idx:
                        labels[tmp_id[dup[1][i]]] = num_classes
                        centers = torch.cat((centers, features[tmp_id[dup[1][i]]]))
                        num_classes += 1
                        #print(centers.shape, num_classes)
    assert num_classes == centers.shape[0]
    return labels, centers, num_classes

@torch.no_grad()
def process_label_with_context_save(labels, centers, features, inds, num_classes, epoch, frac):
    # if the persons in the same image are clustered in the same clust, remove it
    N_p = features.shape[0]
    N_c = centers.shape[0]
    assert num_classes == N_c
    assert N_p == labels.shape[0]
    assert N_p == inds.shape[0]
    unique_inds = set(inds.cpu().numpy())
    #print(unique_inds)
    #print(inds)
    all_dist = []
    for uid in unique_inds:
        #print("uid", uid)
        b = inds == uid
        tmp_id = b.nonzero()
        #print("tmp_id", tmp_id)
        tmp_labels = labels[tmp_id]
        #print(tmp_labels.squeeze(1), tmp_labels.squeeze(1).shape, list(tmp_labels.squeeze(1).cpu().numpy()))
        dups = list_duplicates(list(tmp_labels.squeeze(1).cpu().numpy()))
        if len(dups) > 0:
            for dup in dups:
                #print(features.shape, centers.shape)
                tmp_center = centers[dup[0]].cpu().numpy()
                #print(tmp_center.shape)

                # calculate distance between instances
                inputs_new = features[tmp_id[dup[1]].squeeze(1)]
                n = inputs_new.size(0)
                #print("n", n, inputs_new.shape)
                dist = torch.pow(inputs_new, 2).sum(dim=1, keepdim=True).expand(n, n)
                dist = dist + dist.t()
                dist.addmm_(1, -2, inputs_new, inputs_new.t())
                dist = dist.clamp(min=1e-12).sqrt()
                #print("dist shape", dist.shape)
                all_dist.append(dist.flatten())

                tmp_features = features[tmp_id[dup[1]].squeeze(1)].cpu().numpy()
                #print(tmp_id[dup[1]].squeeze(1), tmp_features.shape)
                sim = np.dot(tmp_center, tmp_features.transpose())
                #print(sim)
                idx = np.argmax(sim)
                for i in range(len(sim)):
                    if i != idx:
                        labels[tmp_id[dup[1][i]]] = num_classes
                        centers = torch.cat((centers, features[tmp_id[dup[1][i]]]))
                        num_classes += 1
                        #print(centers.shape, num_classes)
    
    assert num_classes == centers.shape[0]
    print("len all dist", len(all_dist), 'frac', frac)
    #frac = 0.3
    if len(all_dist) > 0:
        all_dist = torch.cat(all_dist)
        all_dist, _ = torch.sort(all_dist)
        tmp_dist = all_dist[int(len(all_dist)*frac)].item()
        print('tmp_dist', tmp_dist)
        #with open(str(epoch)+'.txt', 'w') as w:
        #    for i in range(all_dist.shape[0]):
        #        w.write(str(all_dist[i])+'\n')
    return labels, centers, num_classes, tmp_dist

@torch.no_grad()
def calculate_dist(features, inds, epoch, frac):
    # if the persons in the same image are clustered in the same clust, remove it
    N_p = features.shape[0]
    assert N_p == inds.shape[0]
    unique_inds = set(inds.cpu().numpy())
    #print(unique_inds)
    #print(inds)
    all_dist = []
    for uid in unique_inds:
        #print("uid", uid)
        b = inds == uid
        tmp_id = b.nonzero()
        #print("tmp_id", tmp_id)
        inputs_new = features[tmp_id.squeeze(1)]
        n = inputs_new.size(0)
        dist = torch.pow(inputs_new, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs_new, inputs_new.t())
        dist = dist.clamp(min=1e-12).sqrt()
        #print("dist shape", dist.shape)
        all_dist.append(dist.flatten())

    print("len all dist", len(all_dist), 'frac', frac)
    #frac = 0.3
    if len(all_dist) > 0:
        all_dist = torch.cat(all_dist)
        all_dist, _ = torch.sort(all_dist)
        tmp_dist = all_dist[int(len(all_dist)*frac)].item()
        print('tmp_dist', tmp_dist)
        
        for tmp_frac in [0.1, 0.12, 0.14, 0.16, 0.18, 0.2]:
            test_dist = all_dist[int(len(all_dist)*tmp_frac)].item()
            print("tmp_frac", tmp_frac, 'test_dist', test_dist)
        '''
        
        with open(str(epoch)+'.txt', 'w') as w:
            for i in range(all_dist.shape[0]):
                w.write(str(all_dist[i])+'\n')
        '''
    return tmp_dist

@torch.no_grad()
def label_generator_dbscan_context_eps_all(cfg, features, cuda=True, indep_thres=None, all_inds=None, epoch=0, eps=None, **kwargs):
    assert cfg.PSEUDO_LABELS.cluster == "dbscan_context_eps_all"

    if not cuda:
        cfg.PSEUDO_LABELS.search_type = 3

    # compute distance matrix by features
    dist = build_dist(cfg.PSEUDO_LABELS, features, verbose=True)

    features = features.cpu()

    # clustering
    
    if eps is not None:
        print("eps", eps)
        eps = [eps-0.02, eps, eps+0.02]
        print("eps", eps)
    else:
        eps = cfg.PSEUDO_LABELS.eps
    
    #eps = cfg.PSEUDO_LABELS.eps
    delta = cfg.PSEUDO_LABELS.delta
    frac = cfg.PSEUDO_LABELS.frac
    frac += delta * epoch

    if len(eps) == 1:
        # normal clustering
        labels, centers, num_classes = label_generator_dbscan_context_single(
            cfg, features, dist, eps[0] + delta * epoch
        )
        if all_inds is not None:
            labels, centers, num_classes = process_label_with_context(labels, centers, features, all_inds, num_classes)
        return labels, centers, num_classes, indep_thres

    else:
        assert (
            len(eps) == 3
        ), "three eps values are required for the clustering reliability criterion"

        print("adopt the reliability criterion for filtering clusters")
        eps = sorted(eps)
        #eps = [ep + delta * epoch for ep in eps]
        print("eps", eps)
        labels_tight, centers_tight, num_classes_tight = label_generator_dbscan_context_single(cfg, features, dist, eps[0])
        labels_normal, centers_normal, num_classes = label_generator_dbscan_context_single(
            cfg, features, dist, eps[1]
        )
        labels_loose, centers_loose, num_classes_loose = label_generator_dbscan_context_single(cfg, features, dist, eps[2])
        #print("num_classes", num_classes_tight, num_classes, num_classes_loose)
        #print(labels_tight.max(), labels_normal.max(), labels_loose.max())

        labels_tight, _, num_classes_tight = process_label_with_context(labels_tight, centers_tight, features, all_inds, num_classes_tight)
        # labels_normal, _, num_classes = process_label_with_context(labels_normal, centers_normal, features, all_inds, num_classes)
        labels_normal, _, num_classes = process_label_with_context(labels_normal, centers_normal, features, all_inds, num_classes)
        tmp_eps = calculate_dist(features, all_inds, epoch, frac)
        labels_loose, _, num_classes_loose = process_label_with_context(labels_loose, centers_loose, features, all_inds, num_classes_loose)
        #print("num_classes", num_classes_tight, num_classes, num_classes_loose)
        #print(labels_tight.max(), labels_normal.max(), labels_loose.max())
        # compute R_indep and R_comp
        N = labels_normal.size(0)
        label_sim = (
            labels_normal.expand(N, N).eq(labels_normal.expand(N, N).t()).float()
        )
        label_sim_tight = (
            labels_tight.expand(N, N).eq(labels_tight.expand(N, N).t()).float()
        )
        label_sim_loose = (
            labels_loose.expand(N, N).eq(labels_loose.expand(N, N).t()).float()
        )

        R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(
            label_sim, label_sim_tight
        ).sum(-1)
        R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(
            label_sim, label_sim_loose
        ).sum(-1)
        assert (R_comp.min() >= 0) and (R_comp.max() <= 1)
        assert (R_indep.min() >= 0) and (R_indep.max() <= 1)

        cluster_R_comp, cluster_R_indep = (
            collections.defaultdict(list),
            collections.defaultdict(list),
        )
        cluster_img_num = collections.defaultdict(int)
        for comp, indep, label in zip(R_comp, R_indep, labels_normal):
            cluster_R_comp[label.item()].append(comp.item())
            cluster_R_indep[label.item()].append(indep.item())
            cluster_img_num[label.item()] += 1

        cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
        cluster_R_indep = [
            min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())
        ]
        cluster_R_indep_noins = [
            iou
            for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys()))
            if cluster_img_num[num] > 1
        ]
        if indep_thres is None:
            indep_thres = np.sort(cluster_R_indep_noins)[
                min(
                    len(cluster_R_indep_noins) - 1,
                    np.round(len(cluster_R_indep_noins) * 0.9).astype("int"),
                )
            ]

        labels_num = collections.defaultdict(int)
        for label in labels_normal:
            labels_num[label.item()] += 1

        centers = collections.defaultdict(list)
        outliers = 0
        #print(cluster_R_indep)
        print(len(cluster_R_indep), num_classes)
        
        for i, label in enumerate(labels_normal):
            label = label.item()
            #print(label)
            indep_score = cluster_R_indep[label]
            comp_score = R_comp[i]
            if label == -1:
                assert not cfg.PSEUDO_LABELS.use_outliers, "exists a bug"
                continue
            if (indep_score > indep_thres) or (
                comp_score.item() > cluster_R_comp[label]
            ):
                if labels_num[label] > 1:
                    labels_normal[i] = num_classes + outliers
                    outliers += 1
                    labels_num[label] -= 1
                    labels_num[labels_normal[i].item()] += 1

            centers[labels_normal[i].item()].append(features[i])

        num_classes += outliers
        assert len(centers.keys()) == num_classes

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]
        centers = torch.stack(centers, dim=0)

        return labels_normal, centers, num_classes, indep_thres, tmp_eps