import argparse
import os
import os.path as osp
import warnings

import mmcv
import torch
import torch.nn as nn
import numpy as np
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.apis import multi_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.core import encode_mask_results, tensor2imgs

from tools.person_search.evaluate import evaluate_detections, evaluate_search_nae
from tools.person_search.psdb import PSDB

def single_gpu_test(model,
                    data_loader,
                    query_data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    load_gallery=False,
                    max_iou_with_proposal=False):
    model.eval()
    gboxes = []
    gfeatures = []
    pfeatures = []
    dataset = data_loader.dataset
    if not load_gallery:
        #prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                #print(data)
                result = model(return_loss=False, rescale=True, **data)

            batch_size = len(result)
            #print(batch_size)
            if result[0][0].shape[0]>0:
                gboxes.append(result[0][0][:, :5])
                gfeatures.append(result[0][0][:, 5:])
            else:
                gboxes.append(np.zeros((0, 5), dtype=np.float32))
                gfeatures.append(np.zeros((0, 256), dtype=np.float32))
            #for _ in range(batch_size):
            #    prog_bar.update()
    
    #inference query
    dataset = query_data_loader.dataset
    #prog_bar = mmcv.ProgressBar(len(dataset))
    #set the scores of all proposal to 1
    try:
        if isinstance(model.module.roi_head.bbox_head, nn.ModuleList):
            for i in range(len(model.module.roi_head.bbox_head)):
                model.module.roi_head.bbox_head[i].proposal_score_max=True
        else:
            model.module.roi_head.bbox_head.proposal_score_max=True
    except:
        print('set proposal max fail')
        exit()
        pass
    for i, data in enumerate(query_data_loader):
        if max_iou_with_proposal:
            proposal = data.pop('proposals')[0][0][0].cpu().numpy()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            proposal = data['proposals'][0][0][0].cpu().numpy()
            scales = data['img_metas'][0].data[0][0]['scale_factor']
        if max_iou_with_proposal:
            scales = data['img_metas'][0].data[0][0]['scale_factor']
            proposal = proposal/scales
            # result[0][0] = result[0][0][:, :5]
            # result[0][0] = result[0][0][:1, :5]
            # result[0][0][0, :4] = proposal

        # if isinstance(data['img'][0], torch.Tensor):
        #         img_tensor = data['img'][0]
        # else:
        #     img_tensor = data['img'][0].data[0]
        # img_metas = data['img_metas'][0].data[0]
        # imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        # assert len(imgs) == len(img_metas)

        # for ii, (img, img_meta) in enumerate(zip(imgs, img_metas)):
        #     h, w, _ = img_meta['img_shape']
        #     img_show = img[:h, :w, :]

        #     ori_h, ori_w = img_meta['ori_shape'][:-1]
        #     img_show = mmcv.imresize(img_show, (ori_w, ori_h))


        #     model.module.show_result(
        #         img_show,
        #         result[ii],
        #         show=show,
        #         out_file='img_result/'+ str(i) + '.jpg',
        #         score_thr=0.5)

        if max_iou_with_proposal:
            boxes = result[0][0][:, :4]
            areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
            area = (proposal[2] - proposal[0] + 1) * (proposal[3] - proposal[1] + 1)
            xx1 = np.maximum(proposal[0], boxes[:, 0])
            yy1 = np.maximum(proposal[1], boxes[:, 1])
            xx2 = np.minimum(proposal[2], boxes[:, 2])
            yy2 = np.minimum(proposal[3], boxes[:, 3])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inters = w * h
            IoU = inters / (areas + area - inters)
            iou_i = np.argmax(IoU)
            pfeatures.append(result[0][0][iou_i:iou_i+1, 5:])
        else:
            pfeatures.append(result[0][0][:, 5:])

        # print(proposal)
        # print(boxes[iou_i])
        # print(IoU[iou_i])
        batch_size = len(result)
        #for _ in range(batch_size):
        #    prog_bar.update()

    return gboxes, gfeatures, pfeatures

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--load_results', action="store_true",
        help="Evaluation with pre extracted features. Default: False")
    parser.add_argument('--load_gallery', action="store_true",
        help="Evaluation with pre extracted features of gallery. Default: False")
    parser.add_argument(
        '--reid_threshold',
        type=float,
        default=0.5,
        help='reid threshold (default: 0.5)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    query_dataset = build_dataset(cfg.data.test)
    query_dataset.load_query()
    query_data_loader = build_dataloader(
        query_dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not args.load_results:
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            gboxes, gfeatures, pfeatures = single_gpu_test(model, data_loader, query_data_loader, args.show, args.show_dir,
                                    args.show_score_thr, args.load_gallery)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                    args.gpu_collect)

        if not args.load_gallery:
            mmcv.dump(gboxes,  "gallery_detections.pkl")
            mmcv.dump(gfeatures, "gallery_features.pkl")
        mmcv.dump(pfeatures, "probe_features.pkl")
    else:
        gboxes = mmcv.load("gallery_detections.pkl")
        gfeatures = mmcv.load("gallery_features.pkl")
        pfeatures = mmcv.load("probe_features.pkl")
    
    if args.load_gallery:
        gboxes = mmcv.load("gallery_detections.pkl")
        gfeatures = mmcv.load("gallery_features.pkl")


     # Evaluate
    dataset = PSDB("psdb_test", cfg.data_root)
    evaluate_detections(dataset, gboxes, threshold=args.reid_threshold)
    evaluate_detections(dataset, gboxes, threshold=args.reid_threshold, labeled_only=True)
    evaluate_search_nae(dataset, gboxes, gfeatures, pfeatures, threshold=args.reid_threshold, gallery_size=100)


if __name__ == '__main__':
    main()
