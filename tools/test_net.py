"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time
import glob

import torch

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine import run_inference
import utils.logging
import utils.misc as misc_utils

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def get_last_weights(path):
  files = glob.glob(path+'/*.pth')
  if len(files) == 0:
    return None
  files.sort(key=lambda fn: os.path.getmtime(fn) if not os.path.isdir(fn) else 0)
  return files[-1]

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_dir', help='path of checkpoint to load')
    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')
    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron) ^ bool(args.load_dir), \
        'Exactly one of --load_ckpt and --load_detectron and --load_dir should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        ckpt_path = ckpt_path if ckpt_path else args.load_dir
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "coco2017":
        cfg.TRAIN.DATASETS = ('coco_2017_train',)
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "keypoints_coco2017":
        cfg.TRAIN.DATASETS = ('keypoints_coco_2017_train',)
        cfg.TEST.DATASETS = ('keypoints_coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 2
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
        if cfg.TEST.USE_BATCH_AVG:
            assert cfg.TRAIN.DATASETS, \
                'cfg.TRAIN.DATASETS shouldn\'t be empty when cfg.TEST.USE_BATCH_AVG is True'

    assert_and_infer_cfg()

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True


    last_iter = -1
    while True:
        if args.load_dir:
            args.load_ckpt = get_last_weights(args.load_dir)

        if not ( args.load_ckpt or args.load_detectron):
            logger.info('Waiting for \'{}\' to exist...'.format(args.load_dir))
            time.sleep(10)
            continue

        if args.load_ckpt:
            try:
                cur_iter = int(args.load_ckpt.split('_')[-1].split('.')[0][4:])
            except:
                logger.info('Fail to parse current iter. (cur_iter = 0)')
                cur_iter = 0
        
        if args.load_detectron:
            try:
                cur_iter = int(args.load_detectron.split('_')[-1].split('.')[0][4:])
            except:
                logger.info('Fail to parse current iter. (cur_iter = 0)')
                cur_iter = 0
        
        if cur_iter <= last_iter:
            logger.info('Waiting for new weight file')
            time.sleep(10)
            continue
        else:
            last_iter = cur_iter

        tblogger = None
        if args.use_tfboard:
            from tensorboardX import SummaryWriter
            # Set the Tensorboard logger
            tblogger = SummaryWriter(args.output_dir)
    
        all_results = run_inference(
            args,
            ind_range=args.range,
            multi_gpu_testing=args.multi_gpu_testing,
            check_expected_results=True,
            tb_logger=tblogger,
            cur_iter=cur_iter)
    
        if args.use_tfboard:
            for dataset in all_results.keys():
                for task, metrics in all_results[dataset].items():
                    for key in metrics.keys():
                        tblogger.add_scalar(dataset + '/' + task + '/' + key, metrics[key], cur_iter)

        if not args.load_dir or cur_iter >= cfg.SOLVER.MAX_ITER:
            break
    logger.info('Finish testing!')
