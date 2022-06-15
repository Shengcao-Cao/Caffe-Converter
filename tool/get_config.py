import argparse
import os

from mmcv import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='../mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco.py', type=str)
    parser.add_argument('--target', default='tool/caffe_config.py', type=str)
    args = parser.parse_args()

    config = Config.fromfile(args.source)
    config.dump(args.target)
