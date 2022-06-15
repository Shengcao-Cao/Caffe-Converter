import argparse
import subprocess
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='checkpoints/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth', type=str)
    parser.add_argument('--target', default='checkpoints/converted-fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8-3d47b323.pth', type=str)
    args = parser.parse_args()

    ckpt = torch.load(args.source)
    weight = ckpt['state_dict']['backbone.conv1.weight']
    weight = weight[:, [2, 1, 0], :, :]
    weight[:, 0, :, :] *= 58.395
    weight[:, 1, :, :] *= 57.12
    weight[:, 2, :, :] *= 57.375
    ckpt['state_dict']['backbone.conv1.weight'] = weight

    torch.save(ckpt, args.target)
    sha = subprocess.check_output(['sha256sum', args.target]).decode()
    print(sha)
    print(sha[:8])
