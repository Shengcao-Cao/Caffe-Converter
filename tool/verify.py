import copy
import torch

from mmcv import Config
from mmdet.models import build_detector

# Get config and build a Caffe-style model
cfg = Config.fromfile('tool/caffe_config.py')
model = build_detector(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))
model.init_weights()
model.eval()
resnet = model.backbone
conv = model.backbone.conv1

# Assume in RGB
image = torch.rand(1, 3, 224, 224) * 255

# Caffe-style computation
image_caffe = image.clone()[:, [2, 1, 0], :, :]
image_caffe[:, 0] = (image_caffe[:, 0] - 103.53) / 1.0
image_caffe[:, 1] = (image_caffe[:, 1] - 116.28) / 1.0
image_caffe[:, 2] = (image_caffe[:, 2] - 123.675) / 1.0

feature_caffe = conv(image_caffe)
final_caffe = resnet(image_caffe)[-1]
print(feature_caffe[0, :3, :2, :2])
print(final_caffe[0, :3, :2, :2])

# Convert: only need to change conv1 in resnet
conv = model.backbone.conv1
conv.weight.data = conv.weight.data[:, [2, 1, 0], :, :]
conv.weight.data[:, 0, :, :] *= 58.395
conv.weight.data[:, 1, :, :] *= 57.12
conv.weight.data[:, 2, :, :] *= 57.375

# PyTorch-style computation
image_pytorch = image.clone()
image_pytorch[:, 0] = (image_pytorch[:, 0] - 123.675) / 58.395
image_pytorch[:, 1] = (image_pytorch[:, 1] - 116.28) / 57.12
image_pytorch[:, 2] = (image_pytorch[:, 2] - 103.53) / 57.375

feature_pytorch = conv(image_pytorch)
final_pytorch = resnet(image_pytorch)[-1]
print(feature_pytorch[0, :3, :2, :2])
print(final_pytorch[0, :3, :2, :2])

# Compare difference
print(torch.norm(feature_pytorch - feature_caffe))
print(torch.norm(final_pytorch - final_caffe))