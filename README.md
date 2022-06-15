# Caffe Converter

Just a simple converter that transforms MMDetection model weights from Caffe-style to PyTorch-style. I write this because I need to unify the image pre-processing pipeline in MMDetection.

## Usage

`python convert.py --source ${CAFFE_WEIGHTS} --target ${PYTORCH_WEIGHTS}`

Make sure PyTorch and MMDetection are installed. Check `test.sh` to verify this conversion is valid.

## Differences between Caffe-style and PyTorch-style

Some model configs/weights are marked with a tag like `caffe`. The differences between Caffe-style and PyTorch-style models are (comparing `tool/caffe_config.py` and `tool/pytorch_config.py`):

- They assume image loading and normalization differently. Caffe-style uses BGR format and `mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0]`, while PyTorch-style uses RGB and `mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]`. Therefore, we can modify the first convolutional layer in ResNet to adapt to this change. That's exactly the job of `convert.py`.
- In the downsampling blocks of ResNet, Caffe-style places the stride at the first 1x1 convolution (same as the original paper and implementation), while PyTorch-style places the stride at the 3x3 convolution (with slightly better performance). We cannot modify the weights to adapt to this change, but should keep the corresponding configuration. More details:
    - Torchvision: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    - MMDetection (`style`): https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py
    - Detectron2 (`stride_in_1x1`): https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/resnet.py
- PyTorch-style allows learnable affine transformation parameters (gamma and beta) in BatchNorm. Again, we cannot modify the weights to adapt to this change.