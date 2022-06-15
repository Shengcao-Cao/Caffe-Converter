mkdir checkpoints
cd checkpoints
wget https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth
cd ..

python convert.py

python test/test.py test/fcos_config.py \
    checkpoints/converted-fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8-3d47b323.pth \
    --work-dir test --eval bbox