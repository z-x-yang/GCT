## Gated Channel Transformation for Visual Recognition (GCT)
The PyTorch implementation of Gated Channel Transformation for Visual Recognition (CVPR 2020) [[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Gated_Channel_Transformation_for_Visual_Recognition_CVPR_2020_paper.pdf)].

The training code is based on [ImageNet training in PyTorch](https://github.com/pytorch/examples/tree/master/imagenet). The code of the ResNet backbone is from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

## Getting Started
First, please install torch and torchvision.

To train and evaluate GCT-ResNet50, run
```
bash run.sh
```
After finished, the best accuracy should be around 77.2%, which is slightly lower than our TensorFlow version. The reason for the lower performance is that the augmentation method in this simple PyTorch version is a little bit different from the TensorFlow version.

## Object Detection
If you want to use GCT-ResNet in [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), an object detection framework, please fllow the guidance below.

First, you need to replace the backbone file of maskrcnn-benchmark, [resnet.py](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/resnet.py) by our backbone. Notably, the batch normalization in [Line 1](https://github.com/z-x-yang/GCT/blob/dc69cc83513fd04b1960512644693aaa15020b67/PyTorch/resnet.py#L403) and [Line 2] should be replaced by forzen batch normalization as [here](https://github.com/facebookresearch/maskrcnn-benchmark/blob/57eec25b75144d9fb1a6857f32553e1574177daf/maskrcnn_benchmark/modeling/backbone/resnet.py#L397) in maskrcnn-benchmark.

Second, you need to remove the weight decay on the beta parameters of GCT, following the default setting in our paper. In detail, you need to modify the code for applying weight decay of maskrcnn-benchmark, [here](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/solver/build.py). About how to remove the weight decay on the beta, you can refer to our code [here](https://github.com/z-x-yang/GCT/blob/78a0b863d6b5cd28cb417ab6c573c3c3364d8825/PyTorch/main.py#L184).

After running Mask-RCNN & ResNet-50 in its default training schedule ([here](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml)) on COCO, the preformance should be around 39.8 (box AP) and 36.0 (mask AP), while the baseline without GCT should be 37.8 (box AP) and 34.2 (mask AP).

## Pretrain Model
We also prepared a pretrain model of GCT-ResNet50 (top-1 acc: 77.2%), which can be downloaded from [here](https://drive.google.com/file/d/1y5a56UzBjUWlWwlrU42lxueJY_cBpWLL/view?usp=sharing).

## Citation
```
@inproceedings{yang2020gated,
  title={Gated Channel Transformation for Visual Recognition},
  author={Yang, Zongxin and Zhu, Linchao and Wu, Yu and Yang, Yi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11794--11803},
  year={2020}
}
```

