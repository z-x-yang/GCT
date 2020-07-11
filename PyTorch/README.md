## Gated Channel Transformation for Visual Recognition (GCT)
The PyTorch implementation of Gated Channel Transformation for Visual Recognition (CVPR 2020) [[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Gated_Channel_Transformation_for_Visual_Recognition_CVPR_2020_paper.pdf)].

The training code is based on [ImageNet training in PyTorch](https://github.com/pytorch/examples/tree/master/imagenet). The code of the ResNet backbone is from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

## Getting Started
First, please install torch and torchvision.

To train and evaluate GCT-ResNet50, run
```
bash run.sh
```
After finished, the best accuracy should be around 77.2%, which is slightly lower than our TensorFlow version. The reason of the lower performance is that the augmentation method in this simple PyTorch version is a little bit different from the TensorFlow version.

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


