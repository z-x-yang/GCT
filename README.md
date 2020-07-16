## Gated Channel Transformation for Visual Recognition (GCT)
The TensorFlow (1.10) and PyTorch implementation of Gated Channel Transformation for Visual Recognition (CVPR 2020) [[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Gated_Channel_Transformation_for_Visual_Recognition_CVPR_2020_paper.pdf)].

The TensorFlow implementation supports the backbones of ResNet-50/101/152, VGG-16 and Inception-V3. For PyTorch implementation, we give an example of GCT-ResNet-50. 

## Apply GCT in Your Network
First, we propose to apply GCT before convolutional layers (2D Conv or 3D Conv). Conveniently, you can apply GCT for every Conv layers in your network as the default setting in our paper. But, if you want to save memory, you can reduce the number of GCT modules. In our experiments, if we apply only one GCT for each ResBlock (before the last 1x1 Conv) in ResNet-50, the performance will drop only 0.1~0.2% on ImageNet, compared to full GCT setting.

Second, we propose not to apply weight decay on the gating bias (beta parameter) of GCT. In most situations, applying weight decay on the gating bias will decreasse the performance.

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


