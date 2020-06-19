## Gated Channel Transformation for Visual Recognition (GCT)
The tensorflow implementation of Gated Channel Transformation for Visual Recognition (CVPR 2020).

The code is based on [tf_cnn_benchmarks](https://github.com/awslabs/deeplearning-benchmark/tree/master/tensorflow_benchmark/tf_cnn_benchmarks). For applying GCT, we modify only the code in [convenet_builder.py](https://github.com/z-x-yang/GCT/blob/db5c5d2feef10becc2203517b46160a07c0161f7/convnet_builder.py#L147).

## Getting Started
Please follow the instruction in [tf_cnn_benchmarks](https://github.com/awslabs/deeplearning-benchmark/tree/master/tensorflow_benchmark/tf_cnn_benchmarks) to prepare your environment and Imagenet data.

To run GCT-ResNet50 following the same setting in our paper, run
```
bash train_gct_resnet50.sh
```

To eval, run
```
bash eval_gct_resnet50.sh
```

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


