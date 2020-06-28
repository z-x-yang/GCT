## Gated Channel Transformation for Visual Recognition (GCT)
The TensorFlow (1.10) implementation of Gated Channel Transformation for Visual Recognition (CVPR 2020) [[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Gated_Channel_Transformation_for_Visual_Recognition_CVPR_2020_paper.pdf)].

The code is based on [tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks). For applying GCT before each convolutional layer, we modify the code in [convenet_builder.py](https://github.com/z-x-yang/GCT/blob/db5c5d2feef10becc2203517b46160a07c0161f7/convnet_builder.py#L147). To apply weight decay (WD) on the gating bias of GCT (which leads to a better GCT performance on ResNet50 but may decreases the performance on other backbones), you can modify the code in [benchmark_cnn.py](https://github.com/z-x-yang/GCT/blob/68a392e4019cc555c6238c55cfa91f285acf1ca3/benchmark_cnn.py#L2629). In this version of GCT, we avoid applying WD on the gating bias as the default setting in our paper.

## Getting Started
First, please install TensorFlow 1.10. And then, please follow the [instruction](https://github.com/tensorflow/models/tree/master/research/inception#getting-started) in [tf_cnn_benchmarks](https://github.com/awslabs/deeplearning-benchmark/tree/master/tensorflow_benchmark/tf_cnn_benchmarks) to prepare the Imagenet data in TFRecord format.

To train GCT-ResNet50, run
```
bash train_gct_resnet50.sh
```

To eval the last checkpoint, run
```
bash eval_gct_resnet50.sh
```
After evaluation, the top-1 accuracy should be about 77.3%, which is same as the result in our paper. If you apply the WD on the gating bias of GCT, the performance should be better, about 77.5%. Without GCT, the top-1 accurracy of ResNet50 should be around 76.2%.

To train on other backbones, such as ResNet101 or Inception, you can change the model name in the above example scripts. All the names of availbale backbones can be found in [here](https://github.com/z-x-yang/GCT/blob/59bba462bb2b9dd14425333625a2e59d6a5eb57d/models/model_config.py#L33).

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


