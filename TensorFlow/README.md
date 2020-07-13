## Gated Channel Transformation for Visual Recognition (GCT)
The TensorFlow implementation of Gated Channel Transformation for Visual Recognition (CVPR 2020) [[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Gated_Channel_Transformation_for_Visual_Recognition_CVPR_2020_paper.pdf)].

The code is based on [tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) (TF 1.10). The code can be obtained via:
```
git clone https://github.com/tensorflow/benchmarks
```
Our GCT can be readily applied to [TF 2.0](https://github.com/tensorflow/models/tree/master/official/vision/image_classification) as well, but we take `tf_cnn_benchmarks` as an example here.
We made two changes on `tf_cnn_benchmarks`.


1. We applied [GCT](https://github.com/z-x-yang/GCT/blob/master/TensorFlow/convnet_builder.py#L123) before each convolutional layer. The definition of GCT can be found [here](https://github.com/z-x-yang/GCT/blob/master/TensorFlow/convnet_builder.py#L147-L211).

2. We added a new argument [weight_decay_on_beta](https://github.com/z-x-yang/GCT/blob/master/TensorFlow/benchmark_cnn.py#L271-L272). When `weight_decay_on_beta` is `True`, weight decay (WD) is applied on the gating bias of GCT.

## Getting Started
First, please install TensorFlow 1.10. And then, please follow the [instruction](https://github.com/tensorflow/models/tree/master/research/inception#getting-started) in [tf_cnn_benchmarks](https://github.com/awslabs/deeplearning-benchmark/tree/master/tensorflow_benchmark/tf_cnn_benchmarks) to prepare the Imagenet data in TFRecord format.

To train GCT-ResNet50, run
```
bash train_gct_resnet50.sh
```

To evaluate the last checkpoint, run
```
bash eval_gct_resnet50.sh
```

To evaluate all the checkpoints, run
```
python eval_all_ckpt.py
```
After running the script, the top-1 accuracy should be about 77.6%. If you remove the WD on the gating bias of GCT, the performance should be about 77.3% as the result in our paper. Without GCT, the top-1 accurracy of ResNet50 should be around 76.2%.

To avoid applying weight decay (WD) on the gating bias of GCT as the default setting in our paper, you can set
```
--weight_decay_on_beta=False
```
in the training script. In this version of GCT, we apply WD on the gating bias of GCT, which we found to be better on some backbones (such as ResNet-50).

To train on other backbones, such as ResNet101 or Inception, you can change the model name in the above example scripts. All the names of availbale backbones can be found in [here](https://github.com/z-x-yang/GCT/blob/59bba462bb2b9dd14425333625a2e59d6a5eb57d/models/model_config.py#L33).

## Performance
The accuracy (top-1/top-5 %) should be close to the results below when using 4 GPUs.

| Backbone  | Original | +GCT (no WD on beta) | +GCT (WD on beta) |
| --------- | -------- | ------------------- | ------------------- |
| VGG-16 | 73.8/91.7 | 74.9/**92.5** | **75.0**/92.4 |
| Inception-V3 | 75.7/92.7 | **76.3**/**92.9** | 76.2/**92.9** |
| ResNet-50 | 76.2/93.0 | 77.3/**93.7** | **77.7**/93.6|
| ResNet-101 | 77.8/93.8 | **78.6**/94.1 | 78.5/**94.3** |
| ResNet-152 | 78.4/94.1 | **79.2**/**94.5** | 79.0/94.4 |


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


