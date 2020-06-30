# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Vgg model configuration.

Includes multiple models: vgg11, vgg16, vgg19, corresponding to
  model A, D, and E in Table 1 of [1].

References:
[1]  Simonyan, Karen, Andrew Zisserman
     Very Deep Convolutional Networks for Large-Scale Image Recognition
     arXiv:1409.1556 (2014)
"""

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from models import model
import datasets




def _construct_vgg(cnn, num_conv_layers):
  """Build vgg architecture from blocks."""
  assert len(num_conv_layers) == 5
  for _ in xrange(num_conv_layers[0]):
    cnn.conv(64, 3, 3, use_batch_norm=True, bias=None)

  cnn.mpool(2, 2)
  for _ in xrange(num_conv_layers[1]):
    cnn.conv(128, 3, 3, use_batch_norm=True, bias=None)

  cnn.mpool(2, 2)
  for _ in xrange(num_conv_layers[2]):
    cnn.conv(256, 3, 3, use_batch_norm=True, bias=None)

  cnn.mpool(2, 2)
  for _ in xrange(num_conv_layers[3]):
    cnn.conv(512, 3, 3, use_batch_norm=True, bias=None)

  cnn.mpool(2, 2)
  for _ in xrange(num_conv_layers[4]):
    cnn.conv(512, 3, 3, use_batch_norm=True, bias=None)

  cnn.mpool(2, 2)
  cnn.reshape([-1, 512 * 7 * 7])
  cnn.affine(4096)
  cnn.dropout()
  cnn.affine(4096)
  cnn.dropout()


class Vgg11Model(model.CNNModel):

  def __init__(self):
    super(Vgg11Model, self).__init__('vgg11', 224, 64, 0.004)

  def add_inference(self, cnn):
    _construct_vgg(cnn, [1, 1, 2, 2, 2])


class Vgg16Model(model.CNNModel):

  def __init__(self):
    super(Vgg16Model, self).__init__('vgg16', 224, 256, 0.025)

  def add_inference(self, cnn):
    cnn.use_batch_norm = True
    cnn.batch_norm_config = {'decay': 0.9, 'epsilon': 1e-5, 'scale': True}
    _construct_vgg(cnn, [2, 2, 3, 3, 3])

  def get_learning_rate(self, global_step, batch_size):
    num_batches_per_epoch = (
        float(datasets.IMAGENET_NUM_TRAIN_IMAGES) / batch_size)
    boundaries = [int(num_batches_per_epoch * x) for x in [30, 60, 90, 100]]

    rescaled_lr = self.learning_rate / self.default_batch_size * batch_size
    print('Init LR: ', rescaled_lr)
    print('Batch size: ', batch_size)
    values = [1, 0.1, 0.01, 0.001, 0.0001]
    values = [rescaled_lr * v for v in values]
    lr = tf.train.piecewise_constant(global_step, boundaries, values)

    warmup_steps = int(num_batches_per_epoch)

    warmup_lr = lr * 0.1

    return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)


class Vgg19Model(model.CNNModel):

  def __init__(self):
    super(Vgg19Model, self).__init__('vgg19', 224, 64, 0.004)

  def add_inference(self, cnn):
    _construct_vgg(cnn, [2, 2, 4, 4, 4])
