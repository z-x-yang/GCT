# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Mobilenet V2 model, branched from slim models for fp16 performance study.

Architecture: https://arxiv.org/abs/1801.04381

The base model gives 72.2% accuracy on ImageNet, with 300MMadds,
3.4 M parameters.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf

from models import mobilenet as lib
from models import mobilenet_conv_blocks as ops
from models import model

slim = tf.contrib.slim
op = lib.op

expand_input = ops.expand_input_by_factor


def get_variable(name, shape, dtype, cast_dtype, *args, **kwargs):
  # TODO(reedwm): Currently variables and gradients are transferred to other
  # devices and machines as type `dtype`, not `cast_dtype`. In particular,
  # this means in fp16 mode, variables are transferred as fp32 values, not
  # fp16 values, which uses extra bandwidth.
  var = tf.get_variable(name, shape, dtype, *args, **kwargs)
  return tf.cast(var, cast_dtype)


def adaption(input_layer):
  epsilon = 1e-5
  variable_dtype = tf.float32
  dtype = tf.float16

  num_channels = input_layer.get_shape().as_list()[3]
  squeeze = [1, 2]
  with tf.variable_scope(default_name='adaption'):
    beta = get_variable('beta', [1, 1, 1, num_channels],
                        variable_dtype, dtype,
                        initializer=tf.constant_initializer(0.))
    alpha = get_variable('alpha', [1, 1, 1, num_channels],
                         variable_dtype, dtype,
                         initializer=tf.constant_initializer(1.))
    gamma = get_variable('gamma', [1, 1, 1, num_channels],
                         variable_dtype, dtype,
                         initializer=tf.constant_initializer(1.))
    theta = get_variable('theta', [1, 1, 1, num_channels],
                         variable_dtype, dtype,
                         initializer=tf.constant_initializer(0.))

    X = input_layer
    alpha_2 = tf.square(alpha)
    alpha_2 = alpha_2 / tf.reduce_mean(alpha_2) + epsilon
    alpha = tf.sqrt(alpha_2)
    A = alpha_2 * tf.reduce_mean(tf.square(X), squeeze, keepdims=True) - (
        2. * alpha * beta) * tf.reduce_mean(X, squeeze, keepdims=True)
    A = tf.reduce_mean(
        A, [1, 2, 3], keepdims=True) + (tf.reduce_mean(tf.square(beta)) + epsilon)
    # B = tf.reduce_sum(alpha_2)
    B = 1.
    l2 = tf.sqrt(B / A)
    adaptor = tf.pow(l2, gamma + theta * l2)
    trans_back = X * adaptor + (beta / alpha) * (1. - adaptor)

  return trans_back


def adaption_conv2d(inputs, *args, **kwargs):
  name = 'adaption'
  inputs = adaption(inputs)
  return slim.conv2d(inputs, *args, **kwargs)


# pyformat: disable
# Architecture: https://arxiv.org/abs/1801.04381
V2_DEF = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (ops.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(adaption_conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(ops.expanded_conv, stride=2, num_outputs=24),
        op(ops.expanded_conv, stride=1, num_outputs=24),
        op(ops.expanded_conv, stride=2, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=2, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=2, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=320),
        op(adaption_conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
    ],
)
# pyformat: enable


@slim.add_arg_scope
def mobilenet(input_tensor,
              num_classes=1001,
              depth_multiplier=1.0,
              scope='MobilenetV2',
              conv_defs=None,
              finegrain_classification_mode=False,
              min_depth=None,
              divisible_by=None,
              **kwargs):
  """Creates mobilenet V2 network.

  Inference mode is created by default. To create training use training_scope
  below.

  with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
     logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

  Args:
    input_tensor: The input tensor
    num_classes: number of classes
    depth_multiplier: The multiplier applied to scale number of
    channels in each layer. Note: this is called depth multiplier in the
    paper but the name is kept for consistency with slim's model builder.
    scope: Scope of the operator
    conv_defs: Allows to override default conv def.
    finegrain_classification_mode: When set to True, the model
    will keep the last layer large even for small multipliers. Following
    https://arxiv.org/abs/1801.04381
    suggests that it improves performance for ImageNet-type of problems.
      *Note* ignored if final_endpoint makes the builder exit earlier.
    min_depth: If provided, will ensure that all layers will have that
    many channels after application of depth multiplier.
    divisible_by: If provided will ensure that all layers # channels
    will be divisible by this number.
    **kwargs: passed directly to mobilenet.mobilenet:
      prediction_fn- what prediction function to use.
      reuse-: whether to reuse variables (if reuse set to true, scope
      must be given).
  Returns:
    logits/endpoints pair

  Raises:
    ValueError: On invalid arguments
  """
  if conv_defs is None:
    conv_defs = V2_DEF
  if 'multiplier' in kwargs:
    raise ValueError('mobilenetv2 doesn\'t support generic '
                     'multiplier parameter use "depth_multiplier" instead.')
  if finegrain_classification_mode:
    conv_defs = copy.deepcopy(conv_defs)
    if depth_multiplier < 1:
      conv_defs['spec'][-1].params['num_outputs'] /= depth_multiplier

  depth_args = {}
  # NB: do not set depth_args unless they are provided to avoid overriding
  # whatever default depth_multiplier might have thanks to arg_scope.
  if min_depth is not None:
    depth_args['min_depth'] = min_depth
  if divisible_by is not None:
    depth_args['divisible_by'] = divisible_by

  with slim.arg_scope((lib.depth_multiplier,), **depth_args):
    return lib.mobilenet(
        input_tensor,
        num_classes=num_classes,
        conv_defs=conv_defs,
        scope=scope,
        multiplier=depth_multiplier,
        **kwargs)


@slim.add_arg_scope
def mobilenet_base(input_tensor, depth_multiplier=1.0, **kwargs):
  """Creates base of the mobilenet (no pooling and no logits) ."""
  return mobilenet(input_tensor,
                   depth_multiplier=depth_multiplier,
                   base_only=True, **kwargs)


def training_scope(**kwargs):
  """Defines MobilenetV2 training scope.

  Usage:
     with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
       logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

  with slim.

  Args:
    **kwargs: Passed to mobilenet.training_scope. The following parameters
    are supported:
      weight_decay- The weight decay to use for regularizing the model.
      stddev-  Standard deviation for initialization, if negative uses xavier.
      dropout_keep_prob- dropout keep probability
      bn_decay- decay for the batch norm moving averages.

  Returns:
    An `arg_scope` to use for the mobilenet v2 model.
  """
  return lib.training_scope(**kwargs)


class MobilenetModel(model.CNNModel):
  """Mobilenet model configuration."""

  def __init__(self):
    super(MobilenetModel, self).__init__('mobilenet', 224, 32, 0.005)

  def add_inference(self, cnn):
    with tf.contrib.slim.arg_scope(training_scope(is_training=cnn.phase_train)):
      cnn.top_layer, _ = mobilenet(cnn.top_layer, is_training=cnn.phase_train)
      cnn.top_size = cnn.top_layer.shape[-1].value
