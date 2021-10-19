import pdb

import torch
import torch.nn as nn
from config import FC_NEURON_LISTS


# Adopeted from official implementation: cognitive_mapping_and_planning-master\tfcode\cmp.py
# def get_map_from_images(imgs, mapper_arch, task_params, freeze_conv, wt_decay,
#                         is_training, batch_norm_is_training_op, num_maps,
#                         split_maps=True):
#   # Hit image with a resnet.
#   n_views = len(task_params.aux_delta_thetas) + 1
#   out = utils.Foo()
#
#   images_reshaped = tf.reshape(imgs,
#       shape=[-1, task_params.img_height,
#              task_params.img_width,
#              task_params.img_channels], name='re_image')
#
#   x, out.vars_to_restore = get_repr_from_image(
#       images_reshaped, task_params.modalities, task_params.data_augment,
#       mapper_arch.encoder, freeze_conv, wt_decay, is_training)
#
#   # Reshape into nice things so that these can be accumulated over time steps
#   # for faster backprop.
#   sh_before = x.get_shape().as_list()
#   out.encoder_output = tf.reshape(x, shape=[task_params.batch_size, -1, n_views] + sh_before[1:])
#   x = tf.reshape(out.encoder_output, shape=[-1] + sh_before[1:])
#
#   # Add a layer to reduce dimensions for a fc layer.
#   if mapper_arch.dim_reduce_neurons > 0:
#     ks = 1; neurons = mapper_arch.dim_reduce_neurons;
#     init_var = np.sqrt(2.0/(ks**2)/neurons)
#     batch_norm_param = mapper_arch.batch_norm_param
#     batch_norm_param['is_training'] = batch_norm_is_training_op
#     out.conv_feat = slim.conv2d(x, neurons, kernel_size=ks, stride=1,
#                     normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_param,
#                     padding='SAME', scope='dim_reduce',
#                     weights_regularizer=slim.l2_regularizer(wt_decay),
#                     weights_initializer=tf.random_normal_initializer(stddev=init_var))
#     reshape_conv_feat = slim.flatten(out.conv_feat)
#     sh = reshape_conv_feat.get_shape().as_list()
#     out.reshape_conv_feat = tf.reshape(reshape_conv_feat, shape=[-1, sh[1]*n_views])
#
#   with tf.variable_scope('fc'):
#     # Fully connected layers to compute the representation in top-view space.
#     fc_batch_norm_param = {'center': True, 'scale': True,
#                            'activation_fn':tf.nn.relu,
#                            'is_training': batch_norm_is_training_op}
#     f = out.reshape_conv_feat
#     out_neurons = (mapper_arch.fc_out_size**2)*mapper_arch.fc_out_neurons
#     neurons = mapper_arch.fc_neurons + [out_neurons]
#     f, _ = tf_utils.fc_network(f, neurons=neurons, wt_decay=wt_decay,
#                                name='fc', offset=0,
#                                batch_norm_param=fc_batch_norm_param,
#                                is_training=is_training,
#                                dropout_ratio=mapper_arch.fc_dropout)
#     f = tf.reshape(f, shape=[-1, mapper_arch.fc_out_size,
#                              mapper_arch.fc_out_size,
#                              mapper_arch.fc_out_neurons], name='re_fc')
#
#   # Use pool5 to predict the free space map via deconv layers.
#   with tf.variable_scope('deconv'):
#     x, outs = deconv(f, batch_norm_is_training_op, wt_decay=wt_decay,
#                      neurons=mapper_arch.deconv_neurons,
#                      strides=mapper_arch.deconv_strides,
#                      layers_per_block=mapper_arch.deconv_layers_per_block,
#                      kernel_size=mapper_arch.deconv_kernel_size,
#                      conv_fn=slim.conv2d_transpose, offset=0, name='deconv')
#
#   # Reshape x the right way.
#   sh = x.get_shape().as_list()
#   x = tf.reshape(x, shape=[task_params.batch_size, -1] + sh[1:])
#   out.deconv_output = x
#
#   # Separate out the map and the confidence predictions, pass the confidence
#   # through a sigmoid.
#   if split_maps:
#     with tf.name_scope('split'):
#       out_all = tf.split(value=x, axis=4, num_or_size_splits=2*num_maps)
#       out.fss_logits = out_all[:num_maps]
#       out.confs_logits = out_all[num_maps:]
#     with tf.name_scope('sigmoid'):
#       out.confs_probs = [tf.nn.sigmoid(x) for x in out.confs_logits]
#   return out



# each fc for each representations ?
# use cnn and then fc ?
# (cognitive_mapping_and_planning-master\tfcode\cmp.py # " Add a layer to reduce dimensions for a fc layer.")
class FC(nn.Module):
    """
    3 layer perceptron class nn.Sequential
    output dimensions:
    """
    def __init__(self):
        super().__init__()
        self.neuron_lists=FC_NEURON_LISTS
        fc_layers = []
        for i in range(3):
            fc_layers.append(nn.Linear(self.neuron_lists[i],self.neuron_lists[i+1]))
            fc_layers.append(nn.BatchNorm1d(self.neuron_lists[i+1]))
            fc_layers.append(nn.ReLU())
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim = 1) # flatten all dimensions except batch
        #pdb.set_trace()
        x = self.fc(x)
        return  x





# def deconv(x, is_training, wt_decay, neurons, strides, layers_per_block,
#             kernel_size, conv_fn, name, offset=0):
#   """Generates a up sampling network with residual connections.
#   """
#   batch_norm_param = {'center': True, 'scale': True,
#                       'activation_fn': tf.nn.relu,
#                       'is_training': is_training}
#   outs = []
#   for i, (neuron, stride) in enumerate(zip(neurons, strides)):
#     for s in range(layers_per_block):
#       scope = '{:s}_{:d}_{:d}'.format(name, i+1+offset,s+1)
#       x = custom_residual_block(x, neuron, kernel_size, stride, scope,
#                                 is_training, wt_decay, use_residual=True,
#                                 residual_stride_conv=True, conv_fn=conv_fn,
#                                 batch_norm_param=batch_norm_param)
#       stride = 1
#     outs.append((x,True))
#   return x, outs

class Deconv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass





