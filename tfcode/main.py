import pdb
import tensorflow as tf
from nav_utils import get_repr_from_image
import numpy as np
from tensorflow.contrib import slim
import tf_utils
import utils
from cmp_utils import deconv
import config_cmp
import cmp as cmp

def get_map_from_images(imgs, mapper_arch, task_params, freeze_conv, wt_decay,
                        is_training, batch_norm_is_training_op, num_maps,
                        split_maps=True):

  # Hit image with a resnet.
  n_views = len(task_params.aux_delta_thetas) + 1
  out = utils.Foo()
  images_reshaped = tf.reshape(imgs,shape=[-1, task_params.img_height,task_params.img_width,task_params.img_channels], name='re_image')

  x, out.vars_to_restore = get_repr_from_image(
      images_reshaped, task_params.modalities, task_params.data_augment,
      mapper_arch.encoder, freeze_conv, wt_decay, is_training)  #(batch,8,8,2048)
  # Reshape into nice things so that these can be accumulated over time steps
  # for faster backprop.
  sh_before = x.get_shape().as_list()
  out.encoder_output = tf.reshape(x, shape=[task_params.batch_size, -1, n_views] + sh_before[1:]) #(batch,8,8,2048)
  x = tf.reshape(out.encoder_output, shape=[-1] + sh_before[1:])
  # Add a layer to reduce dimensions for a fc layer.
  if mapper_arch.dim_reduce_neurons > 0:
    ks = 1; neurons = mapper_arch.dim_reduce_neurons;
    init_var = np.sqrt(2.0/(ks**2)/neurons)
    batch_norm_param = mapper_arch.batch_norm_param
    batch_norm_param['is_training'] = batch_norm_is_training_op
    out.conv_feat = slim.conv2d(x, neurons, kernel_size=ks, stride=1,
                    normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_param,
                    padding='SAME', scope='dim_reduce',
                    weights_regularizer=slim.l2_regularizer(wt_decay),
                    weights_initializer=tf.random_normal_initializer(stddev=init_var))
    reshape_conv_feat = slim.flatten(out.conv_feat)
    sh = reshape_conv_feat.get_shape().as_list()
    out.reshape_conv_feat = tf.reshape(reshape_conv_feat, shape=[-1, sh[1]*n_views])
  with tf.variable_scope('fc'):
    # Fully connected layers to compute the representation in top-view space.
    fc_batch_norm_param = {'center': True, 'scale': True,
                           'activation_fn':tf.nn.relu,
                           'is_training': batch_norm_is_training_op}
    f = out.reshape_conv_feat
    out_neurons = (mapper_arch.fc_out_size**2)*mapper_arch.fc_out_neurons
    neurons = mapper_arch.fc_neurons + [out_neurons]
    f, _ = tf_utils.fc_network(f, neurons=neurons, wt_decay=wt_decay,
                               name='fc', offset=0,
                               batch_norm_param=fc_batch_norm_param,
                               is_training=is_training,
                               dropout_ratio=mapper_arch.fc_dropout)
    f = tf.reshape(f, shape=[-1, mapper_arch.fc_out_size,
                             mapper_arch.fc_out_size,
                             mapper_arch.fc_out_neurons], name='re_fc')
  # Use pool5 to predict the free space map via deconv layers.
  with tf.variable_scope('deconv'):
    x, outs = deconv(f, batch_norm_is_training_op, wt_decay=wt_decay,
                     neurons=mapper_arch.deconv_neurons,
                     strides=mapper_arch.deconv_strides,
                     layers_per_block=mapper_arch.deconv_layers_per_block,
                     kernel_size=mapper_arch.deconv_kernel_size,
                     conv_fn=slim.conv2d_transpose, offset=0, name='deconv')

  # Reshape x the right way.
  sh = x.get_shape().as_list()
  x = tf.reshape(x, shape=[task_params.batch_size, -1] + sh[1:])#?????
  out.deconv_output = x
  # Separate out the map and the confidence predictions, pass the confidence
  # through a sigmoid.
  if split_maps:
    with tf.name_scope('split'):
      out_all = tf.split(value=x, axis=4, num_or_size_splits=2*num_maps)
      out.fss_logits = out_all[:num_maps]
      out.confs_logits = out_all[num_maps:]
    with tf.name_scope('sigmoid'):
      out.confs_probs = [tf.nn.sigmoid(x) for x in out.confs_logits]
  return out


if __name__ == '__main__':
    config_name=  "cmp.lmap_Msc.clip5.sbpd_d_r2r+bench_test"
    configs = config_name.split('.')
    type = configs[0]
    config_name = '.'.join(configs[1:])
    if type == 'cmp':
        args = config_cmp.get_args_for_config(config_name)
        args.setup_to_run = cmp.setup_to_run
        args.setup_train_step_kwargs = cmp.setup_train_step_kwargs

    print("randomly generate some inputs to debug")
    img=np.random.rand(args.navtask.task_params.batch_size,
                       args.navtask.task_params.img_height,
                       args.navtask.task_params.img_width,
                       args.navtask.task_params.img_channels).astype('f')

    print("getting map from image...")
    get_map_from_images(img, args.mapper_arch, args.navtask.task_params, freeze_conv = False, wt_decay=float(0.0001),
                        is_training=False, batch_norm_is_training_op=False, num_maps=len(args.navtask.task_params.map_crop_sizes))

    print("running successfully!")
    # m.vision_ops = get_map_from_images(
    #     m.input_tensors['step']['imgs'], args.mapper_arch,
    #     task_params, args.solver.freeze_conv,
    #     args.solver.wt_decay, is_training, batch_norm_is_training_op,
    #     num_maps=len(task_params.map_crop_sizes))