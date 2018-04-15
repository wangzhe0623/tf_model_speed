#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from nets import nets_factory
import glog as logging
import time


def tf_model_speed():
    num_classes = 1000
    weight_decay = 0.0001
    is_training = False
    batch_size = 64
    test_num = 10
    model_names = [
        'inception_v1', 'inception_v2', 'inception_v3', 'inception_v4',
        'inception_resnet_v2',
        'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v1_200',
        'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'resnet_v2_200',
        'vgg_16', 'vgg_19',
        'alexnet_v2',
        'mobilenet_v1', 'mobilenet_v1_075','mobilenet_v1_050','mobilenet_v1_025',
        'nasnet_mobile', 'nasnet_large',
    ]
    for model_name in model_names:
        logging.info(model_name)
        with tf.Graph().as_default():
            model_fn = nets_factory.get_network_fn(
                model_name,
                num_classes=num_classes,
                weight_decay=weight_decay,
                is_training=is_training)
            default_img_size = model_fn.default_image_size
            inputs = tf.random_normal(
                dtype=tf.float32,
                shape=[batch_size, default_img_size, default_img_size, 3])
            logits, _ = model_fn(
                inputs, )
            # for v in tf.trainable_variables():
            #     logging.info(v.op.name)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                sess.run(logits)
                start = time.time()
                for _ in range(test_num):
                    sess.run(logits)
                end = time.time()
                logging.info("%s--default_image_size: %d--time: %f" %
                             (model_name, default_img_size,
                              1. * (end - start) / test_num))


def main(_):
    tf_model_speed()


if __name__ == '__main__':
    tf.app.run()
