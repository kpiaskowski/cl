import tensorflow as tf

from constants import batch_size
from dataprovider import DataProvider
from network import UNet

dataprovider = DataProvider('data', batch_size=batch_size)

# create data handles
handle, train_iter, val_iter, base_img, target_img, target_angle = dataprovider.dataset_handles()
is_training = tf.placeholder(tf.bool)

# define network and get final output
unet = UNet(activation=tf.nn.relu, is_training=is_training)
generated_imgs = unet.network(base_img, target_angle)

loss = tf.losses.mean_squared_error(labels=target_img, predictions=generated_imgs)
global_step = tf.train.create_global_step()

saver = tf.train.Saver()
with tf.Session() as sess:
    t_handle, v_handle = sess.run([train_iter.string_handle(), val_iter.string_handle()])
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, 'PROVIDE PATH TO MODEL')
    cost, step = sess.run([loss, global_step], feed_dict={handle: t_handle, is_training: True})
    print('Training: iteration: {}, loss: {:.5f}'.format(step, cost))