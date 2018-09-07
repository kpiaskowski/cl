import os
import random
import time

import tensorflow as tf
from clusterone import get_data_path, get_logs_path

from constants import log_ckpt, batch_size, save_ckpt
from dataprovider import DataProvider
from network import UNet

PATH_TO_LOCAL_LOGS = os.path.expanduser('~/Documents/logs')
ROOT_PATH_TO_LOCAL_DATA = os.path.expanduser('~/Documents/data/')
CHIEF_INDEX = 0

# Configure  distributed task
try:
    job_name = os.environ['JOB_NAME']
    task_index = os.environ['TASK_INDEX']
    ps_hosts = os.environ['PS_HOSTS']
    worker_hosts = os.environ['WORKER_HOSTS']
except:
    job_name = None
    task_index = 0
    ps_hosts = None
    worker_hosts = None

flags = tf.app.flags

# Flags for configuring the distributed task
flags.DEFINE_string("job_name", job_name,
                    "job name: worker or ps")
flags.DEFINE_integer("task_index", task_index,
                     "worker task index, should be >= 0 (0 for chief)")
flags.DEFINE_string("ps_hosts", ps_hosts,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", worker_hosts,
                    "Comma-separated list of hostname:port pairs")

# Training related flags
# todo ask why datasets wont load in Matrix
flags.DEFINE_string("data_dir",
                    get_data_path(
                        dataset_name="kpiaskowski/test_dataset",  # all mounted repo
                        local_root=ROOT_PATH_TO_LOCAL_DATA,
                        local_repo="test_dataset",
                        path='data'
                    ),
                    "Path to store logs and checkpoints")
flags.DEFINE_string("log_dir",
                    get_logs_path(root=PATH_TO_LOCAL_LOGS),
                    "Path to dataset")
FLAGS = flags.FLAGS


def device_and_target():
    # for single machine
    if FLAGS.job_name is None:
        print("Running single-machine training")
        return (None, "")

    # for distributed TensorFlow
    print("Running distributed training")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
        raise ValueError("Must specify an explicit `ps_hosts`")
    if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
        raise ValueError("Must specify an explicit `worker_hosts`")

    cluster_spec = tf.train.ClusterSpec({
        "ps": FLAGS.ps_hosts.split(","),
        "worker": FLAGS.worker_hosts.split(","),
    })
    server = tf.train.Server(
        cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()

    worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    return (
        tf.train.replica_device_setter(
            worker_device=worker_device,
            cluster=cluster_spec),
        server.target,
    )


def train(learning_rate):
    tf.reset_default_graph()
    device, target = device_and_target()  # getting node environment

    # define model
    with tf.device(device):
        dataprovider = DataProvider('data', batch_size=batch_size)  # todo change temp data folder when know why datasets wont load in Matrix

        # create data handles
        handle, train_iter, val_iter, base_img, target_img, target_angle = dataprovider.dataset_handles()
        is_training = tf.placeholder(tf.bool)

        # define network and get final output
        unet = UNet(activation=tf.nn.relu, is_training=is_training)
        generated_imgs = unet.network(base_img, target_angle)

        loss = tf.losses.mean_squared_error(labels=target_img, predictions=generated_imgs)
        global_step = tf.train.create_global_step()

        # concatenated base, generated and target img
        concat_img = tf.concat([base_img, generated_imgs, target_img], 2)

        # separate summaries for scalars and imgs
        loss_summary = tf.summary.scalar("loss", loss)
        img_summary = tf.summary.image('images', concat_img)
        loss_merged = tf.summary.merge([loss_summary])
        img_merged = tf.summary.merge([img_summary])

        # use batchnorm and gradient clipping
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gvs = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

        dirname = time.strftime("%Y_%m_%d_%H:%M")
        if FLAGS.task_index == CHIEF_INDEX:
            train_writer = tf.summary.FileWriter(os.path.join(FLAGS.FLAGS.data_dir, 'tblogs', dirname + '_train'))
            val_writer = tf.summary.FileWriter(os.path.join(FLAGS.FLAGS.data_dir, 'tblogs', dirname + '_val'))

    stop_hook = tf.train.StopAtStepHook(last_step=1000000)
    saver_hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=os.path.join(FLAGS.FLAGS.data_dir, 'saved_models', dirname),
        save_secs=None,
        save_steps=save_ckpt,
        saver=tf.train.Saver(max_to_keep=3),
        checkpoint_basename='model.ckpt',
        scaffold=None)
    hooks = [stop_hook, saver_hook]

    """I tried to find as easy an elegant solution, how to restore model from checkpoint when using ALSO feedable iterators for dataset, but to no avail.
    The problem is that MonitoredTrainingSesssion seems to have problems with initializing those iterators after restore. The solutions I found are rather not elegant workarounds,
    therefore I decided to leave it without restoring (I decided that flexibility of feedable iterators is more important for this exemplary task. I wrote however inference.py script
    to check wheter saved models work fine - they do."""
    with tf.train.MonitoredTrainingSession(
            master=target,
            is_chief=(FLAGS.task_index == CHIEF_INDEX),
            checkpoint_dir=None,
            hooks=hooks) as sess:

        # initialize dataset handles
        t_handle, v_handle = sess.run([train_iter.string_handle(), val_iter.string_handle()])
        if FLAGS.task_index == CHIEF_INDEX:
            train_writer.add_graph(sess.graph)

        while not sess.should_stop():
            cost, _, step, summ = sess.run([loss, train_op, global_step, loss_merged], feed_dict={handle: t_handle, is_training: True})
            print('Training: iteration: {}, loss: {:.5f}'.format(step, cost))  # flush used only to make sure outputs in Matrix are most current and fresh

            # write train logs evey iteration
            if FLAGS.task_index == CHIEF_INDEX:
                train_writer.add_summary(summ, step)

            # every log_ckpt steps, log heavier data, like images (and also from validation logs)
            if step % log_ckpt == 0:
                # get imgs and loss on validation set
                cost, step, loss_summ_val, img_summ_val = sess.run([loss, global_step, loss_merged, img_merged],
                                                                   feed_dict={handle: v_handle, is_training: False})
                print('Validation: iteration: {}, loss: {:.5f}'.format(step, cost))

                # get only images from training set
                step, img_summ_train = sess.run([global_step, img_merged], feed_dict={handle: v_handle, is_training: False})

                # dump logs
                if FLAGS.task_index == CHIEF_INDEX:
                    val_writer.add_summary(loss_summ_val, step)
                    val_writer.add_summary(img_summ_val, step)
                    train_writer.add_summary(img_summ_train, step)


def main(unused_argv):
    if FLAGS.log_dir is None or FLAGS.log_dir == "":
        raise ValueError("Must specify an explicit `log_dir`")
    if FLAGS.data_dir is None or FLAGS.data_dir == "":
        raise ValueError("Must specify an explicit `data_dir`")

    train(learning_rate=0.0001)


if __name__ == '__main__':
    tf.app.run()
