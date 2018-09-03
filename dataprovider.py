import os

import numpy as np
import tensorflow as tf

from constants import pi_on_180


class DataProvider:
    """
        Provides data for the neural autoencoder. Uses DatasetAPI.
    """

    def __init__(self, root_path, batch_size):
        """
        :param root_path: Path to the root directory with data. It should contain two subdirs: 'train' and 'val'. Both of them should contain subsequent folders with images inside.
        :param __batch_size: Size of single batch
        """
        self.__batch_size = batch_size
        self.__train_dirs = [os.path.join(root_path, 'train', name) for name in os.listdir(os.path.join(root_path, 'train'))]
        self.__val_dirs = [os.path.join(root_path, 'val', name) for name in os.listdir(os.path.join(root_path, 'val'))]

        self.__n_imgs = 2

    def __create_dataset(self, paths):
        """
        Creates Dataset objects (from TF DatasetAPI).
        :param paths: list of paths to innermost directories
        :return: dataset object
        """
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(paths)))
        dataset = dataset.shuffle(20000)
        dataset = dataset.flat_map(lambda dir: tf.data.Dataset.list_files(dir + '/*.png').
                                   shuffle(30).
                                   map(self.__load_images, num_parallel_calls=8).
                                   map(lambda img, filename: tuple(tf.py_func(self.__decode_name, [img, filename], [tf.float32, tf.float32], stateful=False)),
                                       num_parallel_calls=8).
                                   apply(tf.contrib.data.batch_and_drop_remainder(self.__n_imgs)))
        dataset = dataset.prefetch(self.__batch_size)
        dataset = dataset.shuffle(150)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.__batch_size))
        dataset = dataset.repeat()
        return dataset

    def dataset_handles(self):
        """
        Creates training and validation datasets. Takes care of proper shape setting, splitting images and computing absolute angle. Returns handles to data.
        It uses feedable iterators to allow interleaving train/val data during training.
        :return: 
        """
        _train_dataset = self.__create_dataset(self.__train_dirs)
        __val_dataset = self.__create_dataset(self.__val_dirs)

        handle = tf.placeholder(tf.string, shape=[])
        iter = tf.data.Iterator.from_string_handle(handle, _train_dataset.output_types, _train_dataset.output_shapes)
        image_pair, angle_pair = iter.get_next()

        # reshaping hacks needed, because using tf.py_func cause shape info to vanish
        base_img, target_img = self.split_imgs(image_pair)
        base_img = tf.reshape(base_img, (-1, 128, 128, 3))
        target_img = tf.reshape(target_img, (-1, 128, 128, 3))

        # The same reshaping trick applies to angles
        target_angle = angle_pair[:, -1, :]
        target_angle = self.deg_to_cos(target_angle)
        target_angle = tf.reshape(target_angle, (-1, 2))

        train_iter = _train_dataset.make_one_shot_iterator()
        val_iter = __val_dataset.make_one_shot_iterator()

        return handle, train_iter, val_iter, base_img, target_img, target_angle

    def __load_images(self, filename):
        """Loads and scales image to range <0, 1>. Used within DatasetAPI"""
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string)
        image_scaled = image_decoded / 255
        return image_scaled, filename

    def __decode_name(self, img, filename):
        """Decodes image name and specifies the absolute angle associated to it. Used within DatasetAPI"""
        decoded = filename.decode()
        angles = np.float32(decoded.split('/')[-1].split('_')[:2])
        return img, angles

    @staticmethod
    def deg2rad(deg):
        return deg * pi_on_180

    @staticmethod
    def rad2deg(rad):
        return rad / pi_on_180

    @staticmethod
    def split_imgs(imgs_placeholder):
        """Splits images into base one and target one"""
        base_imgs = imgs_placeholder[:, 0, :, :, :]
        target_imgs = imgs_placeholder[:, -1, :, :, :]
        return base_imgs, target_imgs

    @staticmethod
    def deg_to_cos(angles):
        """Converts angles from degrees to radians and then to cosine values"""
        rad_ang = DataProvider.deg2rad(angles)
        cos_angle = tf.cos(rad_ang)
        return cos_angle
