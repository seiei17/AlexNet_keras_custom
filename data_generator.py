# data preprocessing files


import numpy as np
import pickle
import os
import cv2
from keras.utils import to_categorical


class cifar10_gen(object):
    # using cifar 10 dataset
    # have 5 batch, which form is 'data_batch_i'
    # batches.meta consisting of class name

    def __init__(self, path):
        self.path = path
        self.num = 0

    def load_file(self, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        return data

    def train_generator(self):
        while True:
            for i in range(100):
                # normal
                filepath = os.path.join(self.path, 'data_batch_{}'.format(i))

                data = self.load_file(filepath)     # here data is:
                                                    # dict_keys(['batch_label', 'labels', 'data', 'filenames'])
                labels = data['labels']
                images = data['images']
                # images_ref = images[:, ::-1]

                images = images.reshape(len(labels), 32, 32, 3)
                images = np.divide(images, 255)

                images_ref = images[:, :, :, ::-1]

                new_images = np.zeros((images.shape[0], 224, 224, 3))
                new_images_ref = np.zeros((images.shape[0], 224, 224, 3))

                # images = np.pad(images, ((0, 0), (96, 96), (96, 96), (0, 0)),
                #                 mode='constant', constant_values=0)
                for j in range(500):
                    new_images[j] = cv2.resize(images[j], (224, 224))
                labels = to_categorical(labels, 10)

                # images_ref = images_ref.reshape(len(labels), 32, 32, 3)
                # images_ref = np.divide(images_ref, 255)
                # images_ref = np.pad(images_ref, ((0, 0), (96, 96), (96, 96), (0, 0)),
                #                     mode='constant', constant_values=0)
                labels_ref = labels

                yield new_images, labels

                # reflection
                for j in range(500):
                    new_images_ref[j] = cv2.resize(images_ref[j], (224, 224))
                yield new_images_ref, labels_ref

    def test_data(self):
        filepath = os.path.join('./datasets/cifar10/', 'test_batch')
        data = self.load_file(filepath)
        labels = data['labels']
        # images = data['data'].reshape(len(labels), 32, 32, 3)
        # images /= 255.
        # images = np.pad(images, ((0, 0), (96, 96), (96, 96), (0, 0)), mode='constant', constant_values=0)
        labels = to_categorical(labels, 10)
        return labels

    def test_generator(self):
        while True:
            for i in range(20):
                # normal
                filepath = os.path.join(self.path, 'test_batch_{}'.format(i))
                data = self.load_file(filepath)  # here data is: dict_keys(['batch_label', 'labels', 'data', 'filenames'])
                labels = data['labels']
                images = data['images']
                images_ref = images[:, ::-1]

                new_images = np.zeros((images.shape[0], 224, 224, 3))

                images = images.reshape(len(labels), 32, 32, 3)
                images = np.divide(images, 255)
                for j in range(500):
                    new_images[j] = cv2.resize(images[j], (224, 224))
                # images = np.pad(images, ((0, 0), (96, 96), (96, 96), (0, 0)), mode='constant', constant_values=0)
                # labels = to_categorical(labels, 10)

                # images_ref = images_ref.reshape(len(labels), 32, 32, 3)
                # images_ref = np.divide(images_ref, 255)
                # images_ref = np.pad(images_ref, ((0, 0), (96, 96), (96, 96), (0, 0)), mode='constant',
                #                     constant_values=0)
                # labels_ref = labels

                # yield images, labels
                yield new_images

                # reflection
                # yield images_ref, labels_ref