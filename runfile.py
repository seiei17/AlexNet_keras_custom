# runfile of alexnet training and testing

import keras
import keras.backend as K
import tensorflow
from sklearn.metrics import classification_report
from keras.utils import to_categorical

import numpy as np
import os
import pickle

from AlexNetModel import alexnet
from data_generator import cifar10_gen

# initializing parameters
datapath = './datasets/new_cifar10/'

num_classes = 10
epochs = 100

batch_size = 128
learning_rate = 0.01
momentum = 0.9
decay = 0.0005

# get AlexNet model
alexmodel = alexnet((224, 224, 3,), num_classes)

# data processing
generator = cifar10_gen(datapath)

optimizer = keras.optimizers.sgd(lr=0.01, momentum=0.9, decay=0.0005)

alexmodel.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
alexmodel.fit_generator(generator.train_generator(),
                        steps_per_epoch=200,
                        verbose=1,
                        epochs=epochs)

print()
print('---------now predict----------')
print()

y_test = generator.test_data()
y_pred = alexmodel.predict_generator(generator.test_generator(),
                                     steps=20,
                                     verbose=1)

mask = np.argmax(y_pred, axis=1)
pred = np.zeros(y_pred.shape)
for i in range(y_pred.shape[0]):
    pred[i][mask[i]] = 1

# with open('./datasets/cifar10/test_batch', 'rb') as testfile:
#     data = pickle.load(testfile, encoding='latin1')
#     labels = data['labels']
#     labels = to_categorical(labels, num_classes=num_classes)
#     print(classification_report(labels, pred))
print(classification_report(y_test, pred))