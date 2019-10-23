# runfile of alexnet training and testing

import keras
import keras.backend as K
import tensorflow
from sklearn.metrics import classification_report
import numpy as np
import sys

from AlexNetModel import alexnet
from data_generator import cifar_gen
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# initializing parameters
data_name = 'cifar10'
datapath = '../database/cifar10/'
# datapath = '../database/{}_new/'.format(data_name)
num_classes = 10
epochs = 30
batch_size = 128
learning_rate = 0.01
momentum = 0.9
decay = 0.0005
import math
steps = math.ceil(10000 / batch_size)

# get AlexNet model
alexmodel = alexnet((224, 224, 3,), num_classes)

# data processing
# generator = cifar_gen(datapath,
#                       numclasses=num_classes)
generator = cifar_gen(datapath, batch_size=batch_size
                      )
optimizer = keras.optimizers.SGD(lr=0.01,
                                 momentum=0.9)
alexmodel.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

x_val, y_val = generator.valid_data()

alexmodel.fit_generator(generator.train_generator(),
                        steps_per_epoch=10 * steps,
                        verbose=1,
                        epochs=epochs,
                        validation_data=generator.valid_generator(),
                        validation_steps=steps)

alexmodel.save('./model_save/alex_{}_{}.h5'.format(data_name,
                                                   epochs))
# print()
# print('---------now predict----------')
# print()

# y_test = generator.test_data()
# y_pred = alexmodel.predict_generator(generator.test_generator(),
#                                      steps=20,
#                                      verbose=1)
#
# mask = np.argmax(y_pred, axis=1)
# pred = np.zeros(y_pred.shape)
# for i in range(y_pred.shape[0]):
#     pred[i][mask[i]] = 1
#
# print(classification_report(y_test, pred))

# with open('./datasets/cifar10/test_batch', 'rb') as testfile:
#     data = pickle.load(testfile, encoding='latin1')
#     labels = data['labels']
#     labels = to_categorical(labels, num_classes=num_classes)
#     print(classification_report(labels, pred))
