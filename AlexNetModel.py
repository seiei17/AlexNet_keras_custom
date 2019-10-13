# AlexNet Architecture


from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Input

from keras.models import Model


def alexnet(input_shape, num_classes=1000):
    '''

    :param input_shape: (, 224, 224, 3)
    :param num_classes: default = 1000
    :return: AlexNet model

    using BatchNormalization to replace Local Response Normalization,
    as to paper, dropout rate = 0.5.

    '''


    tensor = Input(input_shape)
    net = {}
    net['input'] = tensor

    # layer1
    net['c1'] = Convolution2D(96, (11, 11), strides=4,
                              activation='relu',
                              name='c1')(net['input'])
    net['rn1'] = BatchNormalization(name='rn1')(net['c1'])
    net['s1'] = MaxPool2D((3, 3), strides=2,
                          name='s1')(net['rn1'])

    # layer2
    net['c2'] = Convolution2D(256, (5, 5), padding='same',
                              activation='relu',
                              name='c2')(net['s1'])
    net['rn2'] = BatchNormalization(name='rn2')(net['c2'])
    net['s2'] = MaxPool2D((3, 3), strides=2,
                          name='s2')(net['rn2'])

    # layer3
    net['c3'] = Convolution2D(384, (3, 3), padding='same',
                              activation='relu',
                              name='c3')(net['s2'])

    # layer4
    net['c4'] = Convolution2D(384, (3, 3), padding='same',
                              activation='relu',
                              name='c4')(net['c3'])

    # layer5
    net['c5'] = Convolution2D(256, (3, 3), padding='same',
                              activation='relu',
                              name='c5')(net['c4'])
    net['s5'] = MaxPool2D((3, 3), strides=2,
                          name='s5')(net['c5'])
    net['flat'] = Flatten(name='flat')(net['s5'])

    # layer6
    net['fc6'] = Dense(4096, activation='relu',
                       name='fc6')(net['flat'])
    net['d6'] = Dropout(rate=0.5, name='d6')(net['fc6'])

    # layer7
    net['fc7'] = Dense(4096, activation='relu',
                       name='fc7')(net['d6'])
    net['d7'] = Dropout(rate=0.5, name='d7')(net['fc7'])

    # layer8
    net['output'] = Dense(num_classes, activation='softmax',
                          name='output')(net['d7'])

    model = Model(net['input'], net['output'])
    return model