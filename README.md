# AlexNet_keras_custom
A practice after reading paper, using cifar10 datasets.

# cifar-10
cifar-10共有5个训练集,1个测试集.
通过data_divide_cifar10.py将数据集分割为100个训练集,20个测试集,每个集合500个数据.

原始数据存放在./datasets/cifar10
分割数据存放在./datasets/new_cifar10

由于cifar10数据的每个图片的shape为32x32,
使用np.pad将图片pad为224x224.

# 数据增强
paper中使用了2种数据增强方式,在这里只实现了水平镜像的增强方式,增强RGB通道的方式没有使用.

# 预测
paper中的预测使用了5种随机裁剪的方式,然后对5种预测结果进行求均值的方式.
由于数据集的原因,没有采用裁剪.
