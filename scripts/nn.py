#加载keras模块
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig("pic.png")
        plt.show()
	
#变量初始化
batch_size = 128 
nb_classes = 2
nb_epoch = 8 

# the data, shuffled and split between train and test sets
def load():
	x = np.loadtxt("./x_train_new",delimiter ="\t")
	y = np.loadtxt("./y_train", skiprows=1, usecols=1)
	print(x.shape, y.shape)
	return x,y

# ##
## load the train and test data, the radio is 9:1
def get_train_test_data(x, y):
	x_train_true= x[0:45000,]
	x_train_false=x[50000:95000,]
	x_train = np.concatenate((x_train_true, x_train_false))

	y_train_true= y[0:45000,]
	y_train_false= y[50000:95000,]
	y_train = np.concatenate((y_train_true, y_train_false))

	x_test_true= x[45000:50000,]
	x_test_false= x[95000:100000,]
	x_test = np.concatenate((x_test_true, x_test_false))

	y_test_true= y[45000:50000,]
	y_test_false= y[95000:100000,]
	y_test = np.concatenate((y_test_true,y_test_false))

	print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
	return x_train, y_train, x_test, y_test


def train(X_train,Y_train,X_test,Y_test):	
        # convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(Y_train, nb_classes)
	Y_test = np_utils.to_categorical(Y_test, nb_classes)
	#建立模型 使用Sequential（）
	model = Sequential()
	model.add(Dense(512, input_shape=(71,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(2))
	model.add(Activation('softmax'))

	#打印模型
	model.summary()

	#训练与评估
	#编译模型
	model.compile(loss='binary_crossentropy',
				  optimizer=RMSprop(),
				  metrics=['accuracy'])
	#创建一个实例history
	history = LossHistory()

	#迭代训练（注意这个地方要加入callbacks）
	model.fit(X_train, Y_train,
				batch_size=batch_size, nb_epoch=nb_epoch,
				verbose=1, 
				validation_data=(X_test, Y_test),
				callbacks=[history])

	#模型评估
	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	#绘制acc-loss曲线
	history.loss_plot('epoch')
def main():
	x,y= load()
	x_train, y_train, x_test, y_test = get_train_test_data(x, y)
	train(x_train, y_train, x_test, y_test)

main()
	
