from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.callbacks import EarlyStopping

(X_train, y_train), (X_test, y_test) = mnist.load_data()
nb_classes = 10

X_train = X_train.reshape(-1, 1, 28, 28).astype('float32')
X_test = X_test.reshape(-1, 1, 28, 28).astype('float32')
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filter = 16, nb_row = 3, nb_col = 3, border_mode = 'same', input_shape = (1, 28, 28)))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filter = 32, nb_row = 3, nb_col = 3, border_mode = 'same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2, 2), border_mode = 'same'))

model.add(Convolution2D(nb_filter = 64, nb_row = 3, nb_col = 3, border_mode = 'same'))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filter = 128, nb_row = 3, nb_col = 3, border_mode = 'same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2, 2), border_mode = 'same'))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, y_train, nb_epoch = 5, batch_size = 100, callbacks = [early_stopping])
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])