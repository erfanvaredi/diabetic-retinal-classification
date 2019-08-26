import keras
from keras.models import Sequential
# from keras.metrics import
from keras.layers import Dense, Conv1D, Flatten, MaxPool2D, Dropout, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler

# create model

import cv2
import os
import numpy as np
from os import listdir
from os.path import isfile, join

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# train_files = [f for f in listdir('data_train_preprocessed_reshape') if isfile(join('data_train', f))]
from nn_final_image.ETools import EImage

train_files = [f for f in listdir('final_data/pre_preprocessed_train') if isfile(join('final_data/pre_preprocessed_train', f))]
# test_files = [f for f in listdir('test_2') if isfile(join('test_2', f))]

X_TRAIN_DATA = []
Y_TRAIN_label = []
# X_TEST_DATA = []
# Y_TEST_label = []

for reshape_file in train_files:
    if '.jpg' not in reshape_file:
        continue
    # x_image = cv2.imread('preprocessed_train/' + reshape_file)
    x_image = EImage.read_image('final_data/pre_preprocessed_train/' + reshape_file, if_read_as_grayscale=True)
    # print(x_image.shape)
    y_label = int(reshape_file.split('_')[0])
    X_TRAIN_DATA.append(x_image)
    y_data = None
    #  5, 10, 25, 50 ,100
    if y_label == 0:
        y_data = 0
    else:
        y_data = 1

    # else:
    #     print('ERROR-Y_TRAIN_label -> unvalid classlabel ', reshape_file)
    #     print('Y_TRAIN_label: ', y_label)
    #     print('Y_TRAIN_label.type: ', type(y_label))
    #     break

    Y_TRAIN_label.append(y_data)


# for reshape_file in test_files:
#     x_image = cv2.imread('test_2/' + reshape_file)
    # y_label = int(reshape_file.split('_')[0])
    # X_TEST_DATA.append(x_image)
    # y_data = None
    #  5, 10, 25, 50 ,100
    # if y_label == 5:
    #     y_data = [1, 0, 0, 0, 0]
    # elif y_label == 10:
    #     y_data = [0, 1, 0, 0, 0]
    # elif y_label == 25:
    #     y_data = [0, 0, 1, 0, 0]
    # elif y_label == 50:
    #     y_data = [0, 0, 0, 1, 0]
    # elif y_label == 100:
    #     y_data = [0, 0, 0, 0, 1]
    # else:
    #     print('ERROR-Y_TEST_label -> unvalid classlabel ', reshape_file)
    #     print('Y_TEST_label: ', y_label)
    #     print('Y_TEST_label.type: ', type(y_label))
    #     break
    #
    # Y_TEST_label.append(y_data)

# scaler_x = MinMaxScaler()
# scaler_y = MinMaxScaler()


X_TRAIN_DATA = np.array(X_TRAIN_DATA)
print('X_TRAIN_DATA.shape {}'.format(X_TRAIN_DATA.shape))
# X_TRAIN_DATA = X_TRAIN_DATA.reshape(X_TRAIN_DATA.shape[0], 512, 512, 1)
# print('X_TRAIN_DATA.shape {}'.format(X_TRAIN_DATA.shape))
# Y_TRAIN_label = np.array(Y_TRAIN_label)

# X_TEST_DATA = np.array(X_TEST_DATA)
# Y_TEST_label = np.array(Y_TEST_label)

# nsamples, nx, ny = X_DATA.shape
# X_DATA = X_DATA.reshape((nsamples,nx*ny))

X_TRAIN_DATA_SCALED = X_TRAIN_DATA / 255.
# X_TEST_DATA_SCALED = X_TEST_DATA / 255.



# scaler_x.fit_transform(X_DATA)
# scaler_y.fit_transform(Y_label)

# X_DATA_SCALED = scaler_x.transform(X_DATA)
# Y_label_SCALED = scaler_y.transform(Y_label)

# scalers = {}
# for i in range(X_DATA.shape[2]):
#     scalers[i] = StandardScaler()
#     X_DATA[:, i, :] = scalers[i].fit_transform(X_DATA[:, i, :])

print('X_DATA_SCALED.shape = {}'.format(X_TRAIN_DATA_SCALED.shape))
#
# scaler = StandardScaler()
# num_instances, num_time_steps, num_features = X_DATA.shape
# X_DATA = np.reshape(X_DATA, shape=(-1, num_features))
# X_DATA = scaler.fit_transform(X_DATA)


print('X_TRAIN_DATA_SCALED.shape = {}'.format(X_TRAIN_DATA_SCALED.shape))
# print('X_TEST_DATA_SCALED.shape = {}'.format(X_TEST_DATA_SCALED.shape))
print('Y_label.shape = {}'.format(len(Y_TRAIN_label)))

# X_train, X_test, y_train, y_test = train_test_split(X_DATA_SCALED, Y_label, test_size=0.33)

model = Sequential()
# add model layers
# model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(80, 80,3)))
# model.add(Conv2D(32, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))
# model.add(Flatten())
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))

model.add(Conv1D(32, kernel_size=5, activation='relu', input_shape=(256, 256)))
# model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(80, 80, 3)))
# model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(80, 80, 3)))
model.add(MaxPooling1D(pool_size=2,strides=2))
model.add(Conv1D(64, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
# model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

print(X_TRAIN_DATA_SCALED)

# model.fit(X_TRAIN_DATA_SCALED, Y_TRAIN_label, batch_size=10, epochs=10,shuffle=True)
model.fit(X_TRAIN_DATA_SCALED, Y_TRAIN_label, batch_size=10, epochs=10,validation_split=0.2,shuffle=True)
# model.
# res = model.predict(X_TEST_DATA_SCALED)
# print(res)
# print(type(res))
# print(res.shape)



# scores = model.evaluate(X_TEST_DATA, Y_TEST_label, verbose=0)
# print("score: ", scores)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

model.save('model_deep_nn_6_pre.h5')