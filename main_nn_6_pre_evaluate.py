import collections
import matplotlib.pyplot as plt
import functools
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels

from keras.models import Sequential, load_model
# from keras.metrics import
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
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


# def specificity(y_true, y_pred):
#     """
#     param:
#     y_pred - Predicted labels
#     y_true - True labels
#     Returns:
#     Specificity score
#     """
#     neg_y_true = 1 - y_true
#     neg_y_pred = 1 - y_pred
#     fp = K.sum(neg_y_true * y_pred)
#     tn = K.sum(neg_y_true * neg_y_pred)
#     specificity = tn / (tn + fp + K.epsilon())
#     return specificity

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def is_eq(list1, list2):
    if len(list1) != len(list2):
        return False
    else:
        for index_row in range(len(list1)):
            if list1[index_row] != list2[index_row]:
                print('{} and {} are NOT equal'.format(list1[index_row], list2[index_row]))
                return False

    return True




# train_files = [f for f in listdir('train_2') if isfile(join('train_2', f))]
test_files = [f for f in listdir('final_data/pre_preprocessed_test') if isfile(join('final_data/pre_preprocessed_test', f))]

X_TEST_DATA = []
Y_TEST_label = []

model_id = 'model_deep_nn_6_pre.h5'
# model_id = '3'

for reshape_file in test_files:
    if '.jpg' not in reshape_file:
        continue
    # x_image = cv2.imread('preprocessed_train/' + reshape_file)
    x_image = EImage.read_image('final_data/pre_preprocessed_test/' + reshape_file, if_read_as_grayscale=True)
    # print(x_image.shape)
    y_label = int(reshape_file.split('_')[0])
    X_TEST_DATA.append(x_image)
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

    Y_TEST_label.append(y_data)



X_TEST_DATA = np.array(X_TEST_DATA)
# Y_TEST_label = np.array(Y_TEST_label)


X_TEST_DATA_SCALED = X_TEST_DATA / 255.
# X_train, X_test, y_train, y_test = train_test_split(X_TRAIN_DATA_SCALED, Y_TEST_label, test_size=0.33)

model = load_model(model_id)

scores = model.evaluate(X_TEST_DATA_SCALED, Y_TEST_label)

# scores = model.evaluate(X_TEST_DATA_SCALED, Y_TEST_label, verbose=0)
print('score: {}'.format(scores))
Y_PREDICT = model.predict(X_TEST_DATA_SCALED)
for index_x in range(Y_PREDICT.shape[0]):
    for index_y in range(Y_PREDICT.shape[1]):
        Y_PREDICT[index_x,index_y] = 1 if Y_PREDICT[index_x,index_y]>=0.5 else 0
print(Y_PREDICT.shape)
print(Y_PREDICT.tolist())
print(Y_TEST_label)

# true_counts=0
# LIST_Y_PREDICT = []
# for index_row in range(len(Y_PREDICT)):
#     # if collections.Counter(Y_PREDICT[index_row].tolist()) == collections.Counter([1,0,0]):
#     list_index = Y_PREDICT[index_row]
#     LIST_Y_PREDICT.insert(index_row, list_index.tolist().index(max(list_index.tolist())))
#     # if is_eq(list_index, [1,0,0]):
#     # if Y_PREDICT[index_row] == [1,0,0]:
#     #     LIST_Y_PREDICT.insert(index_row, 1)
#     # elif is_eq(list_index, [0,1,0]):
#     #     LIST_Y_PREDICT.insert(index_row, 2)
#     # elif is_eq(list_index, [0,0,1]):
#     #     LIST_Y_PREDICT.insert(index_row, 3)
#     # else:
#     #     print('ERROR {}'.format(list_index))
#
# LIST_Y_TRUE = []
# for index_row in range(len(Y_TEST_label)):
#     list_index = Y_TEST_label[index_row]
#     LIST_Y_TRUE.insert(index_row, list_index.tolist().index(max(list_index.tolist())))
#     # if is_eq(list_index, [1,0,0]):
#     # if Y_PREDICT[index_row] == [1,0,0]:
#     #     LIST_Y_TRUE.insert(index_row, 1)
#     # elif is_eq(list_index, [0,1,0]):
#     #     LIST_Y_TRUE.insert(index_row, 2)
#     # elif is_eq(list_index, [0,0,1]):
#     #     LIST_Y_TRUE.insert(index_row, 3)
#     # else:
#     #     print('ERROR {}'.format(list_index))
#
# print('SIZE ARE EQUAL? {}'.format(len(LIST_Y_PREDICT)==len(LIST_Y_TRUE)))
# for index_row in range(len(LIST_Y_PREDICT)):
#     if LIST_Y_PREDICT[index_row]==LIST_Y_TRUE[index_row]:
#         true_counts+=1
#
#
# # print(LIST_Y_TRUE)
# # print(LIST_Y_P)
# print('score: {}'.format(true_counts*100/len(Y_PREDICT)))
#
# # print("score: ", scores)
# # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
#
rep_classification_report = metrics.classification_report(Y_TEST_label,
                                       Y_PREDICT,
                                       # target_names=[0,1,2]
                                       )

rep_cm = metrics.confusion_matrix(Y_TEST_label,Y_PREDICT)
tn, fp, fn, tp = metrics.confusion_matrix(Y_TEST_label,Y_PREDICT).ravel()
# rep_sc = metrics.SCORERS
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
print(rep_classification_report)
print(rep_cm)
# print(rep_sc)
print(tn, fp, fn, tp)
print('sensitivity: {}'.format(sensitivity))
print('specificity: {}'.format(specificity))
#
# ax = plot_confusion_matrix(Y_TEST_label, Y_PREDICT,classes=[0,1], title='Confusion Matrix')
# cm = metrics.confusion_matrix(Y_TEST_label, Y_PREDICT)
# ax.imshow('cm_{}.jpg'.format(model_id))
# print(cm)
#
# model.save('model_nn3.h5')

