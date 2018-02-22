"""
Created on Feb 27, 2017

@author: Siyuan Huang

Training and Testing Code for Subactivity LSTM

"""

from __future__ import print_function
import numpy as np
import json
import h5py
import glob
import scipy.io
import os
from keras.preprocessing import sequence
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout, Input, Convolution2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.layers import LSTM
from keras.models import load_model
from keras.utils.visualize_util import plot


def data_prepare(path, maxlen=200):
    data_dim = 33
    num_train = 900
    label_list = ['null', 'reaching', 'moving', 'placing', 'opening', 'cleaning', 'closing', 'pouring', 'eating', 'drinking']
    path_skeleton = path + 'aligned_skeletons/'
    path_label = path + 'sequence_label.json'
    with open(path_label, 'r') as f:
        sequence_label = json.load(f)
    skeleton_list = glob.glob(path_skeleton + '*.mat')
    skeleton = {}
    # expected input data shape: (batch_size, timesteps, data_dim)
    # expected input label shape: (batch_size)
    x_all = np.zeros((len(sequence_label), maxlen, data_dim))
    y_all = np.zeros((len(sequence_label), len(label_list)))
    index = 0
    for skeleton_video in skeleton_list:
        sequence_id = skeleton_video.split('/')[-1][:-4]
        cur_skeleton = np.transpose(scipy.io.loadmat(skeleton_video)['skeleton'])
        skeleton[sequence_id] = cur_skeleton
    for sequence_cut in sequence_label:
        sequence_id = str(sequence_cut['sequence_id'])
        start_frame = sequence_cut['start_frame']
        end_frame = sequence_cut['end_frame']
        x_temp = skeleton[sequence_id][:, start_frame:end_frame]
        x_temp = skeleton_prune(x_temp)
        x_temp = sequence.pad_sequences(x_temp, dtype=float, maxlen=maxlen)
        x_all[index, :, :] = np.transpose(x_temp)
        y_all[index, label_list.index(sequence_cut['sequence_label'])] = 1
        index += 1
    random_index = np.random.permutation(len(sequence_label))
    train_index = random_index[:num_train]
    test_index = random_index[num_train:]
    # print(test_index)
    x_train = x_all[train_index, :, :]
    x_test = x_all[test_index, :, :]
    y_train = y_all[train_index]
    y_test = y_all[test_index]
    return x_train, x_test, y_train, y_test


def data_prepare_per_frame(path):
    num_frame = 57820
    # num_frame = 10000
    num_train = int(num_frame * 0.8)
    label_list = ['null', 'reaching', 'moving', 'placing', 'opening', 'cleaning', 'closing', 'pouring', 'eating',
                  'drinking']
    path_skeleton = path + 'aligned_skeletons/'
    path_label = path + 'sequence_label.json'
    with open(path_label, 'r') as f:
        sequence_label = json.load(f)
    skeleton_list = glob.glob(path_skeleton + '*.mat')
    skeleton = {}
    # expected input data shape: (image_num, data_dim)
    # expected input label shape: (image_num)
    index = 0
    for skeleton_video in skeleton_list:
        sequence_id = skeleton_video.split('/')[-1][:-4]
        cur_skeleton = np.transpose(scipy.io.loadmat(skeleton_video)['skeleton'])
        skeleton[sequence_id] = cur_skeleton
    for sequence_cut in sequence_label:
        sequence_id = str(sequence_cut['sequence_id'])
        start_frame = sequence_cut['start_frame']
        end_frame = sequence_cut['end_frame']
        x_temp = skeleton[sequence_id][:, start_frame:end_frame]
        x_temp = skeleton_prune(x_temp)
        y_temp = np.zeros((end_frame - start_frame, len(label_list)))
        y_temp[:, label_list.index(sequence_cut['sequence_label'])] = 1
        if index == 0:
            x_all = np.transpose(x_temp)
            y_all = y_temp
        else:
            x_all = np.concatenate((x_all, np.transpose(x_temp)), axis=0)
            y_all = np.concatenate((y_all, y_temp), axis=0)
        index += 1
    random_index = np.random.permutation(num_frame)
    train_index = random_index[:num_train]
    test_index = random_index[num_train:]
    # print(test_index)
    x_train = x_all[train_index, :]
    x_test = x_all[test_index, :]
    y_train = y_all[train_index]
    y_test = y_all[test_index]
    return x_train, x_test, y_train, y_test


def skeleton_prune(skeleton):  # input nparray (45, num_sample)
    anchor_effective = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13]
    point_effective = []
    for i in anchor_effective:
        point_effective.append((i-1) * 3)
        point_effective.append((i-1) * 3 + 1)
        point_effective.append((i-1) * 3 + 2)
    skeleton = skeleton[np.array(point_effective)]
    return skeleton


def result_output(x_test, y_test, x_train, y_train, model_path, cache_path):
    label_list = ['null', 'reaching', 'moving', 'placing', 'opening', 'cleaning', 'closing', 'pouring', 'eating',
                  'drinking']
    model = load_model(model_path)
    predict_test = model.predict(x_test)
    predict_train = model.predict(x_train)
    predict_test_result = []
    predict_train_result = []
    for i in range(len(x_test)):
        predict_test_result.append({})
        score = predict_test[i, :]
        predict_test_result[i]['set'] = 'test'
        predict_test_result[i]['gt_label'] = label_list[list(y_test[i, :]).index(1)]
        predict_test_result[i]['predict_label'] = label_list[list(score).index(max(score))]
        if predict_test_result[i]['predict_label'] == predict_test_result[i]['gt_label']:
            predict_test_result[i]['success'] = 'Yes'
        else:
            predict_test_result[i]['success'] = 'No'
    for i in range(len(x_train)):
        predict_train_result.append({})
        score = predict_train[i, :]
        predict_train_result[i]['set'] = 'train'
        predict_train_result[i]['gt_label'] = label_list[list(y_train[i, :]).index(1)]
        predict_train_result[i]['predict_label'] = label_list[list(score).index(max(score))]
        if predict_train_result[i]['predict_label'] == predict_train_result[i]['gt_label']:
            predict_train_result[i]['success'] = 'Yes'
        else:
            predict_train_result[i]['success'] = 'No'
    with open(cache_path+'predict_test.json', 'w') as f:
        json.dump(predict_test_result, f)
    with open(cache_path+'predict_train.json', 'w') as f:
        json.dump(predict_train_result, f)


def make_path(flipped=0):
    data_path = '/home/siyuan/Documents/iccv2017/'
    model_path = data_path + 'models/'
    cache_path = data_path + 'caches/'
    if flipped:
        data_path += 'flipped/'
        model_path += 'flipped/'
        cache_path += 'flipped/'
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    return data_path, model_path, cache_path


def train(x_train, x_test, y_train, y_test, model_path, maxlen=200):
    batch_size = 16
    model_name = 'layer_2_without_dropout_maxlen_100_epoch_100.h5'
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(maxlen, 33)))
    # model.add(Dropout(0.5))
    # model.add(LSTM(64, return_sequences=True))
    # model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    plot(model, to_file=model_path + model_name[:-3] + '.png', show_shapes=True)
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=100, validation_data=(x_test, y_test))
    model.save(model_path + model_name)
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    return model_path + model_name


def train_per_frame(x_train, x_test, y_train, y_test, model_path):
    model_name = 'layer_3_per_frame_without_dropout_epoch_1000.h5'
    batch_size = 16
    model = Sequential()
    model.add(Dense(64, input_dim=33))
    model.add(Dense(32))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    plot(model, to_file=model_path + model_name[:-3] + '.png', show_shapes=True)
    model.fit(x_train, y_train, nb_epoch=1000, validation_data=(x_test, y_test))
    model.save(model_path + model_name)
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy', acc)
    return model_path + model_name


def main():
    data_path, model_path, cache_path = make_path(1)
    np.random.seed(1337)
    print('loading data')
    # x_train, x_test, y_train, y_test = data_prepare(data_path, 100)
    x_train, x_test, y_train, y_test = data_prepare_per_frame(data_path)
    print('Build model ... ')
    print (x_train.shape, y_train.shape)
    # model_name = 'layer_1_epoch_200_dropout.h5'
    model_dir = train_per_frame(x_train, x_test, y_train, y_test, model_path)
    # result_output(x_test, y_test, x_train, y_train, model_dir, cache_path)


if __name__ == '__main__':
    main()

