"""
Created on March 1, 2017

@author: Siyuan Huang

Training and Testing Code for Affordance LSTM

"""

from __future__ import print_function
import numpy as np
import json
import os
import config
from affordance import load_affordance_data
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Merge
from keras.layers import LSTM
from keras.models import load_model
from keras.utils.visualize_util import plot


def data_splitting(object_feature, affordance_feature, distance_feature, orientation_feature):
    num_sample = len(object_feature)
    num_train = int(num_sample * 0.8)
    random_index = np.random.permutation(num_sample)
    train_index = random_index[:num_train]
    test_index = random_index[num_train:]
    x_1_train = object_feature[train_index, :]
    x_1_test = object_feature[test_index, :]
    x_2_train = np.concatenate((distance_feature[train_index, :, :], orientation_feature[train_index, :, :]), axis=2)
    x_2_test = np.concatenate((distance_feature[test_index, :, :], orientation_feature[test_index, :, :]), axis=2)
    y_train = affordance_feature[train_index, :]
    y_test = affordance_feature[test_index, :]
    return x_1_train, x_2_train, y_train, x_1_test, x_2_test, y_test


def result_output(x_1_train, x_2_train, y_train, x_1_test, x_2_test, y_test, model_path, cache_path):
    label_list = ['reachable', 'movable', 'pourable', 'pourto', 'containable', 'drinkable', 'openable', 'placeable', 'closeable', 'cleanable', 'cleaner', 'stationary']
    model = load_model(model_path)
    predict_test = model.predict([x_1_test, x_2_test])
    predict_train = model.predict([x_1_train, x_2_train])
    predict_test_result = []
    predict_train_result = []
    for i in range(len(x_1_test)):
        predict_test_result.append({})
        score = predict_test[i, :]
        predict_test_result[i]['set'] = 'test'
        predict_test_result[i]['gt_label'] = label_list[list(y_test[i, :]).index(1)]
        predict_test_result[i]['predict_label'] = label_list[list(score).index(max(score))]
        # predict_test_result[i]['score'] = list(np.array2string(score, formatter={'float_kind': lambda x: "%.2f" % x}))
        if predict_test_result[i]['predict_label'] == predict_test_result[i]['gt_label']:
            predict_test_result[i]['success'] = 'Yes'
        else:
            predict_test_result[i]['success'] = 'No'
    for i in range(len(x_1_train)):
        predict_train_result.append({})
        score = predict_train[i, :]
        predict_train_result[i]['set'] = 'train'
        predict_train_result[i]['gt_label'] = label_list[list(y_train[i, :]).index(1)]
        predict_train_result[i]['predict_label'] = label_list[list(score).index(max(score))]
        # predict_train_result[i]['score'] = list(np.array2string(score, formatter={'float_kind': lambda x: "%.2f" % x}))
        if predict_train_result[i]['predict_label'] == predict_train_result[i]['gt_label']:
            predict_train_result[i]['success'] = 'Yes'
        else:
            predict_train_result[i]['success'] = 'No'
    with open(cache_path + 'predict_test.json', 'w') as f:
        json.dump(predict_test_result, f)
    with open(cache_path + 'predict_train.json', 'w') as f:
        json.dump(predict_train_result, f)


def train(x_1_train, x_2_train, y_train, x_1_test, x_2_test, y_test, model_path):
    batch_size = 16
    model_name = 'layer_3_with_dropout_0.5_dropoutWU_0.4_maxlen_200_epoch_300_feature_completion.h5'
    left_branch = Sequential()
    left_branch.add(Dense(32, input_dim=x_1_train.shape[1]))
    # left_branch.add(Flatten())
    right_branch = Sequential()
    right_branch.add(LSTM(128, return_sequences=True, dropout_U=0.4, dropout_W=0.4, input_shape=x_2_train.shape[1:3]))
    right_branch.add(Dropout(0.5))
    right_branch.add(LSTM(64, dropout_U=0.4, dropout_W=0.4))
    right_branch.add(Dropout(0.5))
    right_branch.add(Dense(32))
    merged = Merge([left_branch, right_branch], mode='concat')
    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(12))
    final_model.add(Activation('softmax'))
    final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    plot(final_model, to_file=model_path + model_name[:-3] + '.png', show_shapes=True)
    plot(right_branch, to_file=model_path + model_name[:-3] + '_right_branch.png', show_shapes=True)
    plot(left_branch, to_file=model_path + model_name[:-3] + '_left_branch.png', show_shapes=True)
    final_model.fit([x_1_train, x_2_train], y_train, batch_size=batch_size, nb_epoch=300, validation_data=([x_1_test, x_2_test], y_test))
    final_model.save(model_path + model_name)
    score, acc = final_model.evaluate([x_1_test, x_2_test], y_test, batch_size=batch_size)
    print('Test Score', score)
    print('Test Accuracy', acc)
    return model_path + model_name


def main():
    paths = config.Paths()
    paths.path_huang()
    model_path = paths.metadata_root + 'models/affordance/'
    cache_path = paths.metadata_root + 'caches/affordance/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    np.random.seed(1337)
    print('Loading data')
    object_feature, affordance_feature, distance_feature, orientation_feature = load_affordance_data(paths.metadata_root)
    x_1_train, x_2_train, y_train, x_1_test, x_2_test, y_test = data_splitting(object_feature, affordance_feature, distance_feature, orientation_feature)
    print('Building model')
    model_dir = train(x_1_train, x_2_train, y_train, x_1_test, x_2_test, y_test, model_path)
    result_output(x_1_train, x_2_train, y_train, x_1_test, x_2_test, y_test, model_dir, cache_path)


if __name__ == '__main__':
    main()