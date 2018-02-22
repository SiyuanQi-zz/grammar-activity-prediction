import os
import h5py
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, Merge, initializations, Input, TimeDistributed, LSTM, Reshape
from keras.optimizers import SGD, rmsprop
from keras.utils.visualize_util import plot
from keras.models import Model, load_model
import config
import glob
import cv2
import json
import scipy.io
import scipy.misc
import sklearn.utils
import matplotlib.pyplot as plt
import pickle
from subactivity_lstm import skeleton_prune
from sklearn import preprocessing
import sklearn.metrics
import vizutil
from sklearn.svm import SVC


def vgg_16(weights_path=None, nb_classes=1000):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten(name="flatten"))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)
    return model


def img_from_list(batch_size, x_path, y_path, bdb_path, nb_classes):
    x_file = list()
    with open(x_path, 'r') as f:
        x_file.extend(f.readlines())
    f.close()
    y_file = np.load(y_path)
    bdb_file = np.load(bdb_path).astype(int)
    while 1:
        for i in range(len(x_file) // batch_size):
            x_batch = np.zeros((batch_size, 3, 224, 224))
            y_batch = np.zeros((batch_size, nb_classes))
            for j in range(batch_size):
                x_batch[j] = img_preprocessing_bdb(x_file[i * batch_size + j].split('\n')[-2], bdb_file[i * batch_size + j, :])
                y_batch[j] = y_file[i * batch_size + j, :]
            yield x_batch, y_batch


def img_preprocessing(img_path):
    im = cv2.imread(img_path)
    im = cv2.resize(im, (224, 224)).astype(np.float32)
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return im


def img_preprocessing_bdb(img_path, bdb):
    im = scipy.misc.imread(img_path)
    im = im[bdb[1]:bdb[3], bdb[0]:bdb[2], :]
    im = cv2.resize(im, (224, 224)).astype(np.float32)
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return im


def get_bdb(points, left_handed=0):
    fx = 525.0
    fy = 525.0
    cx = 319.5
    cy = 239.5
    x = points[0]
    y = points[1]
    z = points[2]
    v = cy - y * fy / z
    if not left_handed:
        u = cx + x * fx / z
    else:
        u = cx - x * fx / z
    return u, v


def distance_skeleton(skeleton):
    index = 0
    num_point = 11
    for i in range(num_point):
        for j in range(num_point):
            if j != i:
                if index == 0:
                    dis_skeleton = skeleton[:, i*3:(i+1)*3] - skeleton[:, j*3:(j+1)*3]
                else:
                    dis_skeleton = np.concatenate((dis_skeleton, skeleton[:, i*3:(i+1)*3] - skeleton[:, j*3:(j+1)*3]), axis=1)
                index += 1
    dis_skeleton = np.concatenate((dis_skeleton, skeleton), axis=1)
    return dis_skeleton


def distance_hand_head(skeleton):
    hand_index = []
    head_neck_index = []
    index = 0
    for i in hand_index:
        for j in head_neck_index:
            if index==0:
                dis_skeleton = skeleton[:, i*3:(i+1)*3] - skeleton[:, j*3:(j+1)*3]
            else:
                dis_skeleton = np.concatenate((dis_skeleton, skeleton[:, i*3:(i+1)*3] - skeleton[:, j*3:(j+1)*3]), axis=1)
                index += 1
    dis_skeleton = np.concatenate((dis_skeleton, skeleton), axis=1)
    return dis_skeleton


def data_prepare_subactivity(data_root, metadata_root):
    label_list = ['null', 'reaching', 'moving', 'placing', 'opening', 'cleaning', 'closing', 'pouring', 'eating',
                  'drinking']
    label_path = '/home/siyuan/Documents/iccv2017/flipped/sequence_label.json'
    path_skeleton = metadata_root + 'flipped/skeletons/'
    aligned_skeleton_path = metadata_root + 'flipped/aligned_skeletons/'
    aligned_skeleton_list = glob.glob(aligned_skeleton_path + '*.mat')
    skeleton_list = glob.glob(path_skeleton + '*.mat')
    aligned_skeleton = dict()
    sequential_aligned_skeleton = dict()
    skeleton = dict()
    bdb = dict()

    # initialize the y_all matrix
    with open(label_path, 'r') as f:
        sequence_label = json.load(f)
    subject_set = [1, 3, 4, 5]
    label_id = dict()
    num_frame = dict()
    for sequence_cut in sequence_label:
        sequence_id = str(sequence_cut['sequence_id'])
        label_id[sequence_id] = np.zeros((2000, len(label_list)))
        num_frame[sequence_id] = 0
    for sequence_cut in sequence_label:
        sequence_id = str(sequence_cut['sequence_id'])
        start_frame = sequence_cut['start_frame']
        end_frame = sequence_cut['end_frame']
        if end_frame > num_frame[sequence_id]:
            num_frame[sequence_id] = end_frame
        label_id[sequence_id][start_frame:end_frame + 1, label_list.index(sequence_cut['sequence_label'])] = 1
    for sequence_id in label_id:
        label_id[sequence_id] = label_id[sequence_id][:num_frame[sequence_id] + 1, :]
    dis_skeleton_all = dict()
    # Processing the skeleton feature
    for aligned_skeleton_video in aligned_skeleton_list:
        sequence_id = aligned_skeleton_video.split('/')[-1][:-4]
        cur_aligned_skeleton = np.transpose(scipy.io.loadmat(aligned_skeleton_video)['skeleton'])
        cur_aligned_skeleton = np.transpose(skeleton_prune(cur_aligned_skeleton))
        # print sequence_id
        aligned_skeleton[sequence_id] = cur_aligned_skeleton
        dis_skeleton = distance_skeleton(cur_aligned_skeleton)
        m, n = dis_skeleton.shape
        sequential_num = 3
        cur_sequential_aligned_skeleton = np.zeros((m, n * (sequential_num + 1)))
        for i in range(m):
            if i < sequential_num:
                for j in range(i):
                    cur_sequential_aligned_skeleton[i, j*n:(j+1)*n] = dis_skeleton[i, :] - dis_skeleton[(i-j-1), :]
            else:
                for j in range(sequential_num):
                    cur_sequential_aligned_skeleton[i, j*n:(j+1)*n] = dis_skeleton[i, :] - dis_skeleton[(i-j-1), :]
            cur_sequential_aligned_skeleton[i, sequential_num*n:(sequential_num+1)*n] = dis_skeleton[i, :].copy()
        sequential_aligned_skeleton[sequence_id] = cur_sequential_aligned_skeleton
        dis_skeleton_all[sequence_id] = dis_skeleton


        # processing the bounding box
    for skeleton_video in skeleton_list:
        subject_3_set = ['1204174554', '1204174740', '1204174844', '1204173536', '1204173846', '1204174024', '1204174314', '0510143426',
         '0510143446', '0510143618', '0510142800', '0510142419', '0510142336', '1204180344', '1204180515', '1204180612',
         '1204175712', '1204175902', '1204175622', '1204175316', '1204175451', '1204175103', '0510144139', '0510144057',
         '0510144215', '0510141947', '0510141923', '0510142045', '0510144450', '0510144324', '0510144350']
        sequence_id = skeleton_video.split('/')[-1][:-4]
        # print sequence_id
        cur_skeleton = np.transpose(scipy.io.loadmat(skeleton_video)['skeleton'])
        skeleton[sequence_id] = cur_skeleton
        skeleton_video_frame = cur_skeleton.shape[1]
        bdb[sequence_id] = np.zeros((skeleton_video_frame, 4))
        if str(sequence_id) in subject_3_set:
            left_handed = 1
        else:
            left_handed = 0
        for i in range(skeleton_video_frame):
            padding_size = 50
            im_pos = np.zeros((15, 2))
            for j in range(15):
                u, v = get_bdb(cur_skeleton[j*3:(j+1)*3, i], left_handed)
                im_pos[j, 0] = u
                im_pos[j, 1] = v
            bdb_temp = np.zeros(4)
            bdb_temp[0] = int(min(im_pos[:, 0]))
            bdb_temp[1] = int(min(im_pos[:, 1]))
            bdb_temp[2] = int(max(im_pos[:, 0]))
            bdb_temp[3] = int(max(im_pos[:, 1]))
            bdb_temp[0] = max(0, bdb_temp[0] - padding_size)
            bdb_temp[2] = min(640, bdb_temp[2] + padding_size)
            bdb_temp[1] = max(0, bdb_temp[1] - padding_size)
            bdb_temp[3] = max(480, bdb_temp[3] + padding_size)
            bdb[sequence_id][i, :] = bdb_temp

    # save the data for each subject
    for subject_index in subject_set:
        video_frame_count = dict()
        img_set = list()
        y_all = list()
        bdb_all = list()
        sk_all = list()
        sk_sq_all = list()
        subject = 'Subject' + str(subject_index) + '_rgbd_images/'
        action = sorted(os.listdir(data_root + subject))
        subject_path = metadata_root + 'data/subject' + str(subject_index)
        if not os.path.exists(subject_path):
            os.mkdir(subject_path)
        img_path = subject_path + '/' + 'img_path.txt'
        gt_path = subject_path + '/' + 'subactivity_gt.npy'
        bdb_path = subject_path + '/' + 'bdb_gt.npy'
        frame_count_path = subject_path + '/' + 'frame_count.json'
        sk_path = subject_path + '/' + 'aligned_sk.npy'
        sk_sq_path = subject_path + '/' + 'sk_sq.npy'
        lstm_feature_path = subject_path + '/' + 'subactivity_lstm_feature.npy'
        lstm_gt_path = subject_path + '/' + 'subactivity_gt_feature.npy'
        segment_len = 50
        padding_len = 5
        segment_index = 0
        predict_len = 42
        for action_category in action:
            video = sorted(os.listdir(data_root + subject + action_category))
            for sequence_id in video:
                video_frame_count[sequence_id] = dict()
                video_frame_count[sequence_id]['start_num'] = len(y_all)
                img_list = glob.glob(data_root + subject + action_category + '/' + sequence_id + '/*.png')
                num_img = int(len(img_list) / 2)
                y_all.extend(label_id[str(sequence_id)].tolist())
                sk_sq_all.extend(sequential_aligned_skeleton[str(sequence_id)].tolist())
                sk_all.extend(aligned_skeleton[str(sequence_id)].tolist())
                bdb_all.extend(bdb[str(sequence_id)].tolist())
                video_frame_count[sequence_id]['end_num'] = len(y_all)
                dis_skeleton = dis_skeleton_all[sequence_id]
                m, n = dis_skeleton.shape
                # processing the data for prediction
                m -= predict_len
                # print dis_skeleton.shape
                i = 0
                gt_temp = np.array(label_id[sequence_id])
                print 'processing lstm feature'
                while 1:
                    if segment_index == 0:
                        subactivity_lstm_feature = np.expand_dims(
                            dis_skeleton[i * padding_len:i * padding_len + segment_len, :], axis=0)
                        subactivity_lstm_gt = np.expand_dims(gt_temp[i * padding_len + predict_len:i * padding_len + segment_len + predict_len, :],
                                                             axis=0)
                    else:
                        subactivity_lstm_feature = np.concatenate((subactivity_lstm_feature, np.expand_dims(
                            dis_skeleton[i * padding_len:i * padding_len + segment_len, :], axis=0)), axis=0)
                        # print m, n, gt_temp.shape, subactivity_lstm_gt.shape, np.expand_dims(gt_temp[i * padding_len:i * padding_len + segment_len, :], axis=0).shape
                        subactivity_lstm_gt = np.concatenate((subactivity_lstm_gt, np.expand_dims(
                            gt_temp[i * padding_len + predict_len:i * padding_len + segment_len + predict_len, :], axis=0)), axis=0)
                    i += 1
                    segment_index += 1
                    if i * padding_len + segment_len > m:
                        break
                for frame in range(num_img):
                    img_route = data_root + subject + action_category + '/' + sequence_id + '/' + 'RGB_' + str(frame+1) + '.png'
                    img_set.append(img_route)
                print sequence_id, action_category, subject_index, len(img_set), len(y_all), len(bdb_all), len(sk_all), len(sk_sq_all)
                print video_frame_count[sequence_id]['start_num'], video_frame_count[sequence_id]['end_num']
        with open(img_path, 'w') as f:
            for item in img_set:
                print >>f, item
        with open(frame_count_path, 'w') as f:
            json.dump(video_frame_count, f)
        f.close()
        # normalize the lstm feature
        l, m, n = subactivity_lstm_feature.shape
        subactivity_lstm_feature_temp = np.reshape(subactivity_lstm_feature, (l*m, n))
        feature_mean = np.mean(subactivity_lstm_feature_temp, axis=0)
        feature_var = np.var(subactivity_lstm_feature_temp, axis=0)
        # subactivity_lstm_feature_temp = (subactivity_lstm_feature_temp - feature_mean) / feature_var
        subactivity_lstm_feature_temp = preprocessing.scale(subactivity_lstm_feature_temp)
        print np.mean(subactivity_lstm_feature_temp, 0), np.var(subactivity_lstm_feature_temp, 0)
        subactivity_lstm_feature = np.reshape(subactivity_lstm_feature_temp, (l, m, n))
        # normalize the skeleton sequence feature
        # sk_sq_mean = np.mean(sk_sq_all, axis=0)
        # sk_sq_variance = np.var(sk_sq_all, axis=0)
        # print sk_sq_variance
        # sk_sq_all = (sk_sq_all - sk_sq_mean) / sk_sq_variance
        # print np.mean(sk_sq_all, axis=0), np.var(sk_sq_all, axis=0)
        sk_sq_all = preprocessing.scale(sk_sq_all)
        print np.mean(sk_sq_all, 0), np.std(sk_sq_all, 0)
        np.save(open(sk_path, 'w'), sk_all)
        np.save(open(gt_path, 'w'), y_all)
        np.save(open(bdb_path, 'w'), bdb_all)
        np.save(open(sk_sq_path, 'w'), sk_sq_all)
        np.save(open(lstm_feature_path, 'w'), subactivity_lstm_feature)
        np.save(open(lstm_gt_path, 'w'), subactivity_lstm_gt)
    print len(img_set)


def affordance_feature(affordance_label, object_label, object_pos, skeleton, skeleton_aligned):
    affordance_category = ['reachable', 'movable', 'pourable', 'pourto', 'containable', 'drinkable', 'openable',
                        'placeable', 'closeable', 'cleanable', 'cleaner', 'stationary']
    object_category = ['medcinebox', 'cup', 'bowl', 'box', 'milk', 'book', 'microwave', 'plate', 'remote', 'cloth']
    num_frame = len(object_pos)
    num_skeleton = len(skeleton)
    if num_frame != num_skeleton:
        print 'detect difference'
        skeleton = skeleton[num_skeleton-num_frame:num_skeleton, :]
        skeleton_aligned = skeleton_aligned[num_skeleton-num_frame:num_skeleton, :]
        print num_frame, num_skeleton, skeleton.shape, skeleton_aligned.shape
    object_label_feature = np.zeros((num_frame, 10))
    affordance_label_feature = np.zeros((num_frame, 12))
    normal_feature = np.zeros((num_frame, 33))
    skeleton_feature = distance_skeleton(skeleton_aligned)
    object_pos_feature = np.zeros((num_frame, 3))
    for i in range(num_frame):
        object_label_feature[i, object_category.index(object_label)] = 1
        affordance_label_feature[i, affordance_category.index(affordance_label)] = 1
        object_pos_feature[i, :] = object_pos[i]
        if np.isnan(object_pos_feature[i, 0]):
            print 'there is nan'
        normal_feature[i, :] = np.array([object_pos_feature[i, :] - skeleton[i][j*3:(j+1)*3] for j in range(11)]).flatten()
    # print object_pos_feature.shape, normal_feature.shape, skeleton_feature.shape
    geometry_feature = np.concatenate((object_pos_feature, normal_feature, skeleton_feature), axis=1)
    sequential_num = 3
    m, n = geometry_feature.shape
    sequential_geometry_feature = np.zeros((m, n * (sequential_num + 1)))
    for i in range(m):
        if i < sequential_num:
            for j in range(i):
                sequential_geometry_feature[i, j*n:(j+1)*n] = geometry_feature[i, :] - geometry_feature[(i-j-1), :]
        else:
            for j in range(sequential_num):
                sequential_geometry_feature[i, j*n:(j+1)*n] = geometry_feature[i, :] - geometry_feature[(i-j-1), :]
        sequential_geometry_feature[i, sequential_num*n:(sequential_num+1)*n] = geometry_feature[i, :].copy()
    return object_label_feature, sequential_geometry_feature, affordance_label_feature


def data_prepare_subactivity_cad(data_root, metadata_root, trainable=1):
    label_list = ['reaching', 'moving', 'pouring', 'eating', 'drinking', 'opening', 'placing', 'closing', 'null', 'cleaning']
    label_path = '/home/siyuan/Documents/iccv2017/flipped/sequence_label.json'
    relative_path = 'flipped/all/activity_corpus.p'
    result_path = metadata_root + 'data/subactivity_result_temp/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if os.path.exists(metadata_root + relative_path):
        activity_corpus = pickle.load(open(metadata_root + relative_path, 'rb'))
    tpg_id = dict()
    for activity, tpgs in activity_corpus.items():
        for tpg in tpgs:
            tpg_id[tpg.id] = tpg.terminals
    file_path = metadata_root + 'data/features.p'
    batch_size = 10
    subject_set = [1, 3, 4, 5]
    train_set = [1, 3, 4]
    test_set = [5]
    sub_feature = pickle.load(open(file_path))
    train_data = np.empty((0, 1030))
    train_gt = np.empty((0, 10))
    val_data = np.empty((0, 1030))
    val_gt = np.empty((0, 10))
    model_path = metadata_root + 'models/cnn/'
    # model_name = 'cad_subactivity_3_layer_epoch_100_test_5.h5'
    model_name = 'cad_subactivity_3_layer_epoch_200_test_5_add_feature_final.h5'
    val_sequence = list()
    spg_num = dict()
    for subject_index in subject_set:
        print 'subject' + str(subject_index)
        subject = 'Subject' + str(subject_index) + '_rgbd_images/'
        action = sorted(os.listdir(data_root + subject))
        subject_path = metadata_root + 'data/subject' + str(subject_index)
        feature_path = subject_path + '/' + 'subactivity_cad_feature.npy'
        gt_path = subject_path + '/' + 'subactivity_cad_gt.npy'
        index = 0
        for action_category in action:
            video = sorted(os.listdir(data_root + subject + action_category))
            for sequence_id in video:
                if subject_index in test_set:
                    val_sequence.append(sequence_id)
                sequence_info = sub_feature[sequence_id]
                spg_len = 0
                for sequence_index in sequence_info:
                    y_temp = np.zeros(10)
                    y_temp[sequence_index['h_act'] - 1] = 1
                    x_temp = np.hstack((sequence_index['h_fea'], np.mean(np.array(sequence_index['s_o_fea']), axis=0)))
                    # x_temp = sequence_index['h_fea']
                    if index == 0:
                        y_all = y_temp
                        x_all = x_temp
                    else:
                        y_all = np.vstack((y_all, y_temp))
                        x_all = np.vstack((x_all, x_temp))
                    index += 1
                    spg_len += 1
                spg_num[sequence_id] = spg_len
        if subject_index in train_set:
            print train_data.shape, x_all.shape
            train_data = np.vstack((train_data, x_all))
            train_gt = np.vstack((train_gt, y_all))
        else:
            val_data = np.vstack((val_data, x_all))
            val_gt = np.vstack((val_gt, y_all))
        print x_all.shape
    model = Sequential()
    model.add(Dense(4096, init=my_init, input_dim=train_data.shape[1], activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(1024, init=my_init, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(512, init=my_init, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, init=my_init))
    model.add(Activation('softmax'))
    early_stopping = EarlyStopping(verbose=1, patience=30, monitor='acc')
    model_checkpoint = ModelCheckpoint(
        model_path + model_name, save_best_only=True,
        save_weights_only=True,
        monitor='acc')
    callbacks_list = [early_stopping, model_checkpoint]
    if trainable:
        optimizer = rmsprop(lr=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_gt, batch_size=batch_size, nb_epoch=200, validation_data=(val_data, val_gt), callbacks=callbacks_list)
        model.save(model_path + model_name)
        plot(model, to_file=model_path + model_name[:-3] + '.png', show_shapes=True)
    # evaluation
    else:
        subject_index = test_set[0]
        print val_data.shape
        model.load_weights(model_path + model_name)
        prediction = model.predict(val_data)
        index = 0
        correct_num_all = 0
        frame_all = 0
        result_total = np.empty(0)
        gt_total = np.empty(0)
        print val_gt
        for sequence_id in val_sequence:
            print sequence_id
            tpg = tpg_id[sequence_id]
            spg_index = 0
            spg_len = spg_num[sequence_id]
            for spg in tpg:
                start_frame = spg.start_frame
                end_frame = spg.end_frame
                gt_spg = np.zeros(end_frame - start_frame + 1)
                result_spg = np.zeros(end_frame - start_frame + 1)
                score_spg = np.zeros((end_frame - start_frame + 1, 10))
                score_spg[:] = prediction[index, :]
                for i in range(len(gt_spg)):
                    result_spg[i] = np.argmax(score_spg[i, :])
                    gt_spg[i] = np.argmax(val_gt[index])
                if spg_index == 0:
                    tpg_score = score_spg
                    tpg_result = result_spg
                    tpg_gt = gt_spg
                else:
                    tpg_score = np.vstack((tpg_score, score_spg))
                    tpg_result = np.hstack((tpg_result, result_spg))
                    tpg_gt = np.hstack((tpg_gt, gt_spg))
                index += 1
                spg_index += 1
                if spg_index == spg_len:
                    break
            np.save(open(result_path + 'subject' + str(subject_index) + '/' + sequence_id + '.npy', 'w'),
                    tpg_score)
            correct_num = 0
            for frame_index in range(len(tpg_score)):
                if tpg_gt[frame_index] == tpg_result[frame_index]:
                    correct_num += 1
            correct_num_all += correct_num
            frame_all += len(tpg_gt)
            vizutil.plot_segmentation([tpg_gt, tpg_result, (tpg_gt - tpg_result) == 0],
                                      len(tpg_gt))
            plt.savefig(result_path + 'subject' + str(subject_index) + '/' + sequence_id + '_' + str(
                float(correct_num) / len(tpg_gt)) + '.png')
            plt.close()
            cm = sklearn.metrics.confusion_matrix(tpg_gt, tpg_result,
                                                  labels=range(10))
            vizutil.plot_confusion_matrix(cm, classes=label_list, normalize=True, filename=result_path + 'subject' + str(
                subject_index) + '/' + sequence_id + '_confusion.png')
            result_total = np.hstack((result_total, tpg_result))
            gt_total = np.hstack((gt_total, tpg_gt))
        cm = sklearn.metrics.confusion_matrix(gt_total, result_total,
                                              labels=range(10))
        vizutil.plot_confusion_matrix(cm, classes=label_list, normalize=True, filename=result_path + 'subject' + str(
            subject_index) + '/a_confusion.png')
        print float(correct_num_all) / frame_all


def data_prepare_affordance_cad(data_root, metadata_root, trainable=1):
    label_list = ['movable', 'stationary', 'reachable', 'pourable', 'pourto', 'containable', 'drinkable', 'openable', 'placeable', 'closeable', 'cleanable', 'cleaner']
    label_path = '/home/siyuan/Documents/iccv2017/flipped/sequence_label.json'
    relative_path = 'flipped/all/activity_corpus.p'
    if os.path.exists(metadata_root + relative_path):
        activity_corpus = pickle.load(open(metadata_root + relative_path, 'rb'))
    file_path = metadata_root + 'data/features.p'
    result_path = metadata_root + 'data/affordance_result_temp/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    model_path = metadata_root + 'models/cnn/'
    model_name = 'cad_affordance_3_layer_epoch_150_test_5_add_feature_max.h5'
    nb_epoch = 150
    batch_size = 10
    subject_set = [1, 3, 4, 5]
    train_set = [1, 3, 4]
    test_set = [5]
    sub_feature = pickle.load(open(file_path))
    train_data = np.empty((0, 780))
    train_gt = np.empty((0, 12))
    val_data = np.empty((0, 780))
    val_gt = np.empty((0, 12))
    val_sequence = list()
    tpg_id = dict()
    o_id = dict()
    for activity, tpgs in activity_corpus.items():
        for tpg in tpgs:
            tpg_id[tpg.id] = tpg.terminals
    spg_num = dict()
    for subject_index in subject_set:
        print 'subject' + str(subject_index)
        subject = 'Subject' + str(subject_index) + '_rgbd_images/'
        action = sorted(os.listdir(data_root + subject))
        subject_path = metadata_root + 'data/subject' + str(subject_index)
        feature_path = subject_path + '/' + 'subactivity_cad_feature.npy'
        gt_path = subject_path + '/' + 'subactivity_cad_gt.npy'
        index = 0
        for action_category in action:
            video = sorted(os.listdir(data_root + subject + action_category))
            for sequence_id in video:
                if subject_index in test_set:
                    val_sequence.append(sequence_id)
                sequence_info = sub_feature[sequence_id]
                spg_len = 0
                for sequence_index in sequence_info:
                    obj_index = 0
                    o_id[sequence_id] = sequence_index['o_id']
                    for object in sequence_index['o_id']:
                        y_temp = np.zeros(12)
                        y_temp[sequence_index['o_aff'][obj_index] - 1] = 1
                        o_id_temp = o_id[sequence_id][obj_index]
                        ob_num = len(o_id[sequence_id])
                        if ob_num == 1:
                            o_o_feature = np.zeros(200)
                        else:
                            o_o_feature = np.max(np.array(sequence_index['o_o_fea'])[(o_id_temp-1)*(ob_num-1):o_id_temp*(ob_num-1), :], axis=0)
                        x_temp = np.hstack((sequence_index['o_fea'][obj_index], sequence_index['s_o_fea'][obj_index], o_o_feature))
                        if index == 0:
                            y_all = y_temp
                            x_all = x_temp
                        else:
                            y_all = np.vstack((y_all, y_temp))
                            x_all = np.vstack((x_all, x_temp))
                        index += 1
                        obj_index += 1
                    spg_len += 1
                spg_num[sequence_id] = spg_len
        if subject_index in train_set:
            print train_data.shape, x_all.shape
            train_data = np.vstack((train_data, x_all))
            train_gt = np.vstack((train_gt, y_all))
        else:
            val_data = np.vstack((val_data, x_all))
            val_gt = np.vstack((val_gt, y_all))
    model = Sequential()
    model.add(Dense(4096, init=my_init, input_dim=train_data.shape[1], activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(1024, init=my_init, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(512, init=my_init, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(12, init=my_init))
    model.add(Activation('softmax'))
    if trainable:
        optimizer = rmsprop(lr=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_gt, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(val_data, val_gt))
        model.save(model_path + model_name)
        plot(model, to_file=model_path + model_name[:-3] + '.png', show_shapes=True)
    else:
        subject_index = test_set[0]
        print val_data.shape
        model.load_weights(model_path + model_name)
        prediction = model.predict(val_data)
        index = 0
        correct_num_all = 0
        frame_all = 0
        result_total = np.empty(0)
        gt_total = np.empty(0)
        for sequence_id in val_sequence:
            print sequence_id
            tpg = tpg_id[sequence_id]
            spg_index = 0
            spg_len = spg_num[sequence_id]
            nb_obj = len(tpg[0].objects)
            tpg_score = [list() for _ in range(nb_obj)]
            tpg_result = [list() for _ in range(nb_obj)]
            tpg_gt = [list() for _ in range(nb_obj)]
            for spg in tpg:
                start_frame = spg.start_frame
                end_frame = spg.end_frame
                for obj_index in range(nb_obj):
                    gt_spg = np.zeros(end_frame - start_frame + 1)
                    result_spg = np.zeros(end_frame - start_frame + 1)
                    score_spg = np.zeros((end_frame - start_frame + 1, 12))
                    score_spg[:] = prediction[index, :]
                    for i in range(len(gt_spg)):
                        result_spg[i] = np.argmax(score_spg[i, :])
                        gt_spg[i] = np.argmax(val_gt[index, :])
                    if spg_index == 0:
                        tpg_score[obj_index] = score_spg
                        tpg_result[obj_index] = result_spg
                        tpg_gt[obj_index] = gt_spg
                    else:
                        tpg_score[obj_index] = np.vstack((tpg_score[obj_index], score_spg))
                        tpg_result[obj_index] = np.hstack((tpg_result[obj_index], result_spg))
                        tpg_gt[obj_index] = np.hstack((tpg_gt[obj_index], gt_spg))
                    index += 1
                spg_index += 1
                if spg_index == spg_len:
                    break
            for obj_index in range(nb_obj):
                correct_num = 0
                for frame_index in range(len(tpg_score[obj_index])):
                    if tpg_gt[obj_index][frame_index] == tpg_result[obj_index][frame_index]:
                        correct_num += 1
                correct_num_all += correct_num
                frame_all += len(tpg_gt[obj_index])
                vizutil.plot_segmentation([tpg_gt[obj_index], tpg_result[obj_index], (tpg_gt[obj_index] - tpg_result[obj_index] == 0)], len(tpg_gt[obj_index]))
                plt.savefig(result_path + 'subject' + str(subject_index) + '/' + sequence_id + '_' + 'object' + str(o_id[sequence_id][obj_index] - 1) + '_' + str(
                    float(correct_num) / len(tpg_gt[obj_index])) + '.png')
                plt.close()
                cm = sklearn.metrics.confusion_matrix(tpg_gt[obj_index], tpg_result[obj_index],
                                                      labels=range(12))
                vizutil.plot_confusion_matrix(cm, classes=label_list, normalize=True,
                                              filename=result_path + 'subject' + str(
                                                  subject_index) + '/' + sequence_id + '_' + 'object' + str(o_id[sequence_id][obj_index] - 1) + '_confusion.png')
                result_total = np.hstack((result_total, tpg_result[obj_index]))
                gt_total = np.hstack((gt_total, tpg_gt[obj_index]))
                if obj_index == 0:
                    temp = np.expand_dims(tpg_score[o_id[sequence_id].index(obj_index + 1)], axis=0)
                    sequence_score = temp
                else:
                    temp = np.expand_dims(tpg_score[o_id[sequence_id].index(obj_index + 1)], axis=0)
                    sequence_score = np.concatenate((sequence_score, temp), axis=0)
            np.save(open(result_path + 'subject' + str(subject_index) + '/' + sequence_id + '.npy', 'w'), sequence_score)
        cm = sklearn.metrics.confusion_matrix(gt_total, result_total,
                                              labels=range(12))
        vizutil.plot_confusion_matrix(cm, classes=label_list, normalize=True, filename=result_path + 'subject' + str(
            subject_index) + '/a_confusion.png')
        print float(correct_num_all) / frame_all


def data_prepare_affordance(data_root, metadata_root):
    relative_path = 'flipped/all/activity_corpus.p'
    aligned_skeleton_path = metadata_root + 'flipped/skeletons/'
    tpg_id = dict()
    subject_set = [1, 3, 4, 5]
    if os.path.exists(metadata_root + relative_path):
        activity_corpus = pickle.load(open(metadata_root + relative_path, 'rb'))
    for activity, tpgs in activity_corpus.items():
        for tpg in tpgs:
            tpg_id[tpg.id] = tpg.terminals
    for subject_index in subject_set:
        subject = 'Subject' + str(subject_index) + '_rgbd_images/'
        action = os.listdir(data_root + subject)
        subject_path = metadata_root + 'data/subject' + str(subject_index)
        if not os.path.exists(subject_path):
            os.mkdir(subject_path)
        gt_path = subject_path + '/' + 'affordance_gt.npy'
        feature_path = subject_path + '/' + 'affordance_sequential_feature.npy'
        label_path = subject_path + '/' + 'affordance_object_label_feature.npy'
        index = 0
        frame_count_path = subject_path + '/' + 'affordance_frame_count.json'
        frame_count = dict()
        for action_category in action:
            video = os.listdir(data_root + subject + action_category)
            for sequence_id in video:
                frame_count[sequence_id] = dict()
                frame_count[sequence_id]['object'] = list()
                frame_count[sequence_id]['frame_record'] = list()
                tpg = tpg_id[sequence_id]
                cur_aligned_skeleton = np.transpose(scipy.io.loadmat(aligned_skeleton_path + sequence_id + '.mat')['skeleton'])
                cur_aligned_skeleton = np.transpose(skeleton_prune(cur_aligned_skeleton))
                frame_count[sequence_id]['length'] = cur_aligned_skeleton.shape[0]
                for obj_object in tpg[0].objects:
                    frame_count[sequence_id]['object'].append(obj_object)
                spg_count = 0
                for spg in tpg:  # every spg is a sequence of a video
                    start_frame = spg.start_frame
                    end_frame = spg.end_frame
                    zero_index = 0
                    if start_frame == 0:
                        zero_index = 1
                    # spg.obj_positions could be composed to different objects, then frames
                    obj_count = 0
                    for obj_poses, obj_affordance, obj_object in zip(spg.obj_positions, spg.affordance, spg.objects):
                        affordance_label = obj_affordance
                        skeleton = spg.skeletons
                        object_pos = obj_poses
                        object_label = obj_object
                        if zero_index == 0:
                            skeleton_aligned = cur_aligned_skeleton[start_frame - 1:end_frame, :]
                        else:
                            skeleton_aligned = cur_aligned_skeleton[start_frame:end_frame + 1, :]
                        # print sequence_id, start_frame, end_frame, skeleton_aligned.shape
                        object_label_feature, sequential_geometry_feature, affordance_label_feature = affordance_feature(affordance_label, object_label, object_pos, np.transpose(skeleton_prune(np.transpose(skeleton))), skeleton_aligned)
                        if index == 0:
                            temp_start = 0
                            object_label_feature_subject = object_label_feature.copy()
                            sequential_geometry_feature_subject = sequential_geometry_feature.copy()
                            affordance_label_feature_subject = affordance_label_feature.copy()
                        else:
                            temp_start = len(object_label_feature_subject)
                            object_label_feature_subject = np.concatenate((object_label_feature_subject, object_label_feature), axis=0)
                            sequential_geometry_feature_subject = np.concatenate((sequential_geometry_feature_subject, sequential_geometry_feature), axis=0)
                            affordance_label_feature_subject = np.concatenate((affordance_label_feature_subject, affordance_label_feature), axis=0)
                        temp_end = len(object_label_feature_subject)
                        # frame_count[sequence_id][obj_object].append([temp_start, temp_end])
                        if spg_count == 0:
                            frame_count[sequence_id]['frame_record'].append(list())
                        frame_count[sequence_id]['frame_record'][obj_count].append([temp_start, temp_end])
                        obj_count += 1
                        index += 1
                    spg_count += 1
                print len(object_label_feature_subject), len(sequential_geometry_feature_subject), len(affordance_label_feature_subject)
        np.save(open(feature_path, 'w'), sequential_geometry_feature_subject)
        np.save(open(gt_path, 'w'), affordance_label_feature_subject)
        np.save(open(label_path, 'w'), object_label_feature_subject)
        with open(frame_count_path, 'w') as f:
            json.dump(frame_count, f)
        f.close()


def set_split_subactivity(metadata_root):
    train_path = metadata_root + 'data/train'
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    val_path = metadata_root + 'data/val'
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    x_train_path = train_path + '/img_train.txt'
    x_val_path = val_path + '/img_val.txt'
    y_train_path = train_path + '/subactivity_train.npy'
    y_val_path = val_path + '/subactivity_val.npy'
    bdb_train_path = train_path + '/bdb_train.npy'
    bdb_val_path = val_path + '/bdb_val.npy'
    bf_train_path = train_path + '/bottleneck_feature_train.npy'
    bf_val_path = val_path + '/bottleneck_feature_val.npy'
    sk_train_path = train_path + '/aligned_sk_train.npy'
    sk_val_path = val_path + '/aligned_sk_val.npy'
    sk_sq_train_path = train_path + '/sk_sq_train.npy'
    sk_sq_val_path = val_path + '/sk_sq_val.npy'
    lstm_feature_train_path = train_path + '/subactivity_lstm_feature_train.npy'
    lstm_feature_val_path = val_path + '/subactivity_lstm_feature_val.npy'
    lstm_gt_train_path = train_path + '/subactivity_lstm_gt_train.npy'
    lstm_gt_val_path = val_path + '/subactivity_lstm_gt_val.npy'
    train_set = [1, 3, 4]
    val_set = [5]
    index = 0
    path_train = list()
    for subject_index in train_set:
        subject_path = metadata_root + 'data/subject' + str(subject_index)
        img_path = subject_path + '/' + 'img_path.txt'
        gt_path = subject_path + '/' + 'subactivity_gt.npy'
        bdb_path = subject_path + '/' + 'bdb_gt.npy'
        bf_path = subject_path + '/' + 'bottleneck_feature.npy'
        sk_path = subject_path + '/' + 'aligned_sk.npy'
        sk_sq_path = subject_path + '/' + 'sk_sq.npy'
        lstm_feature_path = subject_path + '/' + 'subactivity_lstm_feature.npy'
        lstm_gt_path = subject_path + '/' + 'subactivity_gt_feature.npy'
        with open(img_path, 'r') as f:
            path_train.extend(f.readlines())
        f.close()
        if index == 0:
            bf_train = np.load(bf_path)
            y_train = np.load(gt_path)
            bdb_train = np.load(bdb_path)
            sk_train = np.load(sk_path)
            sk_sq_train = np.load(sk_sq_path)
            lstm_feature_train = np.load(lstm_feature_path)
            lstm_gt_train = np.load(lstm_gt_path)
        else:
            bf_train = np.concatenate((bf_train, np.load(bf_path)), axis=0)
            y_train = np.concatenate((y_train, np.load(open(gt_path))), axis=0)
            bdb_train = np.concatenate((bdb_train, np.load(open(bdb_path))), axis=0)
            sk_train = np.concatenate((sk_train, np.load(open(sk_path))), axis=0)
            sk_sq_train = np.concatenate((sk_sq_train, np.load(open(sk_sq_path))), axis=0)
            lstm_feature_train = np.concatenate((lstm_feature_train, np.load(lstm_feature_path)), axis=0)
            lstm_gt_train = np.concatenate((lstm_gt_train, np.load(lstm_gt_path)), axis=0)
        index += 1
    index = 0
    path_val = list()
    for subject_index in val_set:
        subject_path = metadata_root + 'data/subject' + str(subject_index)
        img_path = subject_path + '/' + 'img_path.txt'
        gt_path = subject_path + '/' + 'subactivity_gt.npy'
        bdb_path = subject_path + '/' + 'bdb_gt.npy'
        bf_path = subject_path + '/' + 'bottleneck_feature.npy'
        sk_path = subject_path + '/' + 'aligned_sk.npy'
        sk_sq_path = subject_path + '/' + 'sk_sq.npy'
        lstm_feature_path = subject_path + '/' + 'subactivity_lstm_feature.npy'
        lstm_gt_path = subject_path + '/' + 'subactivity_gt_feature.npy'
        with open(img_path, 'r') as f:
            path_val.extend(f.readlines())
        f.close()
        if index == 0:
            bf_val = np.load(bf_path)
            y_val = np.load(gt_path)
            bdb_val = np.load(bdb_path)
            sk_val = np.load(sk_path)
            sk_sq_val = np.load(sk_sq_path)
            lstm_feature_val = np.load(lstm_feature_path)
            lstm_gt_val = np.load(lstm_gt_path)
        else:
            bf_val = np.concatenate((bf_val, np.load(bf_path)), axis=0)
            y_val = np.concatenate((y_val, np.load(open(gt_path))), axis=0)
            bdb_val = np.concatenate((bdb_val, np.load(open(bdb_path))), axis=0)
            sk_val = np.concatenate((sk_val, np.load(open(sk_path))), axis=0)
            sk_sq_val = np.concatenate((sk_sq_val, np.load(open(sk_sq_path))), axis=0)
            lstm_feature_val = np.concatenate((lstm_feature_val, np.load(lstm_feature_path)), axis=0)
            lstm_gt_val = np.concatenate((lstm_gt_val, np.load(lstm_gt_path)), axis=0)
        index += 1
    num_train = len(path_train)
    num_val = len(path_val)
    # num_train = int(num_frame * 0.8)
    # random_index = np.random.permutation(num_frame)
    # train_index = random_index[:num_train]
    # val_index = random_index[num_train:]
    with open(x_train_path, 'w') as f:
        for item in range(num_train):
            print >>f, path_train[item].split('\n')[-2]
    f.close()
    with open(x_val_path, 'w') as f:
        for item in range(num_val):
            print >>f, path_val[item].split('\n')[-2]
    f.close()
    np.save(open(y_train_path, 'w'), [y_train[item] for item in range(num_train)])
    np.save(open(y_val_path, 'w'), [y_val[item] for item in range(num_val)])
    np.save(open(bdb_train_path, 'w'), [bdb_train[item] for item in range(num_train)])
    np.save(open(bdb_val_path, 'w'), [bdb_val[item] for item in range(num_val)])
    np.save(open(bf_train_path, 'w'), [bf_train[item] for item in range(num_train)])
    np.save(open(bf_val_path, 'w'), [bf_val[item] for item in range(num_val)])
    np.save(open(sk_train_path, 'w'), [sk_train[item] for item in range(num_train)])
    np.save(open(sk_val_path, 'w'), [sk_val[item] for item in range(num_val)])
    np.save(open(sk_sq_train_path, 'w'), [sk_sq_train[item] for item in range(num_train)])
    np.save(open(sk_sq_val_path, 'w'), [sk_sq_val[item] for item in range(num_val)])
    np.save(open(lstm_feature_train_path, 'w'), lstm_feature_train)
    np.save(open(lstm_feature_val_path, 'w'), lstm_feature_val)
    np.save(open(lstm_gt_train_path, 'w'), lstm_gt_train)
    np.save(open(lstm_gt_val_path, 'w'), lstm_gt_val)
    print len(bf_train), len(bf_val), len(sk_train), len(sk_val), len(sk_sq_train), len(sk_sq_val), len(lstm_feature_train), len(lstm_feature_val)


def set_split_affordance(metadata_root):
    train_path = metadata_root + 'data/train'
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    val_path = metadata_root + 'data/val'
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    y_train_path = train_path + '/affordance_gt_train.npy'
    y_val_path = val_path + '/affordance_gt_val.npy'
    x_1_train_path = train_path + '/affordance_sequential_feature_train.npy'
    x_1_val_path = val_path + '/affordance_sequential_feature_val.npy'
    x_2_train_path = train_path + '/affordance_object_label_feature_train.npy'
    x_2_val_path = val_path + '/affordance_object_label_feature_val.npy'
    train_set = [1, 3, 4]
    val_set = [5]
    index = 0
    for subject_index in train_set:
        subject_path = metadata_root + 'data/subject' + str(subject_index)
        gt_path = subject_path + '/' + 'affordance_gt.npy'
        feature_path = subject_path + '/' + 'affordance_sequential_feature.npy'
        label_path = subject_path + '/' + 'affordance_object_label_feature.npy'
        if index == 0:
            y_train = np.load(gt_path)
            feature_train = np.load(feature_path)
            label_train = np.load(label_path)
        else:
            y_train = np.concatenate((y_train, np.load(gt_path)), axis=0)
            feature_train = np.concatenate((feature_train, np.load(feature_path)), axis=0)
            label_train = np.concatenate((label_train, np.load(label_path)), axis=0)
        index += 1
    index = 0
    for subject_index in val_set:
        subject_path = metadata_root + 'data/subject' + str(subject_index)
        gt_path = subject_path + '/' + 'affordance_gt.npy'
        feature_path = subject_path + '/' + 'affordance_sequential_feature.npy'
        label_path = subject_path + '/' + 'affordance_object_label_feature.npy'
        if index == 0:
            y_val = np.load(gt_path)
            feature_val = np.load(feature_path)
            label_val = np.load(label_path)
        else:
            y_val = np.concatenate((y_val, np.load(gt_path)), axis=0)
            feature_val = np.concatenate((feature_val, np.load(feature_path)), axis=0)
            label_val = np.concatenate((label_val, np.load(label_path)), axis=0)
        index += 1
    num_train = len(y_train)
    num_val = len(y_val)
    # feature_train[:] = feature_train[:] - np.mean(feature_train, axis=0)
    # feature_val[:] = feature_val[:] - np.mean(feature_val, axis=0)
    np.save(open(y_train_path, 'w'), [y_train[item] for item in range(num_train)])
    np.save(open(y_val_path, 'w'), [y_val[item] for item in range(num_val)])
    np.save(open(x_1_train_path, 'w'), [feature_train[item] for item in range(num_train)])
    np.save(open(x_1_val_path, 'w'), [feature_val[item] for item in range(num_val)])
    np.save(open(x_2_train_path, 'w'), [label_train[item] for item in range(num_train)])
    np.save(open(x_2_val_path, 'w'), [label_val[item] for item in range(num_val)])
    print len(y_train), len(y_val), len(feature_train), len(feature_val), len(label_train), len(label_val)


def subactivity_baseline_svm(metadata_root):
    def get_f1_score(precision, recall):
        return 2 * (precision * recall) / (precision + recall)

    train_path = metadata_root + 'data/train'
    val_path = metadata_root + 'data/val'
    x_sk_sq_train_path = train_path + '/sk_sq_train.npy'
    x_sk_sq_val_path = val_path + '/sk_sq_val.npy'
    y_train_path = train_path + '/subactivity_train.npy'
    y_val_path = val_path + '/subactivity_val.npy'
    y_train = np.load(y_train_path)
    y_train_one = np.zeros(y_train.shape[0])
    y_val = np.load(y_val_path)
    gt_val = np.zeros(y_val.shape[0])
    rand_index = np.random.permutation(len(y_train))
    for i in range(y_train.shape[0]):
        y_train_one[i] = np.argmax(y_train[i, :])
    for i in range(y_val.shape[0]):
        gt_val[i] = np.argmax(y_val[i, :])
    x_sk_sq_train = np.load(x_sk_sq_train_path)
    x_sk_sq_val = np.load(x_sk_sq_val_path)
    clf = SVC(decision_function_shape='ovr')
    clf.fit(x_sk_sq_train[rand_index], y_train_one[rand_index])
    prediction = clf.decision_function(x_sk_sq_val)
    pred = np.zeros(prediction.shape[0])
    for i in range(len(pred)):
        pred[i] = np.argmax(prediction[i, :])
    precision, recall, beta_score, support = sklearn.metrics.precision_recall_fscore_support(gt_val, pred, labels=range(10), average='micro')
    print 'micro result'
    print precision, recall, beta_score, support
    print get_f1_score(precision, recall)
    precision, recall, beta_score, support = sklearn.metrics.precision_recall_fscore_support(gt_val, pred,
                                                                                             labels=range(10),
                                                                                             average='macro')
    print 'macro result'
    print precision, recall, beta_score, support
    print get_f1_score(precision, recall)


def subactivity_train_with_skeleton(metadata_root):
    nb_epoch = 150
    # classes = np.array(range(10))
    train_path = metadata_root + 'data/train'
    val_path = metadata_root + 'data/val'
    model_path = metadata_root + 'models/cnn/'
    # x_1_train_path = train_path + '/bottleneck_feature_train.npy'
    # x_1_val_path = val_path + '/bottleneck_feature_val.npy'
    # x_2_train_path = train_path + '/aligned_sk_train.npy'
    # x_2_val_path = val_path + '/aligned_sk_val.npy'
    x_sk_sq_train_path = train_path + '/sk_sq_train.npy'
    x_sk_sq_val_path = val_path + '/sk_sq_val.npy'
    y_train_path = train_path + '/subactivity_train.npy'
    y_val_path = val_path + '/subactivity_val.npy'
    model_name = 'mixed_feature_last_try_epoch_150_layer_3_with_initialization.h5'
    # x_1_train = np.load(x_1_train_path)
    # x_1_val = np.load(x_1_val_path)
    # x_2_train = np.load(x_2_train_path)
    # x_2_val = np.load(x_2_val_path)
    y_train = np.load(y_train_path)
    y_val = np.load(y_val_path)
    # y = np.zeros(len(y_train))
    # for i in range(len(y_train)):
    #     # print y_train[i, :]
    #     y[i] = int(list(y_train[i, :]).index(1))
    # class_weight = sklearn.utils.compute_class_weight(class_weight='balanced', classes=classes, y=y)
    x_sk_sq_train = np.load(x_sk_sq_train_path)
    x_sk_sq_val = np.load(x_sk_sq_val_path)
    input_dim = x_sk_sq_train.shape[1]
    batch_size = 32
    # left_branch = Sequential()
    # left_branch.add(Dense(64, input_dim=4096))
    right_branch = Sequential()
    right_branch.add(Dense(64, input_dim=33))
    # merged = Merge([left_branch, right_branch], mode='concat')
    final_model = Sequential()
    # final_model.add(merged)
    final_model.add(Dense(512, init=my_init, input_dim=input_dim, activation='relu'))
    final_model.add(Dropout(0.5))
    final_model.add(Dense(128, activation='relu', init=my_init))
    final_model.add(Dropout(0.5))
    final_model.add(Dense(32, activation='relu', init=my_init))
    final_model.add(Dense(10, init=my_init))
    final_model.add(Activation('softmax'))
    optimizer = rmsprop(lr=0.0005)
    final_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    plot(final_model, to_file=model_path + model_name[:-3] + '.png', show_shapes=True)
    final_model.fit(x_sk_sq_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(x_sk_sq_val, y_val))
    final_model.save(model_path + model_name)
    return model_path + model_name


def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.001, name=name)


def sequential_model(input_dim_x1=1596, input_dim_x2=10, weights_path=None):
    left_branch = Sequential()
    left_branch.add(Dense(4096, activation='relu', init=my_init, input_dim=input_dim_x1))
    left_branch.add(Dropout(0.5))
    left_branch.add(Dense(2048, activation='relu', init=my_init))
    left_branch.add(Dropout(0.5))
    left_branch.add(Dense(512, activation='relu', init=my_init))
    right_branch = Sequential()
    right_branch.add(Dense(512, activation='relu', init=my_init, input_dim=input_dim_x2))
    merged = Merge([left_branch, right_branch], mode='concat')
    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(12))
    final_model.add(Activation('softmax'))
    if weights_path:
        final_model.load_weights(weights_path)
    return final_model


def lstm_model(feature_len, nb_classes=10, nodes=512, max_len=10, weights_path=None):
    subactivity_feature = Input(shape=(max_len, feature_len), name='pose2d')
    encode_1 = TimeDistributed(Dense(4096, init=my_init, activation='relu', name='dense_1'))(subactivity_feature)
    dp_1 = TimeDistributed(Dropout(0.5, name='dp_1'))(encode_1)
    encode_2 = TimeDistributed(Dense(2048, init=my_init, activation='relu', name='dense_2'))(dp_1)
    dp_2 = TimeDistributed(Dropout(0.5, name='dp_2'))(encode_2)
    encode_3 = TimeDistributed(Dense(512, init=my_init, activation='relu', name='dense_3'))(dp_2)
    encode_4 = LSTM(nodes, return_sequences=True, name='lstm_1')(encode_3)
    # final_depth = TimeDistributed(Dense(1,init=my_init,activation='sigmoid'))(encode_4)
    # final_depth = TimeDistributed(Dense(1, init=my_init))(encode_4)
    # final_depth1 = Reshape((14,))(final_depth)
    # encode_5 = TimeDistributed(Dense(128, init=my_init, activation='relu', name='dense_4'))(encode_4)
    # dp3 = TimeDistributed(Dropout(0.5, name='dp_3'))(encode_5)
    output = TimeDistributed(Dense(10, init=my_init, activation='softmax'))(encode_4)
    final_output = Reshape((max_len, nb_classes))(output)
    model = Model(input=subactivity_feature, output=final_output)
    if weights_path:
        model.load_weights(weights_path)
    return model


def subactivity_train_lstm(metadata_root):
    nb_epoch = 100
    nb_classes = 10
    batch_size = 32
    train_path = metadata_root + 'data/train'
    val_path = metadata_root + 'data/val'
    model_path = metadata_root + 'models/cnn/'
    x_train_path = train_path + '/subactivity_lstm_feature_train.npy'
    x_val_path = val_path + '/subactivity_lstm_feature_val.npy'
    y_train_path = train_path + '/subactivity_lstm_gt_train.npy'
    y_val_path = val_path + '/subactivity_lstm_gt_val.npy'
    model_name = 'subactivity_lstm_epoch_100_sequencelen_50.h5'
    print 'loading the data'
    x_train = np.load(x_train_path)
    x_val = np.load(x_val_path)
    y_train = np.load(y_train_path)
    y_val = np.load(y_val_path)
    print 'successful initializing the model'
    final_model = lstm_model(x_train.shape[2], max_len=50)
    optimizer = rmsprop(lr=0.001)
    print 'compiling'
    final_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print 'saving the model figure'
    plot(final_model, to_file=model_path + model_name[:-3] + '.png', show_shapes=True)
    print 'fitting'
    final_model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    validation_data=(x_val, y_val))
    final_model.save(model_path + model_name)


def affordance_train_with_skeleton(metadata_root):
    nb_epoch = 30
    classes = np.array(range(12))
    train_path = metadata_root + 'data/train'
    val_path = metadata_root + 'data/val'
    model_path = metadata_root + 'models/cnn/'
    x_1_train_path = train_path + '/affordance_sequential_feature_train.npy'
    x_1_val_path = val_path + '/affordance_sequential_feature_val.npy'
    x_2_train_path = train_path + '/affordance_object_label_feature_train.npy'
    x_2_val_path = val_path + '/affordance_object_label_feature_val.npy'
    y_train_path = train_path + '/affordance_gt_train.npy'
    y_val_path = val_path + '/affordance_gt_val.npy'
    model_name = 'affordance_mixed_feature_epoch_30_with_dropout_3_layer_with_weight_1.4_with_initialization_weight_1.h5'
    x_1_train = np.load(x_1_train_path)
    x_1_val = np.load(x_1_val_path)
    x_2_train = np.load(x_2_train_path)
    x_2_val = np.load(x_2_val_path)
    y_train = np.load(y_train_path)
    y_val = np.load(y_val_path)
    # y = np.zeros(len(y_train))
    # for i in range(len(y_train)):
    #     y[i] = int(list(y_train[i, :]).index(1))
    #     # print y[i]
    # class_weight = sklearn.utils.compute_class_weight(class_weight='balanced', classes=classes, y=y)
    # print class_weight
    # class_weight = {0: 1.15, 1: 0.69, 2: 4.14, 3: 4.14, 4: 8.62, 5: 7.12, 6: 2.23, 7: 1.53, 8: 4.18, 9: 6.06, 10: 6.06, 11: 0.14}
    # class_weight[11] *= 2
    input_dim_x1 = x_1_train.shape[1]
    input_dim_x2 = x_2_train.shape[1]
    batch_size = 32
    left_branch = Sequential()
    left_branch.add(Dense(4096, activation='relu', init=my_init, input_dim=input_dim_x1))
    left_branch.add(Dropout(0.5))
    left_branch.add(Dense(2048, activation='relu', init=my_init))
    left_branch.add(Dropout(0.5))
    left_branch.add(Dense(512, activation='relu', init=my_init))
    right_branch = Sequential()
    right_branch.add(Dense(512, activation='relu', init=my_init, input_dim=input_dim_x2))
    merged = Merge([left_branch, right_branch], mode='concat')
    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(12))
    final_model.add(Activation('softmax'))
    optimizer = rmsprop(lr=0.0001)
    final_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    plot(final_model, to_file=model_path + model_name[:-3] + '.png', show_shapes=True)
    final_model.fit([x_1_train, x_2_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=([x_1_val, x_2_val], y_val))
    final_model.save(model_path + model_name)
    return model_path + model_name


def fine_tune(metadata_root):
    nb_classes = 10
    nb_epoch = 50
    train_path = metadata_root + 'data/train'
    val_path = metadata_root + 'data/val'
    model_path = metadata_root + 'models/cnn/'
    x_train_path = train_path + '/img_train.txt'
    x_val_path = val_path + '/img_val.txt'
    y_train_path = train_path + '/subactivity_train.npy'
    y_val_path = val_path + '/subactivity_val.npy'
    bdb_train_path = train_path + '/bdb_train.npy'
    bdb_val_path = val_path + '/bdb_val.npy'
    batch_size = 32
    nb_train = np.load(y_train_path).shape[0]
    nb_val = np.load(y_val_path).shape[0]
    train_generator = img_from_list(batch_size, x_train_path, y_train_path, bdb_train_path, nb_classes)
    val_generator = img_from_list(batch_size, x_val_path, y_val_path, bdb_val_path, nb_classes)
    model = vgg_16(model_path + 'vgg16_weights.h5')
    for layer in model.layers[:25]:
        layer.trainable = False
    model.pop()
    model.add(Dense(nb_classes, activation='softmax'))
    model_name = 'vgg_tune_subactivity_train_134_learning_rate_-5_.h5'
    early_stopping = EarlyStopping(verbose=1, patience=30, monitor='acc')
    model_checkpoint = ModelCheckpoint(
        model_path + model_name, save_best_only=True,
        save_weights_only=True,
        monitor='acc')
    callbacks_list = [early_stopping, model_checkpoint]
    plot(model, to_file=model_path + model_name[:-3] + '.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])
    model.fit_generator(train_generator, samples_per_epoch=(nb_train//batch_size)*batch_size, nb_epoch=nb_epoch, validation_data=val_generator, nb_val_samples=(nb_val//batch_size)*batch_size, callbacks=callbacks_list)
    model.save(model_path + model_name)


def main():
    np.random.seed(1000)
    paths = config.Paths()
    paths.path_huang()
    # data_prepare_subactivity(paths.data_root, paths.metadata_root)
    # set_split_subactivity(paths.metadata_root)
    # fine_tune(paths.metadata_root)
    # subactivity_train_with_skeleton(paths.metadata_root)
    # data_prepare_affordance(paths.data_root, paths.metadata_root)
    # set_split_affordance(paths.metadata_root)
    # affordance_train_with_skeleton(paths.metadata_root)
    subactivity_train_lstm(paths.metadata_root)
    # data_prepare_subactivity_cad(paths.data_root, paths.metadata_root, trainable=0)
    # data_prepare_affordance_cad(paths.data_root, paths.metadata_root, trainable=0)
    # subactivity_baseline_svm(paths.metadata_root)


if __name__ == '__main__':
    main()
