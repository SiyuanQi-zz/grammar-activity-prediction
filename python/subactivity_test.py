import config
import json
import glob
import numpy as np
import scipy.io
from keras.preprocessing import sequence
import vizutil
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout
from keras.layers import LSTM
from keras.models import load_model
from subactivity_lstm import skeleton_prune
from vgg_fine_tune import img_preprocessing_bdb, vgg_16, img_from_list
import os
import matplotlib.pyplot as plt
from vgg_fine_tune import my_init, lstm_model
import sklearn.metrics


def img_from_list_test(batch_size, x_path, bdb_path):
    x_file = list()
    with open(x_path, 'r') as f:
        x_file.extend(f.readlines())
    f.close()
    bdb_file = np.load(bdb_path).astype(int)
    while 1:
        for i in range(len(x_file) // batch_size):
            x_batch = np.zeros((batch_size, 3, 224, 224))
            for j in range(batch_size):
                x_batch[j] = img_preprocessing_bdb(x_file[i * batch_size + j].split('\n')[-2], bdb_file[i * batch_size + j, :])
            yield x_batch


def without_segmentation_sequence_test(metadata_path, model_path, maxlen=200):
    data_dim = 33
    label_list = ['null', 'reaching', 'moving', 'placing', 'opening', 'cleaning', 'closing', 'pouring', 'eating', 'drinking']
    path_skeleton = metadata_path + 'aligned_skeletons/'
    path_label = metadata_path + 'sequence_label.json'
    with open(path_label, 'r') as f:
        sequence_label = json.load(f)
    skeleton_list = glob.glob(path_skeleton + '*.mat')
    video_index = 5
    sequence_id = skeleton_list[video_index].split('/')[-1][:-4]
    cur_skeleton = np.transpose(scipy.io.loadmat(skeleton_list[video_index])['skeleton'])
    num_frame = 0
    for sequence_cut in sequence_label:
        if sequence_id == sequence_cut['sequence_id']:
            if sequence_cut['end_frame'] > num_frame:
                num_frame = sequence_cut['end_frame']
    y_all = np.zeros(num_frame)
    x_all = np.zeros((num_frame, maxlen, data_dim))
    for sequence_cut in sequence_label:
        if sequence_id == sequence_cut['sequence_id']:
            start_frame = sequence_cut['start_frame']
            end_frame = sequence_cut['end_frame']
            # print label_list.index(str(sequence_cut['sequence_label']))
            y_all[start_frame:end_frame] = label_list.index(sequence_cut['sequence_label'])
    for frame in range(num_frame):
        if frame < maxlen:
            # x_temp = np.zeros((data_dim, frame+1))
            x_temp = skeleton_prune(cur_skeleton[:, :frame+1])
            x_temp = np.reshape(x_temp, (data_dim, frame+1))
        else:
            x_temp = skeleton_prune(cur_skeleton[:, frame-(maxlen-1):frame+1])
        # print type(x_temp)
        x_all[frame, :, :] = np.transpose(sequence.pad_sequences(x_temp, dtype=float, maxlen=maxlen))
    model = load_model(model_path)
    prediction = model.predict(x_all)
    predict_result = np.zeros(num_frame)
    for i in range(num_frame):
        predict_result[i] = int(list(prediction[i, :]).index(max(prediction[i, :])))
        print label_list[int(y_all[i])], label_list[int(predict_result[i])]
    vizutil.plot_segmentation([y_all, predict_result, (y_all - predict_result) == 0], frame)
    # print predict_result - y_all


def without_segmentation_sequence_test_per_frame(metadata_path, model_path):
    data_dim = 33
    label_list = ['null', 'reaching', 'moving', 'placing', 'opening', 'cleaning', 'closing', 'pouring', 'eating',
                  'drinking']
    path_skeleton = metadata_path + 'aligned_skeletons/'
    path_label = metadata_path + 'sequence_label.json'
    with open(path_label, 'r') as f:
        sequence_label = json.load(f)
    skeleton_list = glob.glob(path_skeleton + '*.mat')
    video_index = 60
    sequence_id = skeleton_list[video_index].split('/')[-1][:-4]
    cur_skeleton = np.transpose(scipy.io.loadmat(skeleton_list[video_index])['skeleton'])
    num_frame = 0
    for sequence_cut in sequence_label:
        if sequence_id == sequence_cut['sequence_id']:
            if sequence_cut['end_frame'] > num_frame:
                num_frame = sequence_cut['end_frame']
    y_all = np.zeros(num_frame)
    x_all = np.zeros((num_frame, data_dim))
    for sequence_cut in sequence_label:
        if sequence_id == sequence_cut['sequence_id']:
            start_frame = sequence_cut['start_frame']
            end_frame = sequence_cut['end_frame']
            y_all[start_frame:end_frame] = label_list.index(sequence_cut['sequence_label'])
    for frame in range(num_frame):
        # print frame
        x_temp = skeleton_prune(cur_skeleton[:, frame])
        x_all[frame, :] = x_temp
    model = load_model(model_path)
    prediction = model.predict(x_all)
    predict_result = np.zeros(num_frame)
    correct_num = 0
    for i in range(num_frame):
        predict_result[i] = int(list(prediction[i, :]).index(max(prediction[i, :])))
        if predict_result[i] == int(y_all[i]):
            correct_num += 1
        print i, label_list[int(y_all[i])], label_list[int(predict_result[i])]
    vizutil.plot_segmentation([y_all, predict_result, (y_all - predict_result) == 0], frame)
    print 'accuracy', float(correct_num) / num_frame


def get_bottleneck_feature(metadata_path):
    nb_classes = 10
    batch_size = 1
    model_path = metadata_path + 'models/cnn/'
    model_name = 'vgg_tune_subactivity_train_134_learning_rate_-5_.h5'
    model = vgg_16(model_path + model_name, nb_classes)
    model.pop()
    print 'successful loading the model!'
    test_set = [1, 3, 4, 5]
    for subject_index in test_set:
        path_test = list()
        index = 0
        subject_path = metadata_path + 'data/subject' + str(subject_index)
        img_path = subject_path + '/' + 'img_path.txt'
        bdb_path = subject_path + '/' + 'bdb_gt.npy'
        with open(img_path, 'r') as f:
            path_test.extend(f.readlines())
        f.close()
        index += 1
        num_frame = len(path_test)
        print num_frame - (num_frame // batch_size) * batch_size
        test_generator = img_from_list_test(batch_size, img_path, bdb_path)
        prediction = model.predict_generator(test_generator, val_samples=num_frame)
        print 'successful predicting the data!'
        np.save(open(subject_path + '/bottleneck_feature.npy', 'w'), prediction)


def without_segmentation_sequence_test_per_frame_vgg16(metadata_path):
    nb_classes = 10
    batch_size = 1
    test_path = metadata_path + 'data/test'
    model_path = metadata_path + 'models/cnn/'
    model_name = 'vgg_tune_subactivity_train_134_learning_rate_-5_.h5'
    result_path = metadata_path + 'data/subactivity_result/'
    model = vgg_16(model_path + model_name, nb_classes)
    print 'successful loading the model!'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    test_set = [5]
    for subject_index in test_set:
        path_test = list()
        index = 0
        subject_path = metadata_path + 'data/subject' + str(subject_index)
        img_path = subject_path + '/' + 'img_path.txt'
        gt_path = subject_path + '/' + 'subactivity_gt.npy'
        bdb_path = subject_path + '/' + 'bdb_gt.npy'
        frame_count_path = subject_path + '/' + 'frame_count.json'
        with open(img_path, 'r') as f:
            path_test.extend(f.readlines())
        f.close()
        if index == 0:
            label_all = np.load(gt_path)
        else:
            label_all = np.concatenate((label_all, np.load(open(gt_path))), axis=0)
        index += 1
        num_frame = len(path_test)
        print num_frame - (num_frame // batch_size) * batch_size
        y_all = np.zeros(num_frame)
        test_generator = img_from_list_test(batch_size, img_path, bdb_path)
        prediction = model.predict_generator(test_generator, val_samples=num_frame)
        predict_result = np.zeros(num_frame)
        print 'successful predicting the data!'
        for i in range(num_frame):
            predict_result[i] = int(list(prediction[i, :]).index(max(prediction[i, :])))
            y_all[i] = np.argmax(label_all[i, :])
        with open(frame_count_path, 'r') as f:
            video_count = json.load(f)
        for video in video_count:
            print video
            correct_num = 0
            start_num = video_count[video]['start_num']
            end_num = video_count[video]['end_num']
            for j in range(end_num - start_num):
                if int(predict_result[start_num + j]) == int(y_all[start_num + j]):
                    correct_num += 1
            vizutil.plot_segmentation([y_all[start_num:end_num], predict_result[start_num:end_num], (y_all[start_num:end_num] - predict_result[start_num:end_num]) == 0], end_num - start_num)
            np.save(open(result_path + 'subject' + str(subject_index) + '/' + video + '.npy', 'w'), prediction[start_num:end_num, :])
            plt.savefig(result_path + 'subject' + str(subject_index) + '/' + video + '_' + str(float(correct_num)/(end_num - start_num)) + '.png')
            plt.close()
        precision, recall, beta_score, support = sklearn.metrics.precision_recall_fscore_support(y_all, predict_result,
                                                                                                 labels=range(10),
                                                                                                 average='micro')
        print 'micro result'
        print precision, recall, beta_score, support
        print get_f1_score(precision, recall)
        precision, recall, beta_score, support = sklearn.metrics.precision_recall_fscore_support(y_all, predict_result,
                                                                                                 labels=range(10),
                                                                                                 average='macro')
        print 'macro result'
        print precision, recall, beta_score, support
        print get_f1_score(precision, recall)


def without_segmentation_sequence_test_per_frame_sequential(metadata_path):
    label_list = ['null', 'reaching', 'moving', 'placing', 'opening', 'cleaning', 'closing', 'pouring', 'eating',
                  'drinking']
    test_path = metadata_path + 'data/test'
    model_path = metadata_path + 'models/cnn/'
    model_name = 'mixed_feature_last_try_epoch_150_layer_3_with_initialization.h5'
    result_path = metadata_path + 'data/subactivity_result/'
    model = Sequential()
    # final_model.add(merged)
    model.add(Dense(512, init=my_init, input_dim=1452, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', init=my_init))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', init=my_init))
    model.add(Dense(10, init=my_init))
    model.add(Activation('softmax'))
    model.load_weights(model_path + model_name)
    print 'successful loading the model!'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    test_set = [1, 3, 4, 5]
    for subject_index in test_set:
        path_test = list()
        subject_path = metadata_path + 'data/subject' + str(subject_index)
        img_path = subject_path + '/' + 'img_path.txt'
        gt_path = subject_path + '/' + 'subactivity_gt.npy'
        frame_count_path = subject_path + '/' + 'frame_count.json'
        sk_sq_path = subject_path + '/' + 'sk_sq.npy'
        with open(img_path, 'r') as f:
            path_test.extend(f.readlines())
        f.close()
        label_all = np.load(gt_path)
        num_frame = len(path_test)
        y_all = np.zeros(num_frame)
        sk_sq = np.load(sk_sq_path)
        prediction = model.predict(sk_sq)
        predict_result = np.zeros(num_frame)
        print 'successful predicting the data!'
        for i in range(num_frame):
            predict_result[i] = int(list(prediction[i, :]).index(max(prediction[i, :])))
            y_all[i] = np.argmax(label_all[i, :])
        with open(frame_count_path, 'r') as f:
            video_count = json.load(f)
        for video in video_count:
            print video
            correct_num = 0
            start_num = video_count[video]['start_num']
            end_num = video_count[video]['end_num']
            for j in range(end_num - start_num):
                if int(predict_result[start_num + j]) == int(y_all[start_num + j]):
                    correct_num += 1
            vizutil.plot_segmentation([y_all[start_num:end_num], predict_result[start_num:end_num], (y_all[start_num:end_num] - predict_result[start_num:end_num]) == 0], end_num - start_num)
            plt.savefig(result_path + 'subject' + str(subject_index) + '/' + video + '_' + str(float(correct_num)/(end_num - start_num)) + '.png')
            plt.close()
            cm = sklearn.metrics.confusion_matrix(y_all[start_num:end_num], predict_result[start_num:end_num], labels=range(10))
            vizutil.plot_confusion_matrix(cm, classes=label_list, normalize=True, filename=result_path + 'subject' + str(
                subject_index) + '/' + video + '_confusion.png')
            np.save(open(result_path + 'subject' + str(subject_index) + '/' + video + '.npy', 'w'), prediction[start_num:end_num, :])
        cm = sklearn.metrics.confusion_matrix(y_all, predict_result,
                                              labels=range(10))
        vizutil.plot_confusion_matrix(cm, classes=label_list, normalize=True, filename=result_path + 'subject' + str(
            subject_index) + '/' + 'a_confusion_all.png')


def get_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def without_segmentation_sequence_test_per_frame_lstm(data_root, metadata_root):
    label_list = ['null', 'reaching', 'moving', 'placing', 'opening', 'cleaning', 'closing', 'pouring', 'eating',
                  'drinking']
    model_path = metadata_root + 'models/cnn/'
    model_name = 'subactivity_lstm_epoch_100_sequencelen_50.h5'
    max_len = 50
    predict_len = 42
    feature_len = 363
    subject_set = [1, 3, 4, 5]
    test_set = [5]
    result_path = metadata_root + 'data/subactivity_result/'
    print 'loading the model'
    model = lstm_model(feature_len=feature_len, max_len=max_len, weights_path=model_path+model_name)
    gt_all = np.empty(0)
    pre_all = np.empty(0)
    for subject_index in test_set:
        print 'subject' + str(subject_index)
        subject_path = metadata_root + 'data/subject' + str(subject_index)
        frame_count_path = subject_path + '/' + 'frame_count.json'
        with open(frame_count_path, 'r') as f:
            frame_count = json.load(f)
        subject = 'Subject' + str(subject_index) + '_rgbd_images/'
        action = sorted(os.listdir(data_root + subject))
        segment_len = 50
        padding_len = 5
        subactivity_feature_path = subject_path + '/' + 'subactivity_lstm_feature.npy'
        subactivity_gt_path = subject_path + '/' + 'subactivity_gt_feature.npy'
        lstm_feature = np.load(subactivity_feature_path)
        lstm_gt = np.load(subactivity_gt_path)
        print 'predicting'
        prediction = model.predict(lstm_feature)
        predict_index = 0
        for action_category in action:
            video = sorted(os.listdir(data_root + subject + action_category))
            for sequence_id in video:
                start_frame = frame_count[sequence_id]['start_num']
                end_frame = frame_count[sequence_id]['end_num']
                video_length = end_frame - start_frame - predict_len
                print sequence_id, video_length
                predict_score = [list() for _ in range(video_length)]
                predict_score_output = np.zeros((video_length, 10))
                predict_result = np.zeros(video_length)
                gt_result = np.zeros(video_length)
                i = 0
                while 1:
                    prediction_temp = prediction[predict_index, :, :]
                    gt_temp = lstm_gt[predict_index, :, :]
                    # print i, predict_index
                    for j in range(max_len):
                        if len(predict_score[i*padding_len+j]) == 0:
                            predict_score[i*padding_len+j] = np.expand_dims(prediction_temp[j, :], axis=0)
                        else:
                            predict_score[i*padding_len+j] = np.concatenate((predict_score[i*padding_len+j], np.expand_dims(prediction_temp[j, :], axis=0)), axis=0)
                        gt_result[i*padding_len+j] = np.argmax(gt_temp[j])
                    i += 1
                    predict_index += 1
                    if i * padding_len + segment_len > video_length:
                        break
                for frame_index in range(video_length):
                    if len(predict_score[frame_index]) != 0:
                        predict_score[frame_index] = np.mean(predict_score[frame_index], axis=0)
                        predict_result[frame_index] = np.argmax(predict_score[frame_index])
                if i*padding_len + segment_len > video_length and (i-1)*padding_len + segment_len != video_length:
                    predict_result[(i - 1) * padding_len + segment_len:] = predict_result[
                        (i - 1) * padding_len + segment_len - 1]
                    for k in range((i-1)*padding_len+segment_len, video_length):
                        predict_score[k] = predict_score[(i-1)*padding_len + segment_len - 1]
                    gt_result[(i - 1) * padding_len + segment_len:] = gt_result[
                        (i - 1) * padding_len + segment_len - 1]
                # vizutil.plot_segmentation([gt_result, predict_result,
                #                            (gt_result - predict_result) == 0],
                #                           video_length)
                for frame_index in range(video_length):
                    # print frame_index, video_length, np.array(predict_score[frame_index])
                    predict_score_output[frame_index, :] = np.array(predict_score[frame_index])
                # np.save(open(result_path + 'subject' + str(subject_index) + '/' + sequence_id + '.npy', 'w'),
                #         predict_score_output)
                # correct_num = 0
                # for frame_index in range(video_length):
                #     if gt_result[frame_index] == predict_result[frame_index]:
                #         correct_num += 1
                # cm = sklearn.metrics.confusion_matrix(gt_result, predict_result, labels=range(10))
                # vizutil.plot_confusion_matrix(cm, classes=label_list, filename=result_path + 'subject' + str(subject_index) + '/' + sequence_id + '_confusion.png')
                # plt.savefig(result_path + 'subject' + str(subject_index) + '/' + sequence_id + '_' + str(
                #     float(correct_num) / video_length) + '.png')
                # plt.close()
                gt_all = np.hstack((gt_all, gt_result))
                pre_all = np.hstack((pre_all, predict_result))
        precision, recall, beta_score, support = sklearn.metrics.precision_recall_fscore_support(gt_all, pre_all,
                                                                                                 labels=range(10),
                                                                                                 average='micro')
        print 'micro result'
        print precision, recall, beta_score, support
        print get_f1_score(precision, recall)
        precision, recall, beta_score, support = sklearn.metrics.precision_recall_fscore_support(gt_all, pre_all,
                                                                                                 labels=range(10),
                                                                                                 average='macro')
        print 'macro result'
        print precision, recall, beta_score, support
        print get_f1_score(precision, recall)


def main():
    paths = config.Paths()
    paths.path_huang()
    # model_path = paths.metadata_root + 'models/flipped/layer_3_without_dropout_maxlen_200_epoch_200.h5'
    # model_path_per_frame = paths.metadata_root + 'models/flipped/layer_3_per_frame_without_dropout_epoch_1000.h5'
    # without_segmentation_sequence_test(paths.metadata_root + 'flipped/', model_path, 200)
    # without_segmentation_sequence_test_per_frame(paths.metadata_root + 'flipped/', model_path_per_frame)
    # without_segmentation_sequence_test_per_frame_vgg16(paths.metadata_root)
    # get_bottleneck_feature(paths.metadata_root)
    # without_segmentation_sequence_test_per_frame_sequential(paths.metadata_root)
    without_segmentation_sequence_test_per_frame_lstm(paths.data_root, paths.metadata_root)


if __name__ == '__main__':
    main()