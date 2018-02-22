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
import pickle
import matplotlib.pyplot as plt
from vgg_fine_tune import sequential_model


def fill_in(m, nb_classes, prob_given):
    label = np.zeros((m, nb_classes))
    for i in range(m):
        for j in range(nb_classes):
            if j == 11:
                label[i, j] = prob_given
            else:
                label[i, j] = float(1 - prob_given) / (nb_classes - 1 )
    return label


def without_segmentation_sequence_test_per_frame_sequential(data_root, metadata_path):
    model_path = metadata_path + 'models/cnn/'
    relative_path = 'flipped/all/activity_corpus.p'
    if os.path.exists(metadata_path + relative_path):
        activity_corpus = pickle.load(open(metadata_path + relative_path, 'rb'))
    model_name = 'affordance_mixed_feature_epoch_30_with_dropout_3_layer_with_weight_1.4_with_initialization.h5'
    result_path = metadata_path + 'data/affordance_result/'
    tpg_id = dict()
    for activity, tpgs in activity_corpus.items():
        for tpg in tpgs:
            tpg_id[tpg.id] = tpg.terminals
    print 'successful loading the model!'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    test_set = [1, 3, 4, 5]
    model = sequential_model(weights_path=model_path + model_name)
    for subject_index in test_set:
        if not os.path.exists(result_path + 'subject' + str(subject_index)):
            os.mkdir(result_path + 'subject' + str(subject_index))
        subject_path = metadata_path + 'data/subject' + str(subject_index)
        frame_count_path = subject_path + '/' + 'affordance_frame_count.json'
        subject = 'Subject' + str(subject_index) + '_rgbd_images/'
        action = os.listdir(data_root + subject)
        gt_path = subject_path + '/' + 'affordance_gt.npy'
        feature_path = subject_path + '/' + 'affordance_sequential_feature.npy'
        label_path = subject_path + '/' + 'affordance_object_label_feature.npy'
        gt_all = np.load(gt_path)
        feature_all = np.load(feature_path)
        label_all = np.load(label_path)
        with open(frame_count_path, 'r') as f:
            frame_count = json.load(f)
        f.close()
        prediction = model.predict([feature_all, label_all])
        for action_category in action:
            video = os.listdir(data_root + subject + action_category)
            for sequence_id in video:
                video_prediction = list()
                video_gt = list()
                index = 0
                num_obj = len(frame_count[sequence_id]['object'])
                sequence_prediction = np.zeros((num_obj, frame_count[sequence_id]['length'], 12))
                for obj_index in frame_count[sequence_id]['frame_record']:
                    add_index = 0
                    for sequence_list in obj_index:
                        if add_index == 0:
                            video_prediction.append(prediction[sequence_list[0]:sequence_list[1], :])
                            video_gt.append(gt_all[sequence_list[0]:sequence_list[1], :])
                        else:
                            video_prediction[index] = np.concatenate((video_prediction[index], prediction[sequence_list[0]:sequence_list[1], :]), axis=0)
                            video_gt[index] = np.concatenate((video_gt[index], gt_all[sequence_list[0]:sequence_list[1], :]), axis=0)
                        add_index += 1
                    frame_length = frame_count[sequence_id]['length']
                    if frame_length - video_gt[index].shape[0] != 0:
                        video_prediction[index] = np.concatenate((fill_in(frame_length - video_prediction[index].shape[0], 12, 0.8), video_prediction[index]), axis=0)
                        video_gt[index] = np.concatenate((fill_in(frame_length - video_gt[index].shape[0], 12, 0.8), video_gt[index]), axis=0)
                    predict_result = np.zeros(frame_length)
                    y_all = np.zeros(frame_length)
                    correct_num = 0
                    for i in range(frame_length):
                        predict_result[i] = int(list(video_prediction[index][i, :]).index(max(video_prediction[index][i, :])))
                        y_all[i] = np.argmax(video_gt[index][i, :])
                        if y_all[i] == predict_result[i]:
                            correct_num += 1
                    vizutil.plot_segmentation([y_all, predict_result, (y_all - predict_result) == 0], frame_length)
                    plt.savefig(result_path + 'subject' + str(subject_index) + '/' + sequence_id + '_' + str(frame_count[sequence_id]['object'][index]) + str(index) + '_' + str(float(correct_num) / frame_length) + '.png')
                    plt.close()
                    sequence_prediction[index, :, :] = video_prediction[index]
                    index += 1
                np.save(open(result_path + 'subject' + str(subject_index) + '/' + sequence_id + '.npy', 'w'), sequence_prediction)


def main():
    paths = config.Paths()
    paths.path_huang()
    without_segmentation_sequence_test_per_frame_sequential(paths.data_root, paths.metadata_root)


if __name__ == '__main__':
    main()