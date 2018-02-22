"""
Created on Feb 24, 2017

@author: Siyuan Huang

Process the skeleton, get the input for LSTM.

Input: Aligned human skeleton feature.

"""

import config
import json
import scipy.io
import os
import numpy as np


def json_to_mat(paths, flipped=0):
    if flipped == 1:
        dir_data = paths.metadata_root + 'flipped/all/action.json'
    else:
        dir_data = paths.metadata_root + 'action.json'
    with open(dir_data, 'r') as f:
        action = json.load(f)
    # save skeleton to mat file
    if flipped == 1:
        save_skeleton(action, paths.metadata_root + 'flipped/skeletons')
    else:
        save_skeleton(action, paths.metadata_root + 'skeletons')
    sequence_processing(action, paths.metadata_root + 'flipped/sequence_label.json')


def save_skeleton(action, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for sequence, skeleton_pos in action['skeletons'].items():
        action['skeletons'][sequence] = np.asarray(skeleton_pos)
        scipy.io.savemat(path + '/' + sequence + '.mat', mdict={'skeleton': action['skeletons'][sequence]})


def sequence_processing(action, path):
    label = []
    index = 0
    frame_num = 0
    frame_max = 0
    subactivity_category = {}
    for sequence_id, sequence_label in action['skeleton_labels'].items():
        start_frame = 0
        label_temp = 'null'
        for i in range(len(sequence_label)):
            sequence_label[i] = str(sequence_label[i])
            if i == 0:
                start_frame = 0
                label_temp = sequence_label[i]
            elif sequence_label[i] != label_temp or i == len(sequence_label) - 1:
                label.append({})
                label[index]['sequence_id'] = sequence_id
                label[index]['sequence_label'] = label_temp
                label[index]['start_frame'] = start_frame
                if i == len(sequence_label) - 1:
                    label[index]['end_frame'] = i
                else:
                    label[index]['end_frame'] = i-1
                label[index]['length'] = label[index]['end_frame'] - label[index]['start_frame'] + 1
                if label_temp not in subactivity_category:
                    subactivity_category[label_temp] = 1
                else:
                    subactivity_category[label_temp] += 1
                frame_num += label[index]['length']
                start_frame = i
                label_temp = sequence_label[i]
                if label[index]['length'] > frame_max:
                    frame_max = label[index]['length']
                if label[index]['length'] > 100:
                    print label[index]['length'], label[index]['sequence_label'], label[index]['sequence_id']
                index += 1
    # print(frame_num)
    # print(float(frame_num)/len(label))
    print frame_max
    print subactivity_category
    with open(path, 'w') as f:
        json.dump(label, f)
    return label


def main():
    paths = config.Paths()
    paths.path_huang()
    json_to_mat(paths, 1)


if __name__ == '__main__':
    main()
