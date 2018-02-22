"""
Created on Feb 28, 2017

@author: Siyuan Huang

Data Processing for Affordance

"""

import numpy as np
import pickle
import config
import os
import subactivity_lstm
from keras.preprocessing import sequence


def load_affordance_data(path):
    relative_path = 'flipped/all/activity_corpus.p'
    if os.path.exists(path + relative_path):
        activity_corpus = pickle.load(open(path + relative_path, 'rb'))
    affordance = list()
    skeleton = list()
    object_pos = list()
    object_category = list()
    for activity, tpgs in activity_corpus.items():
        for tpg in tpgs:
            for spg in tpg.terminals:
                for obj_poses, obj_affordance, obj_object in zip(spg.obj_positions, spg.affordance, spg.objects):
                    empty = 0
                    for obj_pos in obj_poses:
                        if obj_pos.shape[0] == 0:
                            empty = 1
                    if empty == 1:
                        continue
                    else:
                        affordance.append(obj_affordance)
                        skeleton.append(spg.skeletons)
                        object_pos.append(obj_poses)
                        object_category.append(obj_object)
    return feature_selection_completion(object_category, object_pos, affordance, skeleton)


def feature_selection(obj_category, obj_pos, obj_affordance, skeleton):
    maxlen = 200
    affordance_label = ['reachable', 'movable', 'pourable', 'pourto', 'containable', 'drinkable', 'openable', 'placeable', 'closeable', 'cleanable', 'cleaner', 'stationary']
    object_label = ['medcinebox', 'cup', 'bowl', 'box', 'milk', 'book', 'microwave', 'plate', 'remote', 'cloth']
    num_sample = len(obj_category)
    object_feature = np.zeros((num_sample, 10))
    affordance_feature = np.zeros((num_sample, 12))
    distance_feature = np.zeros((num_sample, maxlen, 11))
    orientation_feature = np.zeros((num_sample, maxlen, 1))
    for i in range(num_sample):
        object_feature[i, object_label.index(obj_category[i])] = 1
        affordance_feature[i, affordance_label.index(obj_affordance[i])] = 1
        num_frame = len(skeleton[i])
        distance_feature_sub = np.zeros((num_frame, 11))
        orientation_feature_sub = np.zeros((num_frame, 1))
        frame = 0
        for skeleton_frame, obj_pos_frame in zip(skeleton[i], obj_pos[i]):
            skeleton_frame_temp = subactivity_lstm.skeleton_prune(skeleton_frame)
            obj_pos_mid = np.mean(obj_pos_frame, axis=0)
            distance_feature_sub[frame, :] = np.array([np.linalg.norm(obj_pos_mid - skeleton_frame_temp[j*3:(j+1)*3]) for j in range(11)])
            n_1 = skeleton_frame_temp[6:9] - skeleton_frame_temp[9:12]
            n_2 = skeleton_frame_temp[6:9] - skeleton_frame_temp[15:18]
            n_normal = np.cross(n_1, n_2)
            n_center = obj_pos_mid - np.mean([skeleton_frame_temp[6:9], skeleton_frame_temp[9:12], skeleton_frame_temp[15:18]], axis=0)
            orientation_feature_sub[frame, :] = np.arccos(np.clip(np.dot(n_normal, n_center) / np.linalg.norm(n_normal) / np.linalg.norm(n_center), -1, 1))
            frame += 1
        distance_feature[i, :, :] = np.transpose(sequence.pad_sequences(np.transpose(distance_feature_sub), dtype=float, maxlen=maxlen))
        orientation_feature[i, :, :] = np.transpose(sequence.pad_sequences(np.transpose(orientation_feature_sub), dtype=float, maxlen=maxlen))
    return object_feature, affordance_feature, distance_feature, orientation_feature


def feature_selection_completion(obj_category, obj_pos, obj_affordance, skeleton):
    maxlen = 200
    affordance_label = ['reachable', 'movable', 'pourable', 'pourto', 'containable', 'drinkable', 'openable', 'placeable', 'closeable', 'cleanable', 'cleaner', 'stationary']
    object_label = ['medcinebox', 'cup', 'bowl', 'box', 'milk', 'book', 'microwave', 'plate', 'remote', 'cloth']
    num_sample = len(obj_category)
    object_feature = np.zeros((num_sample, 10))
    affordance_feature = np.zeros((num_sample, 12))
    normal_feature = np.zeros((num_sample, maxlen, 33))
    orientation_feature = np.zeros((num_sample, maxlen, 1))
    for i in range(num_sample):
        object_feature[i, object_label.index(obj_category[i])] = 1
        affordance_feature[i, affordance_label.index(obj_affordance[i])] = 1
        num_frame = len(skeleton[i])
        normal_feature_sub = np.zeros((num_frame, 33))
        orientation_feature_sub = np.zeros((num_frame, 1))
        frame = 0
        for skeleton_frame, obj_pos_frame in zip(skeleton[i], obj_pos[i]):
            skeleton_frame_temp = subactivity_lstm.skeleton_prune(skeleton_frame)
            obj_pos_mid = np.mean(obj_pos_frame, axis=0)
            normal_feature_sub[frame, :] = np.array([obj_pos_mid - skeleton_frame_temp[j*3:(j+1)*3] for j in range(11)]).flatten()
            n_1 = skeleton_frame_temp[6:9] - skeleton_frame_temp[9:12]
            n_2 = skeleton_frame_temp[6:9] - skeleton_frame_temp[15:18]
            n_normal = np.cross(n_1, n_2)
            n_center = obj_pos_mid - np.mean([skeleton_frame_temp[6:9], skeleton_frame_temp[9:12], skeleton_frame_temp[15:18]], axis=0)
            orientation_feature_sub[frame, :] = np.arccos(np.clip(np.dot(n_normal, n_center) / np.linalg.norm(n_normal) / np.linalg.norm(n_center), -1, 1))
            frame += 1
        normal_feature[i, :, :] = np.transpose(sequence.pad_sequences(np.transpose(normal_feature_sub), dtype=float, maxlen=maxlen))
        orientation_feature[i, :, :] = np.transpose(sequence.pad_sequences(np.transpose(orientation_feature_sub), dtype=float, maxlen=maxlen))
    return object_feature, affordance_feature, normal_feature, orientation_feature


def main():
    paths = config.Paths()
    paths.path_huang()
    load_affordance_data(paths.metadata_root)


if __name__ == '__main__':
    main()
