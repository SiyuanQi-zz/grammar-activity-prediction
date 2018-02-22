"""
Created on Feb 20, 2016

@author: Siyuan Qi

This file provides the functions to calculate useful stats from ply data and re-color the ply.

"""


import os
import plyfile
import json

import numpy as np
import matplotlib.pyplot as plt

import config


def append_metadata(metadata, ply_path):
    ply_data = plyfile.PlyData.read(ply_path)
    metadata['min'].append([float(np.amin(ply_data['vertex'][dim])) for dim in ['x', 'y', 'z']])
    metadata['max'].append([float(np.amax(ply_data['vertex'][dim])) for dim in ['x', 'y', 'z']])
    metadata['mean'].append([float(np.mean(ply_data['vertex'][dim])) for dim in ['x', 'y', 'z']])

    # positions = np.vstack((ply_data['vertex'][dim] for dim in ['x', 'y'])).T
    # print positions.shape
    # mean_position = positions.mean(axis = 0)
    # print mean_position
    # positions = positions - mean_position[np.newaxis, :]
    # ica = FastICA(n_components=2)
    # ica.fit_transform(positions)
    # axis = ica.mixing_
    # print axis

    return


def create_labels(paths):
    labels = dict()
    data_path = os.path.join(paths.tmp_root, 'scenes')
    for scene in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, scene)):
            labels[scene] = dict()
            objects_in_scene = list()
            for ply in os.listdir(os.path.join(data_path, scene)):
                if os.path.isdir(os.path.join(data_path, scene, ply)):
                    continue

                obj_name = os.path.splitext(ply)[0].split('_')[0]
                if not obj_name in objects_in_scene:
                    objects_in_scene.append(obj_name)
                    labels[scene][obj_name] = dict()
                    labels[scene][obj_name]['max'] = list()
                    labels[scene][obj_name]['min'] = list()
                    labels[scene][obj_name]['mean'] = list()
                    append_metadata(labels[scene][obj_name], os.path.join(data_path, scene, ply))
                else:
                    append_metadata(labels[scene][obj_name], os.path.join(data_path, scene, ply))

    with open(os.path.join(paths.tmp_root, 'scenes', 'labels.json'), 'w') as f:
        json.dump(labels, f, indent=4, separators=(',', ': '))
    return


def get_valid_indices(ply_data):
    valid_indices = list()
    for i in range(ply_data['vertex']['x'].shape[0]):
        if ply_data['vertex'][i][0] > 0 and ply_data['vertex'][i][2] > 0:
            valid_indices.append(i)

    return valid_indices


def get_pos_height(ply_data):
    # truncated_ply_data = truncate_ply(ply_data)
    positions = np.vstack(((ply_data['vertex'][dim]/0.01).astype(int) for dim in ['x', 'z'])).T
    heights = ply_data['vertex']['y']
    return positions, heights


def cal_height_map(ply_data, labels, scene):
    height_map = np.zeros((int(labels[scene]['scene']['max'][0][0]/0.01)+1, int(labels[scene]['scene']['max'][0][2]/0.01)+1))

    positions, heights = get_pos_height(ply_data)
    # vertices =  np.vstack((ply_data['vertex'][dim] for dim in ['x', 'y', 'z'])).T
    # print vertices.shape
    # print vertices[:3, :]
    for i in range(positions.shape[0]):
        if heights[i] > height_map[positions[i, 0], positions[i, 1]]:
            height_map[positions[i, 0], positions[i, 1]] = heights[i]

    # print height_map.shape
    # plt.imshow(height_map, extent=[0, 1, 0, 1])
    # plt.show()

    return height_map


def colorply(amap, height_map, result_path, scene, affordance):
    if not os.path.exists(os.path.join(result_path, scene, affordance+'.ply')):
        return

    ply_data = plyfile.PlyData.read(os.path.join(result_path, scene, affordance+'.ply'))
    positions, heights = get_pos_height(ply_data)

    valid_indices = get_valid_indices(ply_data)
    for i in valid_indices:
        if height_map[positions[i, 0], positions[i, 1]] - heights[i] < 0.05:
            prob = amap[positions[i, 0], positions[i, 1]]
            ply_data['vertex']['red'][i] = min(float(ply_data['vertex']['red'][i]+prob*20), 255)

    if not os.path.exists(os.path.join(result_path, scene, 'noise_reduction')):
        os.makedirs(os.path.join(result_path, scene, 'noise_reduction'))
    ply_data.write(os.path.join(result_path, scene, 'noise_reduction', affordance+'.ply'))


def plot_traj_to_ply(data_path, result_path, tmp_path, scene, targets, trajectories, colors):
    targets = np.array(targets)
    ply_data = plyfile.PlyData.read(os.path.join(data_path, scene, 'scene.ply'))
    positions, heights = get_pos_height(ply_data)

    with open(os.path.join(tmp_path, 'labels.json'), 'r') as f:
        labels = json.load(f)
    height_map = cal_height_map(ply_data, labels, scene)

    valid_indices = get_valid_indices(ply_data)
    for t in range(len(trajectories)):
        trajectory = np.array(trajectories[t])
        color = colors[t]
        for i in valid_indices:
            if height_map[positions[i, 0], positions[i, 1]] - heights[i] < 0.05:
                # print positions
                # print trajectory
                # print trajectory[:, 0] <= positions[i, 0], positions[i, 0] <= trajectory[:, 0]+0.01
                if np.any(
                    np.logical_and(
                        np.logical_and(
                        trajectory[:, 0]-3 <= positions[i, 0], positions[i, 0] <= trajectory[:, 0]+3
                        ),
                    np.logical_and(
                        trajectory[:, 1]-3 <= positions[i, 1], positions[i, 1] <= trajectory[:, 1]+3
                        )
                    )
                    ):
                    ply_data['vertex'][color][i] = 255

    ply_data.write(os.path.join(result_path, scene, 'trajectory.ply'))


def main():
    paths = config.Paths()
    create_labels(paths)


if __name__ == '__main__':
    main()
