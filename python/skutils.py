"""
Created on Feb 17, 2017

@author: Siyuan Qi

This file provides helper functions to process and visualize skeletons.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def plot_point(pt, ax, color, **kwargs):
    ax.scatter(pt[0], pt[2], pt[1], 'o', color=color, **kwargs)


def plot_line(pt1, pt2, ax, color='k', **kwargs):
    # Swapped y and z for visualization
    plot_point(pt1, ax, color, **kwargs)
    plot_point(pt2, ax, color, **kwargs)
    # ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], color=color)
    ax.plot([pt1[0], pt2[0]], [pt1[2], pt2[2]], [pt1[1], pt2[1]], color=color, **kwargs)


def plot_upper_skeleton(skeleton, ax, color=None):
    # Swapped y and z for visualization
    # List of bones for the prime senor skeleton
    center_bone_list = [(0, 1), (1, 2)]
    right_bone_list = [(1, 3), (3, 4), (4, 11)]
    left_bone_list = [(1, 5), (5, 6), (6, 12)]

    skeleton = np.reshape(skeleton, (15, 3))
    skeleton[:, 1] = skeleton[:, 1]
    if color:
        for (i, j) in center_bone_list+right_bone_list+left_bone_list:
            plot_line(skeleton[i, :], skeleton[j, :], ax, color=color)
    else:
        for (i, j) in center_bone_list:
            plot_line(skeleton[i, :], skeleton[j, :], ax, color='g')
        for (i, j) in left_bone_list:
            plot_line(skeleton[i, :], skeleton[j, :], ax, color='b')
        for (i, j) in right_bone_list:
            plot_line(skeleton[i, :], skeleton[j, :], ax, color='r')
    # ax.set_aspect('equal')


def plot_skeleton(skeleton, ax, color=None, **kwargs):
    # Swapped y and z for visualization
    # List of bones for the prime senor skeleton
    center_bone_list = [(0, 1), (1, 2)]
    right_bone_list = [(1, 3), (3, 4), (4, 11), (2, 7), (7, 8), (8, 13)]
    left_bone_list = [(1, 5), (5, 6), (6, 12), (2, 9), (9, 10), (10, 14)]

    skeleton = np.reshape(skeleton, (15, 3))
    skeleton[:, 1] = skeleton[:, 1]
    if color:
        for (i, j) in center_bone_list+right_bone_list+left_bone_list:
            plot_line(skeleton[i, :], skeleton[j, :], ax, color=color, **kwargs)
    else:
        for (i, j) in center_bone_list:
            plot_line(skeleton[i, :], skeleton[j, :], ax, color='g', **kwargs)
        for (i, j) in left_bone_list:
            plot_line(skeleton[i, :], skeleton[j, :], ax, color='b', **kwargs)
        for (i, j) in right_bone_list:
            plot_line(skeleton[i, :], skeleton[j, :], ax, color='r', **kwargs)
    # ax.set_aspect('equal')


def plot_point_cloud(pts, ax, marker='.', color='k'):
    # Swapped y and z for visualization
    if pts.ndim == 1:
        ax.scatter(pts[0], pts[2], pts[1], marker, color=color)
    else:
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], marker, color=color)


def plot_skeleton_with_points(skeleton_sequence, pts_sequences, frames):
    def plot_frame(f):
        plot_skeleton(skeleton_sequence[f, :], ax)
        for pts_sequence in pts_sequences:
            plot_point_cloud(pts_sequence[f], ax)

    if frames < 0:
        frame_range = range(skeleton_sequence.shape[0])
    else:
        frame_range = range(frames)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=0., azim=270)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    plt.axis('equal')

    for f in frame_range:
        plot_frame(f)
        fig.canvas.draw_idle()
        plt.pause(0.05)
        ax.cla()
    ani = animation.FuncAnimation(fig, plot_frame, frame_range, interval=25, blit=True)

    plt.show()


    
#visualizes object
    
def visualize_skeleton_obj(skeletons, obj_positions):
    for sequence_id in obj_positions:
        print sequence_id
        plot_skeleton_with_points(skeletons[sequence_id], obj_positions[sequence_id], -1)
        break


def plot_box(size, center, **kwargs):
    x = (np.array([[0, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 0]])-0.5)*size[0] + center[0]
    y=(np.array([[0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1], [0, 0, 1, 1, 1, 1]])-0.5)*size[1] + center[1]
    z=(np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 1]])-0.5)*size[2] + center[2]
    for i in range(6):
        plt.plot(x[:, i], z[:, i], y[:, i], 'b', **kwargs)


def plot_bounding_box(pts, **kwargs):
    z_limit = 2.5
    valid_indices = pts[:, 2] < z_limit
    valid_pts = pts[valid_indices, :]
    if valid_pts.shape[0] == 0:
        return

    center = (np.max(valid_pts, axis=0) + np.min(valid_pts, axis=0)) / 2
    size = np.abs(np.max(valid_pts, axis=0) - np.min(valid_pts, axis=0))
    plot_box(size, center, **kwargs)


def plot_skeleton_prediction(previous_skeletons, skeleton, objs, background=None, background_color=None, filename=None):
    sampled_skeleton_num = 5
    predict_frame = 10
    background_height_lim = 0.5

    # # Compute predicted skeletons
    # skeleton_difference = np.empty(previous_skeletons.shape)
    # previous_skeletons = np.vstack((previous_skeletons, skeleton))
    # for i in range(skeleton_difference.shape[0]):
    #     skeleton_difference[i, :] = previous_skeletons[i+1, :] - previous_skeletons[i, :]
    # skeleton_difference = np.mean(skeleton_difference, axis=0)*predict_frame

    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig)
    plot_skeleton(skeleton, ax, 'r', zorder=10)
    # for _ in range(sampled_skeleton_num):
    #     noise = np.random.rand(45)*0.05
    #     plot_upper_skeleton(skeleton+skeleton_difference+noise, ax, 'g')

    if background is not None:
        background_color = background_color[background[:, 1] < background_height_lim, :]
        background = background[background[:, 1] < background_height_lim, :]
        if background_color is not None:
            ax.scatter(background[:, 0], background[:, 2], background[:, 1], s=[1 for _ in range(background.shape[0])], marker='.', facecolors=background_color, zorder=0)
        else:
            ax.scatter(background[:, 0], background[:, 2], background[:, 1], s=[1 for _ in range(background.shape[0])], marker='.', color='k', zorder=0)

    # Compute 3D bounding boxes and plot
    for obj in objs:
        # ax.scatter(obj[:, 0], obj[:, 2], obj[:, 1], s=[10 for _ in range(obj.shape[0])], marker='.', color='b', zorder=1)
        plot_bounding_box(obj, zorder=5)

    # ax.locator_params(tight=True)
    # ax.view_init(elev=0, azim=270)
    # ax.view_init(elev=30, azim=240)
    ax.view_init(elev=30, azim=250)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.axis('equal')
    plt.axis('off')
    if filename:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    fig.clear()
    plt.close()


def main():
    pass


if __name__ == '__main__':
    main()
