"""
Created on Feb 17, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import time
import fnmatch
import pickle
import json

import numpy as np
import cv2

# Local imports
import config
import skutils
import parsegraph
import grammarutils


def save_activity_corpus(paths, activity_corpus):
    if not os.path.exists(os.path.join(paths.tmp_root, 'corpus')):
        os.makedirs(os.path.join(paths.tmp_root, 'corpus'))

    for event, tpgs in activity_corpus.items():
        corpus_filename = os.path.join(paths.tmp_root, 'corpus', event+'.txt')
        with open(corpus_filename, 'w') as f:
            for tpg in tpgs:
                f.write(str(tpg)+'\n')


def save_action_gt(paths, skeletons, skeleton_labels):
    action_gt = dict()
    for s in skeletons:
        assert skeletons[s].shape[0] == len(skeleton_labels[s])
        skeletons[s] = skeletons[s].tolist()
    action_gt['skeletons'] = skeletons
    action_gt['skeleton_labels'] = skeleton_labels

    with open(os.path.join(paths.tmp_root, 'action.json'), 'w') as f:
        json.dump(action_gt, f, indent=4, separators=(',', ': '))


def get_position_indices():
    start = 1
    position_indices = list()
    for i in range(11):
        position_indices.extend(range(start+i*14+10, start+i*14+13))
    start += 11*14
    for i in range(4):
        position_indices.extend(range(start+i*4, start+i*4+3))
    return position_indices


def get_left_handed_indices():
    flipped_joint_indices = [0, 1, 2, 5, 6, 3, 4, 9, 10, 7, 8, 12, 11, 14, 13]
    left_handed_indices = list()
    for joint in flipped_joint_indices:
        left_handed_indices.extend([3*joint, 3*joint+1, 3*joint+2])
    return left_handed_indices


def get_skeletons(skeletons, eventdir, sequence_ids, left_handed=False):
    position_indices = get_position_indices()
    left_handed_indices = get_left_handed_indices()
    x_indices = [i*3 for i in range(15)]
    for sequence_id in sequence_ids:
        raw_skeleton_data = np.genfromtxt(os.path.join(eventdir, sequence_id+'.txt'), delimiter=',', skip_footer=1, usecols=range(171))
        assert raw_skeleton_data[-1, 0] == raw_skeleton_data.shape[0]
        joint_positions = raw_skeleton_data[:, position_indices]/1000.0
        if left_handed:
            joint_positions[:, x_indices] = -joint_positions[:, x_indices]
            joint_positions = joint_positions[:, left_handed_indices]
        skeletons[sequence_id] = joint_positions

    return skeletons


def get_obj_positions(obj_positions, eventdir, sequence_ids, left_handed=False):
    # Intrinsic camera parameters
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    z_scale = 12.5

    for sequence_id in sequence_ids:
        print 'get_obj_positions', sequence_id
        obj_positions[sequence_id] = list()
        for filename in sorted(os.listdir(eventdir)):
            if fnmatch.fnmatch(filename, '{}_obj*.txt'.format(sequence_id)):
                print filename
                position_sequence = list()

                with open(os.path.join(eventdir, filename)) as f:
                    last_image_bbx = None
                    for line in f:
                        line = line.split(',')
                        frame = line[0]
                        # if not os.path.exists(os.path.join(eventdir.replace('annotations', 'rgbd_images'), sequence_id, 'Depth_{}.png'.format(frame))):
                        #     exit(1)
                        depth = cv2.imread(os.path.join(eventdir.replace('annotations', 'rgbd_images'), sequence_id, 'Depth_{}.png'.format(frame)), -1)
                        depth = depth.astype(float) / z_scale

                        image_bbx = [int(c) for c in line[2:6]]
                        if not (0 < image_bbx[0] < 640 and 0 < image_bbx[2] < 640 and 0 < image_bbx[1] < 480 and 0 < image_bbx[3] < 480):
                            if last_image_bbx:
                                image_bbx = last_image_bbx
                            else:
                                continue
                        else:
                            last_image_bbx = image_bbx
                        # image_bbx = [0, 0, depth.shape[1], depth.shape[0]]

                        step = 10
                        # # If the object is invisible, this will be an empty array
                        positions = np.empty((len(range(image_bbx[0], image_bbx[2], step))*len(range(image_bbx[1], image_bbx[3], step)), 3))
                        pt_count = 0
                        for u in xrange(image_bbx[0], image_bbx[2], step):
                            for v in xrange(image_bbx[1], image_bbx[3], step):
                                z = depth[v, u]
                                if z == 0:
                                    continue
                                if not left_handed:
                                    x = (u - cx) * z / fx
                                else:
                                    x = -(u - cx) * z / fx
                                y = -(v - cy) * z / fy  # Note: need to flip y to align depth with skeleton

                                positions[pt_count, :] = np.array((x, y, z))
                                pt_count += 1
                        positions = positions[:pt_count, :]
                        position_sequence.append(positions)
                        # if np.isnan(np.mean(positions, axis=0)[0]):
                        #     print image_bbx, pt_count, positions.shape, positions
                        #     exit(1)
                        # position_sequence.append(np.mean(positions, 0))
                        # position_sequence.append(np.ones((10, 3)))

                obj_positions[sequence_id].append(position_sequence)

    return obj_positions


def parse_data(paths):
    if os.path.exists(os.path.join(paths.tmp_root, 'activity_corpus.p')):
        activity_corpus = pickle.load(open(os.path.join(paths.tmp_root, 'activity_corpus.p'), 'rb'))
    else:
        activity_corpus = dict()
        skeletons = dict()
        skeleton_labels = dict()
        obj_positions = dict()
        for datadir in os.listdir(os.path.join(paths.data_root)):
            datadir = os.path.join(paths.data_root, datadir)
            if os.path.isdir(datadir) and datadir.endswith('annotations'):
                subject = os.path.split(datadir)[1].strip('_annotations')
                print subject
                left_handed = subject == 'Subject3'
                for event in os.listdir(datadir):
                    # if event != 'stacking_objects':
                    #     continue
                    if event not in activity_corpus:
                        activity_corpus[event] = list()
                    eventdir = os.path.join(datadir, event)

                    sequence_objects = dict()
                    sequence_ids = list()
                    with open(os.path.join(eventdir, 'activityLabel.txt')) as f:
                        for line in f:
                            activity_labels = line.strip(',\n').split(',')
                            sequence_ids.append(activity_labels[0])
                            activity_corpus[event].append(parsegraph.TParseGraph(event, activity_labels[0], subject))
                            sequence_objects[activity_labels[0]] = [o.split(':')[-1] for o in activity_labels[3:]]

                    get_skeletons(skeletons, eventdir, sequence_ids, left_handed)
                    get_obj_positions(obj_positions, eventdir, sequence_ids, left_handed)
                    # skutils.visualize_skeleton_obj(skeletons, obj_positions)

                    # Parse data into spatial-temporal parse graphs
                    with open(os.path.join(eventdir, 'labeling.txt')) as f:
                        for line in f:
                            sequence_labeling = line.strip().split(',')
                            sequence_id = sequence_labeling[0]
                            tpg = next(tpg for tpg in activity_corpus[event] if tpg.id == sequence_id)
                            start_frame = int(sequence_labeling[1])
                            end_frame = int(sequence_labeling[2])
                            subactivity = sequence_labeling[3]
                            affordance_labels = sequence_labeling[4:]

                            # Create ground truth for action recognition
                            if sequence_id not in skeleton_labels:
                                skeleton_labels[sequence_id] = ['null' for _ in range(skeletons[sequence_id].shape[0])]

                            for frame in range(start_frame-1, end_frame):
                                if frame >= skeletons[sequence_id].shape[0]:
                                    break
                                skeleton_labels[sequence_id][frame] = subactivity

                            # Create ground truth ST-pgs
                            spg = parsegraph.SParseGraph(start_frame-1, end_frame-1, subactivity, subactivity, sequence_objects.get(sequence_id), affordance_labels)
                            spg.set_skeletons(skeletons[sequence_id][start_frame - 1:end_frame, :])
                            spg.set_obj_positions(obj_positions[sequence_id])
                            tpg.append_terminal(spg)

                    # break
                # break

        pickle.dump(activity_corpus, open(os.path.join(paths.tmp_root, 'activity_corpus.p'), 'wb'))
        pickle.dump(skeletons, open(os.path.join(paths.tmp_root, 'skeletons.p'), 'wb'))
        pickle.dump(obj_positions, open(os.path.join(paths.tmp_root, 'obj_positions.p'), 'wb'))
        # save_action_gt(paths, skeletons, skeleton_labels)
        # save_activity_corpus(paths, activity_corpus)


def main():
    paths = config.Paths()
    start_time = time.time()

    parse_data(paths)
    # grammarutils.induce_activity_grammar(paths)
    # grammarutils.read_induced_grammar(paths)
    print('Time elapsed: {}'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
