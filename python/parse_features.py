"""
Created on Mar 13, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import time
import json
import pickle

import config


def parse_colon_seperated_features(colon_seperated):
    f_list = [int(x.split(':')[1]) for x in colon_seperated]
    return f_list


def read_features(filename):
    data = dict()
    with open(filename) as f:
        first_line = f.readline().strip()
        object_num = int(first_line.split(' ')[0])
        object_object_num = int(first_line.split(' ')[1])
        skeleton_object_num = int(first_line.split(' ')[2])

        # Object feature
        o_aff = []
        o_id = []
        o_fea = []
        for _ in range(object_num):
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            o_aff.append(int(colon_seperated[0]))
            o_id.append(int(colon_seperated[1]))
            object_feature = parse_colon_seperated_features(colon_seperated[2:])
            assert len(object_feature) == 180
            o_fea.append(object_feature)
        data['o_aff'] = o_aff
        data['o_id'] = o_id
        data['o_fea'] = o_fea

        # Skeleton feature
        line = f.readline()
        colon_seperated = [x.strip() for x in line.strip().split(' ')]
        data['h_act'] = int(colon_seperated[0])
        skeleton_feature = parse_colon_seperated_features(colon_seperated[2:])
        assert len(skeleton_feature) == 630
        data['h_fea'] = skeleton_feature

        o_o_id = []
        o_o_fea = []
        # Object-object feature
        for _ in range(object_object_num):
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            o_o_id.append([int(colon_seperated[2]), int(colon_seperated[3])])
            object_object_feature = parse_colon_seperated_features(colon_seperated[4:])
            assert len(object_object_feature) == 200
            o_o_fea.append(object_object_feature)
        data['o_o_id'] = o_o_id
        data['o_o_fea'] = o_o_fea

        s_o_id = []
        s_o_fea = []
        # Skeleton-object feature
        for _ in range(skeleton_object_num):
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            s_o_id.append(int(colon_seperated[2]))
            skeleton_object_feature = parse_colon_seperated_features(colon_seperated[3:])
            assert len(skeleton_object_feature) == 400
            s_o_fea.append(skeleton_object_feature)
        data['s_o_id'] = s_o_id
        data['s_o_fea'] = s_o_fea

        for o_id, s_o_id in zip(data['o_id'] , data['s_o_id']):
            assert o_id == s_o_id
    return data


def collect_data(paths):
    segments_files_path = os.path.join(paths.data_root, 'features_cad120_ground_truth_segmentation', 'segments_svm_format')
    segments_feature_path = os.path.join(paths.data_root, 'features_cad120_ground_truth_segmentation', 'features_binary_svm_format')

    activity_corpus = pickle.load(open(os.path.join(paths.tmp_root, 'activity_corpus.p'), 'rb'))

    subject5_sequences = list()
    segment_count_dict = dict()
    for activity, tpgs in activity_corpus.items()[:]:
        for tpg in tpgs:
            segment_count_dict[tpg.id] = len(tpg.terminals)
            if tpg.subject == 'Subject5':
                subject5_sequences.append(tpg.id)

    data = dict()
    for sequence_path_file in os.listdir(segments_files_path):
        sequence_id = os.path.splitext(sequence_path_file)[0]
        data[sequence_id] = list()
        if sequence_id not in segment_count_dict:
            continue

        with open(os.path.join(segments_files_path, sequence_path_file)) as f:
            first_line = f.readline()
            segment_feature_num = int(first_line.split(' ')[0])
            # if sequence_id in subject5_sequences:
            #     print sequence_id, segment_count_dict[sequence_id], segment_feature_num
            #     assert segment_count_dict[sequence_id] == segment_feature_num

            last_oid = None
            for _ in range(segment_feature_num):
                segment_feature_filename = f.readline().strip()
                segment_data = read_features(os.path.join(segments_feature_path, os.path.basename(segment_feature_filename)))
                data[sequence_id].append(segment_data)
                if last_oid:
                    for o_id, s_o_id in zip(last_oid, segment_data['o_id']):
                        assert o_id == s_o_id
                last_oid = segment_data['o_id']

    pickle.dump(data, open(os.path.join(paths.tmp_root, 'features.p'), 'wb'))
    with open(os.path.join(paths.tmp_root, 'features.json'), 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))


def main():
    paths = config.Paths()
    start_time = time.time()
    collect_data(paths)
    print('Time elapsed: {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
