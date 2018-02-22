"""
Created on Mar 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import pickle
import json
import bisect
import time
import shutil

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage

import plyfile

import config
import metadata
import grammarutils
import inference
import plyutils
import rrt
import supplementary


# ========================= Planning qualitative results =========================
def visualize_map(m):
    plt.figure()
    plt.pcolor(m)
    plt.colorbar()
    plt.show()


def plot_line(start_point, end_point, color):
    plt.plot([start_point[0]+0.5, end_point[0]+0.5], [start_point[1]+0.5, end_point[1]+0.5], color, lw=3)


def plot_point(point, color):
    plt.plot(point[0]+0.5, point[1]+0.5, color, markersize=5.0)


def visualize_trajectory(underlying_map, trajectories, targets, filename):
    underlying_map = np.ma.masked_array(underlying_map, underlying_map <= 0.1)
    plt.pcolor(underlying_map, cmap=plt.get_cmap('bone'))

    interpolation = 20
    heatmap = np.zeros(underlying_map.shape)
    end_point_cutoff = 1
    for i_traj in range(len(trajectories)):
        if not trajectories[i_traj]:
            continue
        trajectory = np.array(trajectories[i_traj])
        for i in range(trajectory.shape[0]-1-end_point_cutoff):
            if i < end_point_cutoff:
                continue
            x = trajectory[i, 0]
            y = trajectory[i, 1]
            dist = np.sqrt((trajectory[i + 1, 0] - trajectory[i, 0])**2 + (trajectory[i + 1, 1] - trajectory[i, 1])**2)
            dx = (trajectory[i + 1, 0] - trajectory[i, 0]) / dist
            dy = (trajectory[i + 1, 1] - trajectory[i, 1]) / dist
            for inter in range(interpolation):
                x += dx
                y += dy
                heatmap[int(y), int(x)] += 1

    heatmap = scipy.ndimage.grey_dilation(heatmap, size=(3, 3))
    heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=(5, 5))

    ax = plt.gca()
    plt.imshow(heatmap, origin='lower')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.axis('off')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.colorbar()
        plt.show()


def cal_height_affordance(scene_ply_data, height_map, mean, std):
    positions, heights = plyutils.get_pos_height(scene_ply_data)
    affordance_map = np.zeros(height_map.shape)

    dist = scipy.stats.norm(mean, std)
    probs = np.zeros((1, positions.shape[0]))

    # Calculate probabilities for all points
    for i in range(positions.shape[0]):
        pos = positions[i, :]
        if height_map[pos[0], pos[1]] - heights[i] < 0.05:
            if height_map[pos[0], pos[1]] < 0.20:
                h = 0
            else:
                h = height_map[pos[0], pos[1]]
            prob = dist.pdf(h)
            if probs[0, i] < prob:
                probs[0, i] = prob

    # Project to the 2D planes for the valid indices
    valid_indices = plyutils.get_valid_indices(scene_ply_data)
    for i in valid_indices:
        affordance_map[positions[i, 0], positions[i, 1]] = probs[0, i]

    return affordance_map, probs


def cal_obj_affordance(scene_ply_data, obj_pos, distance_dist, height_map, walk_probs, sit_probs):
    positions, heights = plyutils.get_pos_height(scene_ply_data)
    affordance_map = np.zeros(height_map.shape)

    multi_obj = (len(obj_pos) > 1)
    obj_pos = np.array(obj_pos)
    obj_pos = obj_pos[:, [0, 2]]

    mean, std = distance_dist['mu'], distance_dist['std']
    dist = scipy.stats.norm(mean, std)
    probs = np.zeros((1, positions.shape[0]))

    # Calculate probabilities for all points
    for i in range(positions.shape[0]):
        pos = positions[i, :]
        if height_map[pos[0], pos[1]] - heights[i] < 0.05:
            if multi_obj:
                min_distance = float('Inf')
                for o in range(obj_pos.shape[0]):
                    distance = np.sqrt(np.sum((pos*0.01-obj_pos[o, :])**2))
                    if distance < min_distance:
                        min_distance = distance
            else:
                min_distance = np.sqrt(np.sum((pos*0.01-obj_pos)**2))

            prob = dist.pdf(min_distance)*(walk_probs[0, i]+sit_probs[0, i])
            if probs[0, i] < prob:
                probs[0, i] = prob
                scene_ply_data['vertex']['red'][i] = min(float(scene_ply_data['vertex']['red'][i]+prob*20), 255)

    # Project to the 2D planes for the valid indices
    valid_indices = plyutils.get_valid_indices(scene_ply_data)
    for i in valid_indices:
        affordance_map[positions[i, 0], positions[i, 1]] = probs[0, i]

    return affordance_map


def cal_all_affordance(paths):
    for scene in os.listdir(os.path.join(paths.tmp_root, 'scenes')):
        if os.path.isdir(os.path.join(paths.tmp_root, 'scenes', scene)):
            affordance_maps = dict()
            scene_ply_data = plyfile.PlyData.read(os.path.join(paths.tmp_root, 'scenes', scene,'scene.ply'))
            with open(os.path.join(paths.tmp_root, 'scenes', 'labels.json'), 'r') as f:
                labels = json.load(f)
            with open(os.path.join(paths.tmp_root, 'scenes', 'distributions.json'), 'r') as f:
                dist = json.load(f)

            height_map = plyutils.cal_height_map(scene_ply_data, labels, scene)
            affordance_maps['height'] = height_map

            print "Calculating walking affordance..."
            walk_map, walk_probs = cal_height_affordance(scene_ply_data, height_map, 0, 0.1)
            affordance_maps['walk'] = walk_map

            print "Calculating sitting affordance..."
            sit_map, sit_probs = cal_height_affordance(scene_ply_data, height_map, 0.6, 0.08)
            affordance_maps['sit'] = sit_map

            for obj_name in labels[scene].keys():
                if obj_name == 'scene':
                    continue
                else:
                    print "Calculating affordances for {0}...".format(obj_name)
                    affordance_maps[obj_name] = cal_obj_affordance(scene_ply_data, labels[scene][obj_name]['mean'], dist['distance_dist'][obj_name], height_map, walk_probs, sit_probs)

            open(os.path.join(paths.tmp_root, 'scenes', scene+'.ply'), 'a').close()
            scene_ply_data.write(open(os.path.join(paths.tmp_root, 'scenes', scene+'.ply'), 'w'))

            with open(os.path.join(paths.tmp_root, 'scenes', scene+'.p'), 'w') as f:
                pickle.dump(affordance_maps, f)


def erode_and_dilate(m):
    m *= 100000.0
    m = scipy.ndimage.morphology.grey_dilation(m, size=(5, 5))
    m = scipy.ndimage.morphology.grey_erosion(m, size=(15, 15))
    m = scipy.ndimage.morphology.grey_dilation(m, size=(10, 10))
    m /= 100000.0
    return m


def sample_target_obj(dist, action):
    obj_index = np.random.choice(len(dist['actions'][action]['objects']), p=dist['actions'][action]['probs'])
    return dist['actions'][action]['objects'][obj_index]


def sample_target_pos(amap):
    normed_amap = amap.flatten()/np.sum(amap.flatten())
    cdf = np.cumsum(normed_amap)
    pos = bisect.bisect_left(cdf, np.random.rand())
    # target = [pos/amap.shape[1], pos%amap.shape[1]]
    target = [pos%amap.shape[1], pos/amap.shape[1]]
    # target = [pos%amap.shape[0], pos/amap.shape[0]]
    return target


def planning(paths, seed):
    task = 'microwaving_food'
    # actions = ['reaching', 'opening', 'moving']
    actions = ['reaching', 'opening']

    # task = 'random'
    # actions = ['random']

    target_num = 20
    traj_num = 50

    # languages = grammarutils.read_languages(paths)
    # actions = languages[task][np.random.choice(len(languages[task]))]

    with open(os.path.join(paths.tmp_root, 'scenes', 'distributions.json'), 'r') as f:
        dist = json.load(f)
    while 'null' in actions:
        actions.remove('null')

    fig_folder = os.path.join(paths.tmp_root, 'results', 'planning')
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    for scene in os.listdir(os.path.join(paths.tmp_root, 'scenes')):
        if os.path.isdir(os.path.join(paths.tmp_root, 'scenes', scene)):
            with open(os.path.join(paths.tmp_root, 'scenes', scene+'.p'), 'r') as f:
                affordance_maps = pickle.load(f)
                normed_affordance_maps = dict()
                for obj, amap in affordance_maps.items():
                    normed_affordance_maps[obj] = erode_and_dilate(amap)

            for obj, amap in normed_affordance_maps.items():
                normed_affordance_maps[obj] = (amap - np.min(amap)) / (np.max(amap) - np.min(amap))

            walk_map = normed_affordance_maps['walk'] + normed_affordance_maps['sit']
            walk_map[walk_map > 1] = 1.0
            normed_walk_map = (walk_map - np.min(walk_map)) / (np.max(walk_map) - np.min(walk_map))
            # normed_affordance_maps['walk'] = 1 - normed_walk_map + 0.01
            trajectories = list()
            # visualize_map(normed_affordance_maps['height'])

            target_pos = dict()
            for obj, amap in affordance_maps.items():
                target_pos[obj] = sample_target_pos(amap)

            for i_target in range(target_num):
                targets = list()
                targets.append(sample_target_pos(affordance_maps['start']))
                # targets.append([330, 10])

                for action in actions:
                    target_obj = sample_target_obj(dist, action)
                    # targets.append(target_pos[target_obj])
                    targets.append(sample_target_pos(affordance_maps[target_obj]))
                    # print action, target_obj
                print actions, targets

                for i_traj in range(traj_num):
                    print i_target, i_traj
                    for i in range(len(targets)-1):
                        img = normed_walk_map
                        # rrt.plan_trajectory_with_ui((img > 0.01))
                        traj = rrt.plan_trajectory((img > 0.01), targets[i], targets[i+1])
                        trajectories.append(traj)

            visualize_trajectory(normed_affordance_maps['height'], trajectories, targets, filename=os.path.join(fig_folder, '{}_heatmap_{}.png'.format(task, seed)))


# ========================= RGB prediction qualitative results =========================
def get_affordance_posterior(spg, priors, likelihoods, io, start_frame, current_frame):
    action_log_cpt, object_log_cpt, affordance_log_cpt, duration_prior, combined_log_cpt = priors
    action_log_likelihood_sum, object_log_likelihood_sum, affordance_log_likelihood_sum = likelihoods
    frames = action_log_likelihood_sum.shape[-1]

    # log_prior = affordance_log_cpt[spg.subactivity, spg.affordance[io]]
    # log_likelihood = affordance_log_likelihood_sum[io, spg.affordance[io], spg.start_frame, spg.end_frame]
    # sum_prob = np.exp(log_prior+log_likelihood)

    sum_prob = 0
    max_prob = -np.inf
    for u in range(len(metadata.affordances)):
        if metadata.affordances[u] != 'stationary':
            log_prior = affordance_log_cpt[spg.subactivity, u]
            # log_likelihood = affordance_log_likelihood_sum[io, u, min(spg.start_frame-start_frame, frames-1), min(spg.end_frame-start_frame, frames-1)]
            log_likelihood = affordance_log_likelihood_sum[io, u, min(current_frame, frames-1), min(current_frame, frames-1)]
            prob = np.exp(log_prior + log_likelihood)
            sum_prob += prob
            if prob > max_prob:
                max_prob = prob

    # TODO
    return max_prob
    # return sum_prob


def get_action_posterior(spg, priors, likelihoods, start_frame, current_frame):
    action_log_cpt, object_log_cpt, affordance_log_cpt, duration_prior, combined_log_cpt = priors
    action_log_likelihood_sum, object_log_likelihood_sum, affordance_log_likelihood_sum = likelihoods
    frames = action_log_likelihood_sum.shape[-1]

    log_prior = action_log_cpt[spg.subactivity, :]
    log_likelihood = action_log_likelihood_sum[:, min(current_frame, frames - 1), min(current_frame, frames - 1)]
    prob = np.exp(log_prior + log_likelihood)

    return prob


def get_image_coor(point, left_handed=0):
    fx = 525.0
    fy = 525.0
    cx = 319.5
    cy = 239.5
    x = point[0]
    y = point[1]
    z = point[2]
    v = cy - y * fy / z
    if not left_handed:
        u = cx + x * fx / z
    else:
        u = cx - x * fx / z
    return np.array([u, v], dtype=int)


def connect_points(point1, point2):
    # Linear interpolation adding a sine function
    interpolation = 50
    xs = np.linspace(point1[0], point2[0], num=interpolation)
    ys = np.linspace(point1[1], point2[1], num=interpolation)
    ys -= np.sin(np.linspace(0, np.pi, interpolation)) * 10
    path = np.vstack([xs, ys]).T
    return path.astype(int).tolist()


def infer(paths, gt_tpg, priors, grammar_dict, languages, duration):
    fig_folder = os.path.join(paths.project_root, 'tmp', 'results', 'qualitative_max', gt_tpg.id)
    if os.path.exists(fig_folder):
        shutil.rmtree(fig_folder)
    os.makedirs(os.path.join(fig_folder, 'rgb_prediction'))
    os.makedirs(os.path.join(fig_folder, 'action_posterior'))
    os.makedirs(os.path.join(fig_folder, 'parse_graphs'))

    gt_subactivity, gt_objects, gt_affordance = inference.get_ground_truth_label(gt_tpg)
    likelihoods = inference.get_intermediate_results(paths, gt_tpg)
    obj_num = gt_objects.shape[0]
    start_frame = gt_tpg.terminals[0].start_frame

    # Segmentation
    trace_begin, trace_a, trace_o, trace_u, trace_s = inference.dp_segmentation(priors, likelihoods)

    for end_frame in range(1, int(trace_begin.shape[0])):
        tpg = inference.generate_parse_graph(trace_begin, trace_a, trace_o, trace_u, trace_s, start_frame, end_frame)
        tpg.activity = gt_tpg.activity
        # vizutil.visualize_tpg_labeling(gt_subactivity, gt_affordance, tpg, obj_num, end_frame)
        tpg = inference.gibbs_sampling(tpg, grammar_dict, languages, priors, likelihoods)
        # vizutil.visualize_tpg_labeling(gt_subactivity, gt_affordance, tpg, obj_num, end_frame)

        # Prediction
        predicted_tpg = inference.predict(grammar_dict, languages, tpg, end_frame, duration, priors, likelihoods)
        predict_frame = start_frame + end_frame + duration
        if predicted_tpg.terminals[-1].end_frame >= predict_frame:
            current_frame = start_frame + end_frame

            frame_valid = False
            for spg in gt_tpg.terminals:
                if spg.start_frame <= current_frame <= spg.end_frame:
                    skeleton = spg.skeletons[current_frame-spg.start_frame, :]
                    skeleton = np.reshape(skeleton, (15, 3))
                    hand_pos = get_image_coor(skeleton[11, :])
                    frame_valid = True
                    break
            if not frame_valid:
                continue

            for spg in predicted_tpg.terminals:
                if spg.start_frame <= predict_frame <= spg.end_frame:
                    affordance_probs = np.empty(obj_num)
                    for io in range(obj_num):
                        # TODO
                        # probabilities[io] = get_affordance_posterior(spg, priors, likelihoods, io, start_frame, current_frame)
                        affordance_probs[io] = get_affordance_posterior(spg, priors, likelihoods, io, start_frame, predict_frame)
                    if np.sum(affordance_probs) == 0:
                        break
                    affordance_probs = affordance_probs / np.sum(affordance_probs)

                    # Predicted action probabilities
                    action_posterior = np.empty(len(metadata.actions))
                    action_posterior = get_action_posterior(spg, priors, likelihoods, start_frame, predict_frame)
                    if np.sum(action_posterior) == 0:
                        break
                    action_posterior = action_posterior / np.sum(action_posterior)

                    # Get the object bounding boxes
                    obj_pos = np.zeros((obj_num, 2), dtype=int)
                    for io in range(obj_num):
                        annotation_file_path = os.path.join(paths.data_root, '{}_annotations'.format(gt_tpg.subject), gt_tpg.activity, '{}_obj{}.txt'.format(gt_tpg.id, io+1))
                        with open(annotation_file_path) as f:
                            last_image_bbx = None
                            for line in f:
                                line = line.split(',')
                                frame = int(line[0])
                                image_bbx = [int(c) for c in line[2:6]]
                                if not (0 < image_bbx[0] <= 640 and 0 < image_bbx[2] < 640 and 0 < image_bbx[1] < 480 and 0 < image_bbx[3] < 480):
                                    image_bbx = last_image_bbx
                                last_image_bbx = image_bbx
                                if image_bbx and frame >= current_frame:
                                    obj_pos[io, 0] = int((image_bbx[0] + image_bbx[2])/2)
                                    obj_pos[io, 1] = int((image_bbx[1] + image_bbx[3])/2)
                                    break

                    # Draw on RGB and save
                    img_path = os.path.join(paths.data_root, '{}_rgbd_images'.format(gt_tpg.subject), gt_tpg.activity, gt_tpg.id, 'RGB_{}.png'.format(current_frame))
                    img = scipy.ndimage.imread(img_path)

                    kernel_size = (10, 10)
                    heatmap = np.zeros(img.shape[:2])
                    for io in range(obj_num):
                        # heatmap[hand_pos[1], hand_pos[0]] = probabilities[io]
                        # heatmap[obj_pos[io, 1], obj_pos[io, 0]] = probabilities[io]
                        path = connect_points(hand_pos, obj_pos[io, :])
                        for point in path:
                            heatmap[point[1], point[0]] = affordance_probs[io]
                    heatmap = scipy.ndimage.grey_dilation(heatmap, size=kernel_size)
                    heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=kernel_size)
                    heatmap = np.ma.masked_array(heatmap, heatmap <= 0.01)

                    plt.imshow(img)
                    plt.imshow(heatmap, cmap=plt.get_cmap('jet'), alpha=.3)
                    ax = plt.gca()
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    plt.axis('off')
                    plt.tight_layout()

                    # plt.show()
                    plt.savefig(os.path.join(fig_folder, 'rgb_prediction', '{:04d}.png'.format(current_frame)), bbox_inches='tight', pad_inches=0)
                    plt.close()

                    # Plot action posteriors
                    labels = ['reach', 'move', 'pour', 'eat', 'drink', 'open', 'place', 'close', 'clean', 'null']
                    plt.figure(figsize=(8, 3))
                    plt.bar(np.arange(len(action_posterior))+0.5, action_posterior)
                    plt.xticks(np.arange(len(action_posterior))+0.5, labels)
                    ax = plt.gca()
                    ax.set_ylim([0.0, 1.1])
                    # plt.show()
                    plt.savefig(os.path.join(fig_folder, 'action_posterior', '{:04d}.png'.format(current_frame)), bbox_inches='tight', pad_inches=0)
                    plt.close()

                    # Parse graphs
                    parse_graph_ps_file = os.path.join(fig_folder, 'parse_graphs', '{:04d}.ps'.format(current_frame))
                    d, matched_tokens = grammarutils.find_closest_tokens(languages[predicted_tpg.activity], inference.tpg_to_tokens(predicted_tpg, np.inf), truncate=True)
                    grammarutils.get_prediciton_parse_tree(grammar_dict[predicted_tpg.activity], matched_tokens, parse_graph_ps_file)


def evaluate(paths):
    priors = inference.load_prior(paths)
    activity_corpus = pickle.load(open(os.path.join(paths.tmp_root, 'activity_corpus.p'), 'rb'))

    grammar_dict = grammarutils.read_induced_grammar(paths)
    languages = grammarutils.read_languages(paths)

    # Prediction duration
    duration = 20 + 1

    # sequences = ['0126141850', '0129112015', '0129114054', '0505002942']
    # sequences = ['0126142037', '0129114356']
    sequences = ['0129112522']
    for activity, tpgs in activity_corpus.items()[:]:
        print activity
        for tpg in tpgs:
            if tpg.subject != 'Subject5':
                continue
            # if tpg.activity == 'stacking_objects' or tpg.activity == 'picking_objects':
            #     continue

            # if tpg.activity != 'making_cereal' and tpg.activity != 'microwaving_food' and tpg.activity != 'taking_food' and tpg.activity != 'stacking_objects':
            #     continue

            if tpg.activity != 'unstacking_objects':
                continue
            # if tpg.id not in sequences:
            #     continue
            # if tpg.id != '1204142858':  # Taking medicine, start_frame != 0
            #     continue
            print tpg.id, tpg.terminals[-1].end_frame
            infer(paths, tpg, priors, grammar_dict, languages, duration)
            # break
        # break


def main():
    paths = config.Paths()
    start_time = time.time()
    seed = int(time.time())
    print 'seed:', seed
    np.random.seed(0)

    evaluate(paths)
    # supplementary.rgb_videos(paths)

    # plyutils.create_labels(paths)
    # cal_all_affordance(paths)
    # planning(paths, seed)

    print('Time elapsed: {}'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
