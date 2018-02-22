"""
Created on Mar 19, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import shutil
import pickle
import time

import scipy.misc
import numpy as np
import cv2

import config
import skutils


def get_background(depth_image_path, rgb_image_path=None):
    # Intrinsic camera parameters
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    z_scale = 12.5

    depth = cv2.imread(depth_image_path, -1)
    depth = depth.astype(float) / z_scale
    background = np.empty((depth.size, 3))

    if rgb_image_path:
        rgb = scipy.misc.imread(rgb_image_path)
        background_color = np.empty((depth.size, 3))
    else:
        background_color = None

    valid_count = 0
    step = 4
    for v in range(0, depth.shape[0], step):
        for u in range(0, depth.shape[1], step):
            z = depth[v, u]
            if z == 0:
                continue

            x = (u - cx) * z / fx
            y = -(v - cy) * z / fy  # Note: need to flip y to align depth with skeleton

            background[valid_count, :] = np.array([x, y, z])
            if rgb_image_path:
                background_color[valid_count, :] = rgb[v, u]/255.0
            valid_count += 1

    background = background[:valid_count, :]
    background_color = background_color[:valid_count, :]
    return background, background_color


def plot_demo(paths):
    with open(os.path.join(paths.tmp_root, 'skeletons.p'), 'rb') as f:
        all_skeletons = pickle.load(f)
    with open(os.path.join(paths.tmp_root, 'obj_positions.p'), 'rb') as f:
        all_obj_positions = pickle.load(f)

    activity_corpus = pickle.load(open(os.path.join(paths.tmp_root, 'activity_corpus.p'), 'rb'))
    rgb_figure_folder = os.path.join(paths.tmp_root, 'results', '315', 'qualitative_max')

    # demo_videos = ['0126142037', '0126143115', '0129111131', '0129112630', '0129114356']
    # demo_videos = ['0126142037', '0126143115']
    # demo_videos = ['0126142037', '0129114356']
    demo_videos = ['0126143115']
    # demo_videos = ['0129112015', '0129112226', '0129111131', '0504232829', '0504233253', '0504233320', '0126143115', '0126143251', '0126143431', '0504235245', '0504235647', '0504235908', '0505002750', '0505002942', '0505003237', '0126141638', '0126141850', '0126142037', '0126142253', '0511141007', '0511141231', '0511141338', '0129114054', '0129114153', '0129114356', '0129112342', '0129112522', '0129112630', '0511140410', '0511140450', '0511140553']
    for activity, tpgs in activity_corpus.items():
        for tpg in tpgs:
            if tpg.subject != 'Subject5':
                continue
            if tpg.id not in demo_videos:
                continue
            # if tpg.id != '0126142037':
            #     continue

            print tpg.id
            fig_folder = os.path.join(paths.tmp_root, 'results', 'skeletons_top_view', tpg.id)
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)

            # # ================ For Plotting
            # # frame_list = list()
            # # for img_path in os.listdir(os.path.join(rgb_figure_folder, tpg.id)):
            # #     if os.path.splitext(img_path)[1] != '.png':
            # #         continue
            # #     frame_list.append(int(os.path.splitext(img_path)[0])+1)
            # # frame_list = sorted(frame_list)
            #
            # start_frame = tpg.terminals[0].start_frame
            # end_frame = tpg.terminals[-1].end_frame
            # obj_num = len(tpg.terminals[0].objects)
            # # skeletons = np.empty((end_frame - start_frame + 1, 45))
            # # obj_positions = [list() for _ in range(obj_num)]
            # #
            # # for spg in tpg.terminals:
            # #     # print spg.start_frame, spg.end_frame
            # #     skeletons[spg.start_frame - start_frame:spg.end_frame - start_frame+1, :] = spg.skeletons
            # #
            # #     frames = spg.end_frame - spg.start_frame + 1
            # #     for i in range(obj_num):
            # #         obj_positions[i].extend(spg.obj_positions[i])
            # #         obj_positions[i].extend([np.empty((0, 3)) for _ in range(frames-len(spg.obj_positions[i]))])
            #
            # skeletons = all_skeletons[tpg.id]
            # obj_positions = all_obj_positions[tpg.id]
            # # frame_list = range(skeletons.shape[0])
            # frame_list = range(end_frame + 1)
            # # print start_frame, end_frame, skeletons.shape, len(frame_list)
            #
            # avg_frame = 5
            # for frame in frame_list:
            #     depth_image_path = os.path.join(paths.data_root, tpg.subject+'_rgbd_images', activity, tpg.id, 'Depth_{}.png'.format(frame+1))
            #     rgb_image_path = os.path.join(paths.data_root, tpg.subject+'_rgbd_images', activity, tpg.id, 'RGB_{}.png'.format(frame+1))
            #     background, background_color = get_background(depth_image_path, rgb_image_path)
            #     filename = os.path.join(fig_folder, '{:04d}.png'.format(frame))
            #     # filename = None
            #
            #     # if len(obj_positions) > 0 and len(obj_positions[0]) > 0:
            #     # skutils.plot_skeleton_prediction(skeletons[max(0, frame-start_frame-avg_frame):min(frame-start_frame, skeletons.shape[0]), :], skeletons[min(frame-start_frame, skeletons.shape[0]), :], [obj_positions[i][frame-start_frame] for i in range(obj_num)], background, filename=filename)
            #     objs = list()
            #     for i in range(obj_num):
            #         if len(obj_positions[i]) > frame:
            #             objs.append(obj_positions[i][frame])
            #     skutils.plot_skeleton_prediction(skeletons[max(0, frame-avg_frame):min(frame, skeletons.shape[0]), :], skeletons[min(frame, skeletons.shape[0]), :], objs, background, background_color=background_color, filename=filename)
            #     # break
            # # ================ For Plotting

            if os.path.exists('{}/{}.mp4'.format(fig_folder, tpg.id)):
                os.remove('{}/{}.mp4'.format(fig_folder, tpg.id))
            # os.system('ffmpeg -framerate 15 -i {}/%*.png -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {}/{}.mp4'.format(fig_folder, fig_folder, tpg.id))
            os.system('ffmpeg -framerate 15 -i {}/%*.png -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf scale=640:480 {}/{}.mp4'.format(fig_folder, fig_folder, tpg.id))
            # break
        # break


def rgb_videos(paths):
    activity_corpus = pickle.load(open(os.path.join(paths.tmp_root, 'activity_corpus.p'), 'rb'))

    # demo_videos = ['0126142037', '0126143115', '0129111131', '0129112630', '0129114356']
    # demo_videos = ['0126142037', '0126143115']
    # demo_videos = ['0126142037', '0129114356']
    demo_videos = ['0129112522']
    for activity, tpgs in activity_corpus.items():
        for tpg in tpgs:
            if tpg.subject != 'Subject5':
                continue

            if tpg.id in demo_videos:
                continue

            print tpg.id
            # fig_folder = os.path.join(paths.project_root, 'tmp', 'results', '315', 'qualitative_max', tpg.id)
            fig_folder = os.path.join(paths.project_root, 'tmp', 'results', 'qualitative_max', tpg.id)
            rgb_fig_folder = os.path.join(fig_folder, 'rgb_prediction')
            pg_fig_folder = os.path.join(fig_folder, 'parse_graphs')
            prob_fig_folder = os.path.join(fig_folder, 'action_posterior')

            frame_list = list()
            for img_path in os.listdir(os.path.join(rgb_fig_folder)):
                if os.path.splitext(img_path)[1] != '.png':
                    continue
                frame_list.append(int(os.path.splitext(img_path)[0]))
            frame_list = sorted(frame_list)

            start_frame = tpg.terminals[0].start_frame
            end_frame = tpg.terminals[-1].end_frame
            # for f in range(start_frame-1):
            #     dest_img_path = os.path.join(rgb_fig_folder, '{:04d}.png'.format(f))
            #     if os.path.exists(dest_img_path):
            #         os.remove(dest_img_path)

            # first_valid_frame = frame_list[0]
            # for f in range(first_valid_frame):
            #     src_img_path = os.path.join(paths.data_root, '{}_rgbd_images'.format(tpg.subject), tpg.activity, tpg.id, 'RGB_{}.png'.format(f+1))
            #     dest_img_path = os.path.join(rgb_fig_folder, '{:04d}.png'.format(f))
            #     shutil.copyfile(src_img_path, dest_img_path)
            #
            #     src_img_path = os.path.join(pg_fig_folder, '{:04d}.png'.format(first_valid_frame))
            #     dest_img_path = os.path.join(pg_fig_folder, '{:04d}.png'.format(f))
            #     shutil.copyfile(src_img_path, dest_img_path)
            #
            #     src_img_path = os.path.join(prob_fig_folder, '{:04d}.png'.format(first_valid_frame))
            #     dest_img_path = os.path.join(prob_fig_folder, '{:04d}.png'.format(f))
            #     shutil.copyfile(src_img_path, dest_img_path)

            last_valid_frame = None
            for f in range(end_frame + 1):
                if f not in frame_list:
                    # src_img_path = os.path.join(paths.data_root, '{}_rgbd_images'.format(tpg.subject), tpg.activity, tpg.id, 'RGB_{}.png'.format(f+1))
                    # dest_img_path = os.path.join(rgb_fig_folder, '{:04d}.png'.format(f))
                    # shutil.copyfile(src_img_path, dest_img_path)
                    if last_valid_frame:
                        src_img_path = os.path.join(rgb_fig_folder, '{:04d}.png'.format(last_valid_frame))
                        dest_img_path = os.path.join(rgb_fig_folder, '{:04d}.png'.format(f))
                        shutil.copyfile(src_img_path, dest_img_path)

                        src_img_path = os.path.join(pg_fig_folder, '{:04d}.png'.format(last_valid_frame))
                        dest_img_path = os.path.join(pg_fig_folder, '{:04d}.png'.format(f))
                        shutil.copyfile(src_img_path, dest_img_path)

                        src_img_path = os.path.join(prob_fig_folder, '{:04d}.png'.format(last_valid_frame))
                        dest_img_path = os.path.join(prob_fig_folder, '{:04d}.png'.format(f))
                        shutil.copyfile(src_img_path, dest_img_path)
                else:
                    last_valid_frame = f

            # Make videos
            if os.path.exists('{}/{}_rgb.mp4'.format(fig_folder, tpg.id)):
                os.remove('{}/{}_rgb.mp4'.format(fig_folder, tpg.id))
            os.system('ffmpeg -framerate 15 -i {}/%*.png -vf scale=640:480 -vcodec libx264 -crf 25 -pix_fmt yuv420p {}/{}_rgb.mp4'.format(rgb_fig_folder, fig_folder, tpg.id))

            if os.path.exists('{}/{}_pg.mp4'.format(fig_folder, tpg.id)):
                os.remove('{}/{}_pg.mp4'.format(fig_folder, tpg.id))
            os.system('ffmpeg -framerate 15 -i {}/%*.png -vf scale=640:-2 -vcodec libx264 -crf 25 -pix_fmt yuv420p {}/{}_pg.mp4'.format(pg_fig_folder, fig_folder, tpg.id))

            if os.path.exists('{}/{}_prob.mp4'.format(fig_folder, tpg.id)):
                os.remove('{}/{}_prob.mp4'.format(fig_folder, tpg.id))
            os.system('ffmpeg -framerate 15 -i {}/%*.png -vf scale=640:-2 -vcodec libx264 -crf 25 -pix_fmt yuv420p {}/{}_prob.mp4'.format(prob_fig_folder, fig_folder, tpg.id))
            # os.system('ffmpeg -framerate 15 -i {}/%*.png  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -vcodec libx264 -crf 25 -pix_fmt yuv420p {}/{}_prob.mp4'.format(prob_fig_folder, fig_folder, tpg.id))

            # break
        # break


def main():
    paths = config.Paths()
    start_time = time.time()
    plot_demo(paths)
    # rgb_videos(paths)
    print('Time elapsed: {}'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
