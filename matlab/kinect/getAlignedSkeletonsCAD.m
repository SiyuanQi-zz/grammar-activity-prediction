clc;
clear all;
addpath(genpath('utils'));
anchor = [3,4,6];
data_path = '/home/siyuan/Documents/iccv2017/flipped/skeletons/';
aligned_path = '/home/siyuan/Documents/iccv2017/flipped/aligned_skeletons/';
skeleton_dir = dir([data_path '*.mat']);
skeleton_set = {};
ave = zeros(size(skeleton_dir,1),45);
for i = 1:size(skeleton_dir,1)
    skeleton_set{i} = load([data_path skeleton_dir(i).name]);
    ave(i,:) = mean(skeleton_set{i}.skeleton);
end
ave = mean(ave);
anchor_mean = getAnchor(ave);
num = 0;
for i = 1:size(skeleton_dir,1)
    skeleton_video = skeleton_set{i}.skeleton;
    for j = 1:size(skeleton_video,1)
        disp(j);
        anchor_temp = getAnchor(skeleton_video(j,:));
        skeleton_temp = zeros(15,3);
        skeleton_temp(:,1) = skeleton_video(j,1:3:end);
        skeleton_temp(:,2) = skeleton_video(j,2:3:end);
        skeleton_temp(:,3) = skeleton_video(j,3:3:end);
        [param_w, ~, ~, ~, sof] = helmert3d(anchor_temp,anchor_mean,'7p');
        if sof~=0
            alignedSkeleton = d3trafo(skeleton_temp,param_w,[],0);
            skeleton_video(j,1:3:end) = alignedSkeleton(:,1);
            skeleton_video(j,2:3:end) = alignedSkeleton(:,2);
            skeleton_video(j,3:3:end) = alignedSkeleton(:,3);
            param_temp = param_w;
        else
            alignedSkeleton = d3trafo(skeleton_temp,param_temp,[],0);
            skeleton_video(j,1:3:end) = alignedSkeleton(:,1);
            skeleton_video(j,2:3:end) = alignedSkeleton(:,2);
            skeleton_video(j,3:3:end) = alignedSkeleton(:,3);
            disp('warning: skeleton not aligned');
            num = num + 1;
            fail_example(num).name = skeleton_dir(i).name;
            fail_exmaple(num).frame = j;
            skeleton_plot(skeleton_video(j,:));
        end
    end
    skeleton = skeleton_video;
    skeleton_plot(ave);
    save([aligned_path skeleton_dir(i).name],'skeleton');
end
save('/home/siyuan/Documents/iccv2017/fail_example.mat','fail_example')