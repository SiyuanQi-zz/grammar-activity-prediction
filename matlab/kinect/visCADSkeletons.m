clc;
clear all;
addpath(genpath('utils'));
data_path = '/home/siyuan/Documents/iccv2017/skeletons/';
skeleton_dir = dir([data_path '*.mat']);
% visualize first frame
load([data_path skeleton_dir(1).name]);
% 95 100 dance
line1 = [1,2,3];
line2 = [4,5,12];
line3 = [6,7,13];
line4 = [3,8,9,14];
line5 = [3,10,11,15];
line6 = [4,6,3,4];
for i = 1:size(skeleton,1)
    plot3(skeleton(i,line1*3-2),skeleton(i,line1*3),skeleton(i,line1*3-1),'b');
    hold on;
    plot3(skeleton(i,line2*3-2),skeleton(i,line2*3),skeleton(i,line2*3-1),'y');
    hold on;
    plot3(skeleton(i,line3*3-2),skeleton(i,line3*3),skeleton(i,line3*3-1),'y');
    hold on;
    plot3(skeleton(i,line4*3-2),skeleton(i,line4*3),skeleton(i,line4*3-1),'r');
    hold on;
    plot3(skeleton(i,line5*3-2),skeleton(i,line5*3),skeleton(i,line5*3-1),'r');
    hold on;
    plot3(skeleton(i,line6*3-2),skeleton(i,line6*3),skeleton(i,line6*3-1),'g');
    set(gca,'XLim',[-0.5 0.5]);
    set(gca,'YLim',[1.2 2.2]);
    set(gca,'ZLim',[-0.5 0.5]);
    pause(0.1);
    clf('reset') 
end


