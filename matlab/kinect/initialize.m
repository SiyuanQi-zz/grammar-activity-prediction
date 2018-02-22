clear all; close all; clc;

% Tracking module
addpath('/home/siyuan/libraries/matlab/matconvnet-1.0-beta21/matlab');
addpath(genpath('/home/siyuan/libraries/matlab/MDNet'));
vl_setupnn();
% addpath('/home/siyuan/libraries/matlab/MDNet/models');
% addpath('/home/siyuan/libraries/matlab/MDNet/tracking');
% addpath('/home/siyuan/libraries/matlab/MDNet/utils');
% addpath('/home/siyuan/libraries/matlab/MDNet/pretraining');

addpath(genpath('utils'));

dataroot = '/home/siyuan/data/prediction/data/';
% dataroot = '/home/siyuan/data/prediction_old/';
resultroot = '../../../tmp/';

files = dir(dataroot);
files = files(3:end);
dirFlags = [files.isdir];
subDirs = files(dirFlags);

objects = {'cup', 'cooler', 'monitor', 'keyboard', 'mouse'};
% objects = {'cup'};