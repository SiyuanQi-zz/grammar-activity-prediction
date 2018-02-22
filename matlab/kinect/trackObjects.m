initialize;

%% Important:
% Link to cuda library when starting up matlab to run this script:
% LD_LIBRARY_PATH=/usr/local/cuda/lib64:local matlab

net = 'mdnet_vot-otb.mat';

%% Track the labelled objects from the first frame
for dirIndex = 1:length(subDirs)
    datadir = subDirs(dirIndex).name;
    datapath = [dataroot, datadir, '/'];
    load(fullfile(resultroot, 'initial_boxes', [datadir, '.mat']));
    
    trackImgPath = fullfile(resultroot, 'tracked_objects', datadir);
    mkdir(trackImgPath);
    
    % Load images
%     imgFiles = dir([datapath, 'aligned_rgb_*.png']);
    imgFiles = dir([datapath, 'raw_rgb_*.bmp']);
    totalFrames = size(imgFiles, 1);
    imgList = cell(1, totalFrames);
    for frame = 1:totalFrames
        imgList{frame} = [datapath, imgFiles(frame).name];
    end
    
    trackedBoxes = zeros(size(initialBoxes, 1), totalFrames, 4);
    trackedScores = zeros(size(initialBoxes, 1), totalFrames);
    for objIndex = 1:size(initialBoxes, 1)
        [result, scores] = mdnet_run(imgList, initialBoxes(objIndex, :), net, true, fullfile(trackImgPath, int2str(objIndex)));
        trackedBoxes(objIndex, :, :) = result;
        trackedScores(objIndex, :) = scores;
    end
    save(fullfile(resultroot, 'tracked_objects', [datadir, '.mat']), 'trackedBoxes', 'trackedScores');
end