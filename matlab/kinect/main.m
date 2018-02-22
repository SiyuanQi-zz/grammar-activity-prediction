initialize;

%% Extract skeleton and object features
% Generate skeleton features
[alignedSkeletons, meanSkeleton] = getSkeletonFeatures(dataroot, subDirs);
objTrackedPos = cell(1, length(subDirs));
objTruePos = cell(1, length(subDirs));
featureIDs = cell(1, length(subDirs));
dirnames = cell(1, length(subDirs));
for dirIndex = 1:length(subDirs)
    datadir = subDirs(dirIndex).name;
    datapath = [dataroot, datadir, '/'];
    dirnames{dirIndex} = datadir;
    load(fullfile(resultroot, 'tracked_objects', [datadir, '.mat']));
%     load(fullfile(resultroot, 'labeled_objects', [datadir, '.mat']));

    % Generate object features based on bounding boxes
    objTrackedPos{dirIndex} = getObjectPos(datapath, datadir, trackedBoxes);

    % Save ground truth bouding boxes
    groundTruthBoxes = getGroundTruthBoxes(dataroot, datadir, objects, size(trackedBoxes));
    objTruePos{dirIndex} = getObjectPos(datapath, datadir, groundTruthBoxes);
    
    % Generate skeleton-object feature
    
%     break;
end
% save([resultroot, 'extracted_data.mat'], 'dirnames', 'alignedSkeletons', 'meanSkeleton', 'objTrackedPos', 'objTruePos');
