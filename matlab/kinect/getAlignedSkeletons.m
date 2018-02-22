function [ alignedSkeletons, meanSkeleton ] = getSkeletonFeatures( dataroot, subDirs )
%GETSKELETONFEATURES Summary of this function goes here
%   Detailed explanation goes here

% Feature size: [25, 3]

% Anchor joints
anc = [1,5,9];

%% Read in all skeletons and compute mean skeleton
allSkeletons = [];
for dirIndex = 1:length(subDirs)
    datadir = subDirs(dirIndex).name;
    datapath = [dataroot, datadir, '/'];
    
    % Load images
%     totalFrames = size(dir(fullfile(datapath, 'aligned_rgb_*.png')), 1);
    totalFrames = size(dir(fullfile(datapath, 'raw_rgb_*.bmp')), 1);
    for frame = 1:totalFrames
%         frame
        % Read skeleton
        [skeleton3DPosition, handState] = readSkeleton(datapath, frame);
        if size(skeleton3DPosition, 1) == 0
            continue;
        end
        allSkeletons = cat(3, allSkeletons, skeleton3DPosition);
    end
end
meanSkeleton = sum(allSkeletons, 3)/size(allSkeletons, 3);


%% Compute aligned skeletons
alignedSkeletons = cell(1, length(subDirs));
for dirIndex = 1:length(subDirs)
    datadir = subDirs(dirIndex).name;
    datapath = [dataroot, datadir, '/'];
    
    % Load images
%     totalFrames = size(dir(fullfile(datapath, 'aligned_rgb_*.png')), 1);
    totalFrames = size(dir(fullfile(datapath, 'raw_rgb_*.bmp')), 1);
    skeletons = cell(1, totalFrames);
    for frame = 1:totalFrames
        % Read skeleton
        [skeleton3DPosition, handState] = readSkeleton(datapath, frame);
        if size(skeleton3DPosition, 1) == 0
            continue;
        end

        [param_w, ~, ~, ~, sof] = helmert3d(skeleton3DPosition(anc,:),meanSkeleton(anc,:),'7p');
        if sof~=0
            alignedSkeleton = d3trafo(skeleton3DPosition,param_w,[],0);
            skeletons{frame} = alignedSkeleton;
        else
            disp('warning: skeleton not aligned');
        end
    end

    alignedSkeletons{dirIndex} = skeletons;
end


end

