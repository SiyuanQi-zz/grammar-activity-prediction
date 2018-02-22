function [ skeleton3DPosition, handState ] = readSkeleton( datapath, frame )
%READSKELETON Summary of this function goes here
%   Detailed explanation goes here

skeletonName = dir(fullfile(datapath, sprintf('skeleton_%05d_*.txt', frame)));

if size(skeletonName, 1) == 0
    skeleton3DPosition = [];
    handState = [];
else
    skeletonName = [datapath, skeletonName(1).name];
    fid = fopen(skeletonName);
    sktData = textscan(fid, '%f', 'TreatAsEmpty', '-1.#INF');
    sktData = sktData{1};
    sktData(isnan(sktData)) = 0;
    fclose(fid);

    timestamp = sktData(1);
    skeletonLocation = reshape(sktData(2:226), [9, 25])';
    handState = sktData(227:228);
    skeletonOrientation = reshape(sktData(229:end), [4, 25])';

    skeleton3DPosition = skeletonLocation(:, 2:4);
    skeleton3DPosition(:, 1) = -skeleton3DPosition(:, 1);
%     skeleton3DPosition(:, 3) = skeleton3DPosition(:, 3) - 0.1; % for better visualization
end

end

