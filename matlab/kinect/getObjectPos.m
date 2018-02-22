function objTrackedPos = getObjectPos(datapath, datadir, boundingBoxes )
%GETOBJECTFEATURES Summary of this function goes here
%   Detailed explanation goes here

% Save the tracked bounding boxes in 3D
% load(fullfile(resultroot, 'tracked_objects', [datadir, '.mat']));

totalFrames = size(dir(fullfile(datapath, 'aligned_rgb_*.png')), 1);
objTrackedPos = zeros(size(boundingBoxes, 1), totalFrames, 3);
for objIndex = 1:size(boundingBoxes, 1)
    for frame = 1:totalFrames-1
        bbox = uint8(boundingBoxes(objIndex, frame, :));

        % Plot point cloud of object
        points = getDepthPoints(datapath, frame);
        bbsPoints = points(bbox(2):bbox(2)+bbox(4), bbox(1):bbox(1)+bbox(3), :);
        bbsPoints = getValidPoints(bbsPoints);
        if sum(isnan(mean(bbsPoints))) > 0
            % No valid points inside the bounding box
            warning(sprintf('Sequence %s: empty depth bounding box for object %d at frame %d.', datadir, objIndex, frame));
        end
        objTrackedPos(objIndex, frame, :) = mean(bbsPoints);
    end
end


