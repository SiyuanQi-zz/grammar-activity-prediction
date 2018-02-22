initialize;

depthHandle = 1;

% Visualize the tracked bounding boxes in 2D and 3D
for dirIndex = 1:length(subDirs)
    datadir = subDirs(dirIndex).name;
    datapath = [dataroot, datadir, '/'];
    load(fullfile(resultroot, 'tracked_objects', [datadir, '.mat']));
    % interactedBoxes = trackedBoxes;

    zOffset = 0.3;

    rgbHandle = figure(1); hold on;
    depthHandle = figure(2); hold on;

    totalFrames = size(dir(fullfile(datapath, 'aligned_rgb_*.png')), 1);
        
    for objIndex = 1:size(trackedBoxes, 1)
        for frame = 45
            frame

            % Plot bounding boxes on image
            figure(rgbHandle); clf; hold on;
            imshow(imread(sprintf([datapath, 'aligned_rgb_%05d.png'], frame)));
        
            bbox = trackedBoxes(objIndex, frame, :);
            
            % Plot bounding box
            figure(rgbHandle);
            rectangle('Position', bbox, 'EdgeColor','r', 'LineWidth',3);
            hold on;
            pause(0.1);

            % Read and plot skeleton
            [skeleton3DPosition, handState] = readSkeleton(datapath, frame);
            if size(skeleton3DPosition, 1) > 0
                figure(depthHandle); clf; hold on;
                plot3dSkeleton(skeleton3DPosition, 'r');
            end

            % Plot point cloud of object
            points = getDepthPoints(datapath, frame);
            bbsPoints = points(bbox(2):bbox(2)+bbox(4), bbox(1):bbox(1)+bbox(3), :);
            
            % Plot skeleton and point cloud
%             plotDepth(points, depthHandle);
            plotDepth(bbsPoints, depthHandle);
            hold on;
        end
        break;
    end
    break;
end
