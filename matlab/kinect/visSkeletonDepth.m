initialize;

depthHandle = 1;

%% Visualize the skeleton with the point cloud
for dirIndex = 1:length(subDirs)
    datadir = subDirs(dirIndex).name;
    datapath = [dataroot, datadir, '/'];
    

    totalFrames = size(dir(fullfile(datapath, 'aligned_rgb_*.png')), 1);
%     for frame = 1:totalFrames-1
    for frame = 20
        % Read and plot skeleton
        [skeleton3DPosition, handState] = readSkeleton(datapath, frame);
        if size(skeleton3DPosition, 1) > 0
            figure(depthHandle); clf; hold on;
            plot3dSkeleton(skeleton3DPosition, 'r');
        end

        % Plot point cloud
        points = getDepthPoints(datapath, frame);

%         pc = read_ply(sprintf([datapath, 'rgbd_pc_%05d.ply'], frame));
%         pc(:, 1) = -pc(:, 1);
%         showPointCloud(pc);
            
        % Plot skeleton and point cloud
        plotDepth(points, depthHandle);
        hold on;
    end
end

