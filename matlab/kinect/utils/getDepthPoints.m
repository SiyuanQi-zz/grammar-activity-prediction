function points = getDepth( datapath, frame )
%GETDEPTH Summary of this function goes here
%   Detailed explanation goes here

depthName = sprintf([datapath, 'raw_depth_%05d.png'], frame);   
depth = single(imread(depthName))/125.0;
points = depth2world([1, 1, size(depth, 2)-1, size(depth, 1)-1], depth);

end

