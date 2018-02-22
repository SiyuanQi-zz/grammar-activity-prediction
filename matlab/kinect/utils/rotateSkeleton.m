function [ skeleton3DPosition ] = rotateSkeleton( skeleton3DPosition, rotation )
%ROTATESKELETON Summary of this function goes here
%   Detailed explanation goes here

tform = [cos(rotation) -sin(rotation) 0 0; ...
     sin(rotation) cos(rotation) 0 0; ...
     0 0 1 0; ...
     0 0 0 1];
 
skeleton3DPosition = (tform*([skeleton3DPosition, ones(length(skeleton3DPosition), 1)])')';
skeleton3DPosition = skeleton3DPosition(:, 1:3);

end

