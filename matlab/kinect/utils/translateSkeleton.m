function [ skeleton3DPosition ] = translateSkeleton( skeleton3DPosition, translation )
%TRANSLATESKELETON Summary of this function goes here
%   Detailed explanation goes here

tform = eye(4);
tform(1:3, 4) = translation;

skeleton3DPosition = (tform*([skeleton3DPosition, ones(length(skeleton3DPosition), 1)])')';
skeleton3DPosition = skeleton3DPosition(:, 1:3);

end

