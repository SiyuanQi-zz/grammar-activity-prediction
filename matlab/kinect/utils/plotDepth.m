function plotDepth( points , handle)
%PLOTDEPTH Summary of this function goes here
%   Detailed explanation goes here

% validPoints = getValidPoints(points);

if nargin == 2
    figure(handle);
end

% scatter3(validPoints(:, :, 1), validPoints(:, :, 2), validPoints(:, :, 3));
pcshow(points);
campos([0, 0, 0]);
camup([0, 1, 0]);
axis equal;

end

