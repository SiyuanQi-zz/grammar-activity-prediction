function validPoints = getValidPoints( points )
%GETVALIDPOINTS Summary of this function goes here
%   Detailed explanation goes here

[non_zero_i, non_zero_j] = ind2sub([size(points, 1), size(points, 2)], find(points(:, :, 3)));
non_zero_x = points(sub2ind(size(points), non_zero_i, non_zero_j, ones(size(non_zero_i))));
non_zero_y = points(sub2ind(size(points), non_zero_i, non_zero_j, ones(size(non_zero_i))*2));
non_zero_z = points(sub2ind(size(points), non_zero_i, non_zero_j, ones(size(non_zero_i))*3));
validPoints = cat(3, non_zero_x, non_zero_y, non_zero_z);
validPoints = reshape(validPoints, [], 3);

end

