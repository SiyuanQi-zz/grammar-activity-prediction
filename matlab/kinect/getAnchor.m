function [pos] = getAnchor(skeleton)
    pos = zeros(3,3);
    pos(1,:) = skeleton([7,8,9]);
    pos(2,:) = skeleton([10,11,12]);
    pos(3,:) = skeleton([16,17,18]);
end