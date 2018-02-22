function points3d = depth2world( bbox, depthImg )
%DEPTH2WORLD Summary of this function goes here
%   Detailed explanation goes here

x = bbox(1);
y = bbox(2);
w = bbox(3);
h = bbox(4);

depths = depthImg(y:y+h, x:x+w);
[xx,yy] = meshgrid(x:x+w, y:y+h);
  
% camera_params;
% X = -(xx - cx_d) .* depths / fx_d;
% Y = -(yy - cy_d) .* depths / fy_d;

X = -(xx - 261.696594) .* depths / 368.096588;
Y = -(yy - 202.522202) .* depths / 368.096588;
Z = depths;

points3d = cat(3, X, Y, Z);

end

