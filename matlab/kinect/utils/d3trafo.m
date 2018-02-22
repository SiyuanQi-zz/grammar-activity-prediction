function xyz=d3trafo(XYZ,p,O,dir,FileOut)

% D3TRAFO performs 3D-transformation with 7 or 10 parameters (at arbitrary rotation center)
%         in either transformation direction
%
% xyz=d3trafo(XYZ,p,O,dir,FileOut)
%
% Also usable:   Transformations.mat   (see beneath)
%
% Inputs:  XYZ  nx3-matrix to be transformed. 3xn-matrices are allowed, be careful with
%               3x3-matrices!
%               XYZ may also be a file name with ASCII data to be processed. No point IDs, only
%               coordinates as if it was a matrix.
%
%            p  The vector of transformation parameters [dx dy dz ex ey ez s] with
%                 dx,dy,dz = translations [unit of XYZ]
%                 ex,ey,ez = rotations in [rad]
%                        s = scale factor
%               p may also be a string with the name of a predefined transformation stored in
%               Transformations.mat. In this case, the following input of O is ignored.
%
%            O  The center [xO yO zO] of the rotation in units of XYZ.
%               Default if omitted or set to [] is [0 0 0] (ellipsoid centered, Bursa-Wolf type).
%
%          dir  the transformation direction.
%               If dir=0 (default if omitted or set to []), p are used as given.
%               If dir=1, inverted p' is used to calculate the back-transformation (i.e. if p was
%                  calculated in the direction Sys1 -> Sys2, p' is for Sys2 -> Sys1).
%
%      FileOut  File to write the output to. If omitted, no output file is generated.
%
% Output:  xyz  nx3-matrix with the transformed coordinates.
%
% Systems need to be right-handed, i.e. [x y z].
% Used for transforming cartesian coordinates from one system to another, e.g. when changing 
% the reference ellipsoid,  called "datum transformation" in geodesy.
% The standard parameters in Transformation.mat may be used only when minor accuracy is desired
% (in most times there is only some m-accuracy.) For precise geodetic datum transformations,
% you need identical points in both systems to determine parameters using helmert3d function.

% Author:
% Peter Wasmeier, Technical University of Munich
% p.wasmeier@bv.tum.de
% Jan 18, 2006

%% Do input checking , set defaults and adjust vectors

if nargin<5
    FileOut=[]; 
end
if nargin<4 || isempty(dir)
    dir=0;
elseif ~isscalar(dir)
    error('Parameter ''dir'' must be a scalar expression.')
end
if nargin<3 || isempty(O)
    O=[0 0 0];
elseif numel(O)~=3
    error('Parameter ''O'' must be a 1x3-vector!')
else
    O=O(:)';
end
if nargin<2
    error('Too few parameters for D3trafo execution. Check your inputs!')
end
if ischar(p)    
    load Transformations;
    if ~exist(p,'var')
        error(['Transformation set ',p,' is not defined in Transformations.mat - check your definitions!.'])
    elseif (length(p)~=10)
        error(['Transformation set ',p,' is of wrong size - check your definitions!.'])
    end
    eval(['p=',p,'(1:7);']);
    eval(['O=',p,'(8:10);']);
end
if numel(p)~=7
    error('Parameter ''p'' must be a 1x7-vector!')
else
    p=p(:);
end

% Load input file if specified
if ischar(XYZ)
    XYZ=load(XYZ);
end

if (size(XYZ,1)~=3)&&(size(XYZ,2)~=3)
    error('Coordinate list XYZ must be a nx3-matrix!')
elseif (size(XYZ,1)==3)&&(size(XYZ,2)~=3)
    XYZ=XYZ';
end

%% Do the calculations

% number of coordinate triplets to transform
n=size(XYZ,1);

% Create rotation matrix
c=cos(p(4:6));
s=sin(p(4:6));
D=zeros(3);
D(:,1)=[c(2)*c(3) -c(2)*s(3) s(2)]';
D(:,2)=[s(1)*s(2)*c(3)+c(1)*s(3) -s(1)*s(2)*s(3)+c(1)*c(3) -s(1)*c(2)]';
D(:,3)=[-c(1)*s(2)*c(3)+s(1)*s(3) c(1)*s(2)*s(3)+s(1)*c(3) c(1)*c(2)]';

% Translate the rotation center to coordinate origin
dXYZ=XYZ-repmat(O,n,1);

% Perform transformation
if ~dir
    T=repmat(p(1:3),1,n)+p(7)*D*dXYZ';
else
    T=D'/p(7)*(dXYZ'-repmat(p(1:3),1,n));
end

% Translate the rotation center back again
xyz=T'+repmat(O,n,1);

%% Write output to file if specified

if ~isempty(FileOut)
    fid=fopen(FileOut,'w+');
    fprintf(fid,'%12.6f  %12.6f  %12.6f\n',xyz');
    fclose(fid);
end