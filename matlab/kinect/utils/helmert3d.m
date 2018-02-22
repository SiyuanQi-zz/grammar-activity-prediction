function [tp,rc,ac,tr,sof]=helmert3d(datum1,datum2,Type,WithOutScale,Approx,NameToSave)

% HERLMERT3D    overdetermined cartesian 3D similarity transformation ("Helmert-Transformation")
%
% [param, rotcent, accur, resid] = helmert3D(datum1,datum2,Type,DontUseScale,ApproxRot,SaveIt)
%
% Inputs:  datum1  n x 3 - matrix with coordinates in the origin datum (x y z)
%                  datum1 may also be a file name with ASCII data to be processed. No point IDs, only
%                  coordinates as if it was a matrix.
%
%          datum2  n x 3 - matrix with coordinates in the destination datum (x y z)
%                  datum2 may also be a file name with ASCII data to be processed. No point IDs, only
%                  coordinates as if it was a matrix.
%                  If either datum1 and/or datum2 are ASCII files, make sure that they are of same
%                  length and contain corresponding points. there is no auto-assignment of points!
%
%            Type  is either '7p' for 7-parameter-transformation (Bursa-Wolf)
%                  or '10p' for 7-parameter-transformation with rotation center at centroid of
%                  datum1 (Molodensky-Badekas).
%                  Default is '7p' which is also choosen if Type is not a string (e.g. []).
%
%    DontUseScale  if this is not 0, do not calculate scale factor but set it to the inputted value
%                  Default: 0 (Use scale)
%
%       ApproxRot  1 x 3 approximate initial values for rotations. If rotation values are too big,
%                  the adjustment may fail if no or bad approximate values are given. Especially this
%                  is the case if rotation around Y-axis is close to multiples of pi/2. You might
%                  purport approximate values to overcome this weakness.
%                  Input as [ex ey ez] in radians.
%                  If omitted or left empty, default is [0 0 0].
%
%          SaveIt  string with the name to save the resulting parameters in Transformations.mat.
%                  Make sure only to use characters which are allowed in Matlab variable names
%                  (e.g. no spaces, umlaute, leading numbers etc.)
%                  If left out or empty, no storage is done.
%                  If the iteration is not converging, no storage is done and a warning is thrown.
%                  If the name to store already is existing, it is not overwritten automatically.
%                  To overwrite an existing name, add '#o' to the name, e.g. 'wgs84_to_local#o'.
%
% Outputs:  param  7 x 1 Parameter set of the 3D similarity transformation
%                      3 translations (x y z) in [Unit of datums]
%                      3 rotations (ex ey ez) in [rad]
%                      1 scale factor
%
%         rotcent  3 x 1 vector of rotation center in datum1 [x y z]
%
%           accur  7 x 1 accuracy of the parameters (or 6 x 1 if scale factor is set to be 1)
%
%           resid  n x 3 - matrix with the residuals datum2 - f(datum1,param)
%
% Used to calculate datum transformation parameters e.g. for global change of reference ellipsoid
% when at least 3 identical points in both datum systems are known.
% 
% ATTENTION: Please be aware of the approximate values-problem mentioned above. In the scope of
%            datum transformation, this is no problem as rotation angles are sufficiently small.
%            This function is trying to use an affine approach to determine proper approximations
%            if more than 3 ID points are given and throws a warning if results become unsecure.
%            (Results may be ambiguous, while either solution is of equal accuracy, though. There
%            may be different sets of rotations leading to nearly the same result.) If no approximate
%            values are given with only 3 ID points, a warning may be thrown if big rotations are
%            probably. But don't rely on that; please check the results for plausibility in any case.
%            If no approximate information is known, but more than 3 ID points are given, you might
%            use an affine transformation (helmertaffine3d) instead.
%
% Parameters can be used with d3trafo.m
% This function needs helmertaffine3d.m

% 04/14/09 Peter Wasmeier - Technische Universität München
% p.wasmeier@bv.tum.de
 
%% Argument checking and defaults
sof = 1;

if nargin<6
    NameToSave=[];
else
    NameToSave=strtrim(NameToSave);
    if strcmp(NameToSave,'#o')
        NameToSave=[];
    end
end

if nargin<5 || isempty(Approx)
    Approx=[0 0 0];
elseif numel(Approx)~=3
    error('ApproxRot needs to be a 3-element vector.')
else
    Approx=Approx(:)';
end

if nargin<4 || isempty(WithOutScale)
    WithOutScale=0;
end
if nargin<3 || ~isstr(Type) || isempty(Type)
    Type='7p'; 
end

% Load input file if specified
if ischar(datum1)
    datum1=load(datum1);
end
if ischar(datum2)
    datum2=load(datum2);
end


if (size(datum1,1)==3)&&(size(datum1,2)~=3)
    datum1=datum1'; 
end
if (size(datum2,1)==3)&&(size(datum2,2)~=3)
    datum2=datum2'; 
end

s1=size(datum1);
s2=size(datum2);
if any(s1~=s2)
    error('The datum sets are not of equal size')
elseif any([s1(2) s2(2)]~=[3 3])
    error('At least one of the datum sets is not 3D')
elseif any([s1(1) s2(1)]<3)
    error('At least 3 points in each datum are necessary for calculating')
end

switch Type
    case '7p'
        rc=[0 0 0];
    case '10p'
        rc=mean(datum1);
    otherwise
        error ('Transformation type needs to be ''7p'' or ''10p''.')
end


%% Adjustment

naeh=[0 0 0 Approx 1];

if all(Approx==[0 0 0]) && s1(1)>3
    x0=helmertaffine3d(datum1,datum2);
    s=(sqrt(x0(4)^2+x0(5)^2+x0(6)^2)+sqrt(x0(7)^2+x0(8)^2+x0(9)^2)+sqrt(x0(10)^2+x0(11)^2+x0(12)^2))/3;
    
    if abs(x0(11))<1e-6 && abs(x0(12))<1e-6
        if x0(10)<0
            ey=3*pi/2;
        else
            ey=pi/2;
        end
        warning('Helmert3D:Ambiguous_rotations','Y-rotation is close to a multiple of pi/2. X- and Z-rotation therefore cannot be approximated.')
        ex=0;
        ez=0; 
    else
        ex=atan2(-x0(11),x0(12));
        if ex<0,ex=ex+2*pi;end
        ey=atan2(x0(10),sqrt((x0(4))^2+(x0(7))^2));
        if ey<0,ey=ey+2*pi;end
        ez=atan2(-x0(7),x0(4));
        if ez<0,ez=ez+2*pi;end
    end
    naeh=[0 0 0 ex ey ez s];
end

if WithOutScale
    naeh(7)=WithOutScale;
end
WertA=[1e-8 1e-8];
zaehl=0;

x0=naeh(1);
y0=naeh(2);
z0=naeh(3);
ex=naeh(4);
ey=naeh(5);
ez=naeh(6);
m=naeh(7);

tp=[x0 y0 z0 ex ey ez m];

Qbb=eye(3*s1(1));

while(1)
    A=zeros(3*s1(1),7);
    for i=1:s1(1)
        A(i*3-2,1)=-1;
        A(i*3-1,2)=-1;
        A(i*3,3)=-1;
        A(i*3-2,4)=-m*((cos(ex)*sin(ey)*cos(ez)-sin(ex)*sin(ez))*(datum1(i,2)-rc(2))+(sin(ex)*sin(ey)*cos(ez)+cos(ex)*sin(ey))*(datum1(i,3)-rc(3)));
        A(i*3-2,5)=-m*((-sin(ey)*cos(ez))*(datum1(i,1)-rc(1))+(sin(ex)*cos(ey)*cos(ez))*(datum1(i,2)-rc(2))+(-cos(ex)*cos(ey)*cos(ez))*(datum1(i,3)-rc(3)));
        A(i*3-2,6)=-m*((-cos(ey)*sin(ez))*(datum1(i,1)-rc(1))+(-sin(ex)*sin(ey)*sin(ez)+cos(ex)*cos(ez))*(datum1(i,2)-rc(2))+(+cos(ex)*sin(ey)*sin(ez)+sin(ex)*cos(ex))*(datum1(i,3)-rc(3)));
        A(i*3-2,7)=-((cos(ey)*cos(ez))*(datum1(i,1)-rc(1))+(sin(ex)*sin(ey)*cos(ez)+cos(ex)*sin(ez))*(datum1(i,2)-rc(2))+(-cos(ex)*sin(ey)*cos(ez)+sin(ex)*sin(ez))*(datum1(i,3)-rc(3)));
        A(i*3-1,4)=-m*((-cos(ex)*sin(ey)*sin(ez)-sin(ex)*cos(ez))*(datum1(i,2)-rc(2))+(-sin(ex)*sin(ey)*sin(ez)+cos(ex)*cos(ez))*(datum1(i,3)-rc(3)));
        A(i*3-1,5)=-m*((sin(ey)*sin(ez))*(datum1(i,1)-rc(1))+(-sin(ex)*cos(ey)*sin(ez))*(datum1(i,2)-rc(2))+(cos(ex)*cos(ey)*sin(ez))*(datum1(i,3)-rc(3)));
        A(i*3-1,6)=-m*((-cos(ey)*cos(ez))*(datum1(i,1)-rc(1))+(-sin(ex)*sin(ey)*cos(ez)-cos(ex)*sin(ez))*(datum1(i,2)-rc(2))+(cos(ex)*sin(ey)*cos(ez)+sin(ex)*sin(ez))*(datum1(i,3)-rc(3)));
        A(i*3-1,7)=-((-cos(ey)*sin(ez))*(datum1(i,1)-rc(1))+(-sin(ex)*sin(ey)*sin(ez)+cos(ex)*cos(ez))*(datum1(i,2)-rc(2))+(cos(ex)*sin(ey)*sin(ez)+sin(ex)*cos(ez))*(datum1(i,3)-rc(3)));
        A(i*3,4)=-m*((-cos(ex)*cos(ey))*(datum1(i,2)-rc(2))+(-sin(ex)*cos(ey))*(datum1(i,3)-rc(3)));
        A(i*3,5)=-m*((cos(ey))*(datum1(i,1)-rc(1))+(-sin(ex)*(-sin(ey)))*(datum1(i,2)-rc(2))+(-cos(ex)*sin(ey))*(datum1(i,3)-rc(3)));
        A(i*3,6)=0;
        A(i*3,7)=-((sin(ey))*(datum1(i,1)-rc(1))+(-sin(ex)*cos(ey))*(datum1(i,2)-rc(2))+(cos(ex)*cos(ey))*(datum1(i,3)-rc(3)));

        w(i*3-2,1)=-rc(1)+datum2(i,1)-x0-m*((cos(ey)*cos(ez))*(datum1(i,1)-rc(1))+(sin(ex)*sin(ey)*cos(ez)+cos(ex)*sin(ez))*(datum1(i,2)-rc(2))+(-cos(ex)*sin(ey)*cos(ez)+sin(ex)*sin(ez))*(datum1(i,3)-rc(3)));
        w(i*3-1,1)=-rc(2)+datum2(i,2)-y0-m*((-cos(ey)*sin(ez))*(datum1(i,1)-rc(1))+(-sin(ex)*sin(ey)*sin(ez)+cos(ex)*cos(ez))*(datum1(i,2)-rc(2))+(cos(ex)*sin(ey)*sin(ez)+sin(ex)*cos(ez))*(datum1(i,3)-rc(3)));
        w(i*3,1)=-rc(3)+datum2(i,3)-z0-m*((sin(ey))*(datum1(i,1)-rc(1))+(-sin(ex)*cos(ey))*(datum1(i,2)-rc(2))+(cos(ex)*cos(ey))*(datum1(i,3)-rc(3)));
    end

    if WithOutScale
        A=A(:,1:6);
    end
    
    warning off;
    w=-1*w;
    r=size(A,1)-size(A,2);
    Pbb=inv(Qbb);
    deltax=inv(A'*Pbb*A)*A'*Pbb*w;
    v=A*deltax-w;
    sig0p=sqrt((v'*Pbb*v)/r);
    Qxxda=inv(A'*Pbb*A);
    Kxxda=sig0p^2*Qxxda;
    ac=sqrt(diag(Kxxda));
    warning on;

    testv=sqrt((deltax(1)^2+deltax(2)^2+deltax(3)^2)/3);
    testd=sqrt((deltax(4)^2+deltax(5)^2+deltax(6)^2)/3);
    zaehl=zaehl+1;
    x0=x0+deltax(1);
    y0=y0+deltax(2);
    z0=z0+deltax(3);
    ex=ex+deltax(4);
    ey=ey+deltax(5);
    ez=ez+deltax(6);
    if ~WithOutScale && (m+deltax(7))>1e-15     % This condition is to prevent numerical problems with m-->0
        m=m+deltax(7);
    end
    tp=[x0 y0 z0 ex ey ez m]';
    if abs(testv) < WertA(1) && abs(testd) < WertA(2)
        break;
    elseif zaehl>1000
        sof = 0;
        warning('Helmert3D:Too_many_iterations','Calculation not converging after 1000 iterations. I am aborting. Results may be inaccurate.')
        break;
    end
end
if any (abs(tp(4:6))>2*pi)
    warning('Helmert3D:Unsufficient_approximation_values','Rotation angles seem to be big. A better approximation is regarded. Results will be inaccurate.')
end

%% Transformation residuals
idz=zeros(s1);
for i=1:s1(1)
    idz(i,2)=rc(2)+tp(2)+tp(7)*((-cos(tp(5))*sin(tp(6)))*(datum1(i,1)-rc(1))+(-sin(tp(4))*sin(tp(5))*sin(tp(6))+cos(tp(4))*cos(tp(6)))*(datum1(i,2)-rc(2))+(cos(tp(4))*sin(tp(5))*sin(tp(6))+sin(tp(4))*cos(tp(6)))*(datum1(i,3)-rc(3)));
    idz(i,1)=rc(1)+tp(1)+tp(7)*((cos(tp(5))*cos(tp(6)))*(datum1(i,1)-rc(1))+(sin(tp(4))*sin(tp(5))*cos(tp(6))+cos(tp(4))*sin(tp(6)))*(datum1(i,2)-rc(2))+(-cos(tp(4))*sin(tp(5))*cos(tp(6))+sin(tp(4))*sin(tp(6)))*(datum1(i,3)-rc(3)));
    idz(i,3)=rc(3)+tp(3)+tp(7)*((sin(tp(5)))*(datum1(i,1)-rc(1))+(-sin(tp(4))*cos(tp(5)))*(datum1(i,2)-rc(2))+(cos(tp(4))*cos(tp(5)))*(datum1(i,3)-rc(3)));
end
tr=datum2-idz;

if ~isempty(NameToSave)
    load Transformations;
    if zaehl>1000
        warning('Helmert3D:Results_too_inaccurate_to_save','Results may be inaccurate and do not get stored.')
    elseif exist(NameToSave,'var') && length(NameToSave)>=2 && ~strcmp(NameToSave(end-1:end),'#o')
        warning('Helmert3D:Parameter_already_exists',['Parameter set ',NameToSave,' already exists and therefore is not stored.'])
    else
        if strcmp(NameToSave(end-1:end),'#o')
            NameToSave=NameToSave(1:end-2);
        end
        if any(regexp(NameToSave,'\W')) || any(regexp(NameToSave(1),'\d'))
            warning('Helmert3D:Parameter_name_invalid',['Name ',NameToSave,' contains invalid characters and therefore is not stored.'])
        else
            eval([NameToSave,'=[',num2str(tp'),' ',num2str(rc),'];']);
            save('Transformations.mat',NameToSave,'-append');
        end
    end
end
