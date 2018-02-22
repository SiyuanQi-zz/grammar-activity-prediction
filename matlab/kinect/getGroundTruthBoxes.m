function objTrackedPos = getGroundTruthBoxes( dataroot, datadir, objects, s )
%GETGROUNDTRUTH Summary of this function goes here
%   Detailed explanation goes here

load(fullfile(dataroot, '..', 'gt', [datadir, '.mat']));
objTrackedPos = zeros(s);

for objectIndex = 1:length(objects)
    objectName = objects{objectIndex};
    for annotationIndex = 1:length(annotations)
        annotation = annotations{annotationIndex};
        if strcmp(annotation.label, objectName)
            bbox = [annotation.xtl, annotation.ytl, annotation.xbr-annotation.xtl, annotation.ybr-annotation.ytl];
            objTrackedPos(objectIndex, annotation.frame+1, :) = bbox;
        end
    end
end


end

