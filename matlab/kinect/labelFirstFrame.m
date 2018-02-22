initialize;

mkdir(fullfile(resultroot, 'initial_boxes'));

frame = 0;
fig = figure();
% %% Label the objects in the first frame
for dirIndex = 1:length(subDirs)
    datadir = subDirs(dirIndex).name;
    datapath = [dataroot, datadir, '/'];
    
%     imgName = sprintf([datapath, 'aligned_rgb_%05d.png'], frame);
    imgName = sprintf([datapath, 'raw_rgb_%05d.bmp'], frame);
    handle = imshow(imread(imgName));
    
    initialBoxes = zeros(length(objects), 4);
    for objIndex = 1:length(objects)
        title(objects{objIndex});

        initialBoxes(objIndex, :) = getrect(fig);
%         rectangle('Position', initialBoxes(objIndex, :), 'EdgeColor','r', 'LineWidth',3);
    end
    save(fullfile(resultroot, 'initial_boxes', [datadir, '.mat']), 'initialBoxes');
end

close all;