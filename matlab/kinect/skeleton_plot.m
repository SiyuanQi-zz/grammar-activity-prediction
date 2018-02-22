function [] = skeleton_plot(skeleton)
line1 = [1,2,3];
line2 = [4,5,12];
line3 = [6,7,13];
line4 = [3,8,9,14];
line5 = [3,10,11,15];
line6 = [4,6,3,4];
    plot3(skeleton(line1*3-2),skeleton(line1*3),skeleton(line1*3-1),'b');
    hold on;
    plot3(skeleton(line2*3-2),skeleton(line2*3),skeleton(line2*3-1),'y');
    hold on;
    plot3(skeleton(line3*3-2),skeleton(line3*3),skeleton(line3*3-1),'y');
    hold on;
    plot3(skeleton(line4*3-2),skeleton(line4*3),skeleton(line4*3-1),'r');
    hold on;
    plot3(skeleton(line5*3-2),skeleton(line5*3),skeleton(line5*3-1),'r');
    hold on;
    plot3(skeleton(line6*3-2),skeleton(line6*3),skeleton(line6*3-1),'g');
    set(gca,'XLim',[-0.5 0.5]);
    set(gca,'YLim',[1.2 2.2]);
    set(gca,'ZLim',[-0.5 0.5]);
end