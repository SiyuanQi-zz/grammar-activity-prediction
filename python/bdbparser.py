"""
Created on Feb 19, 2017

@author: Siyuan Huang

Process the bounding boxes of CAD120, save them as XML files.

"""

import xml.etree.ElementTree as ET
import glob
import Image
import numpy as np


def data_generate():
    ori_address = '/home/siyuan/Documents/Dataset/CAD120/'
    des_address = ori_address + 'CAD120/'
    namelist = ['Subject1', 'Subject3', 'Subject4', 'Subject5']
    image_index = 0
    category_total = []
    for i in range(len(namelist)):
        imagefile = ori_address + namelist[i] + '_rgbd_images'
        annotationfile = ori_address + namelist[i] + '_annotations'
        subaction = glob.glob(imagefile + '/*')
        for j in range(len(subaction)):
            subvideo = glob.glob(subaction[j] + '/*')
            # get the video information
            video_info = subaction[j].split('/')
            video_annotation_dir = annotationfile + '/' + video_info[-1] + '/'
            video_info_dic = activitylabel_parsing(video_annotation_dir)
            # get the category information
            for video in video_info_dic:
                for category in video_info_dic[video]:
                    if category not in category_total:
                        category_total.append(category)
            print category_total
            for k in range(len(subvideo)):
                image_namelist = glob.glob(subvideo[k] + '/*.png')
                rgb_namelist = []
                video_id = subvideo[k].split('/')[-1]
                boundingbox = labeling_parsing(video_annotation_dir, video_id, video_info_dic)
                for t in range(len(image_namelist)):  # choose the RGB image
                    image_name = image_namelist[t].split('/')[-1]
                    if image_name[0] == 'R':
                        rgb_namelist.append(image_namelist[t])
                rgb_namelist = file_sort(rgb_namelist)
                for t in range(len(rgb_namelist)):
                    image_index += 1
                    des = des_address + 'JPEGImages/' + '%06d' % image_index + '.jpg'
                    xml_des = des_address + 'Annotations/' + '%06d' % image_index + '.xml'
                    r = create_xml(xml_des, boundingbox[t], image_index)
                    if r == 0:
                        im = Image.open(rgb_namelist[t])
                        im.save(des)
                    elif r == -1:
                        image_index -= 1
                    # print rgb_namelist[t]
    imageset_generate(des_address, image_index)


def imageset_generate(path, num):
    random_list = np.random.permutation(num)
    random_list += 1
    num_test = int(0.5 * num)
    num_trainval = num - num_test
    num_train = int(0.5 * num_trainval)
    with open(path + 'ImageSets/Main/test.txt', 'w') as f:
        for index in random_list[0:num_test]:
            f.write('%06d\n' % index)
    f.close()
    with open(path + 'ImageSets/Main/trainval.txt', 'w') as f:
        for index in random_list[num_test:]:
            f.write('%06d\n' % index)
    f.close()
    with open(path + 'ImageSets/Main/train.txt', 'w') as f:
        for index in random_list[num_test:num_test+num_train]:
            f.write('%06d\n' % index)
    f.close()
    with open(path + 'ImageSets/Main/val.txt', 'w') as f:
        for index in random_list[num_test+num_train:]:
            f.write('%06d\n' % index)
    f.close()


def create_xml(path, info, image_index):
    out = ET.Element('annotation')
    folder = ET.SubElement(out, 'folder')
    folder.text = 'VOC2007'
    filename = ET.SubElement(out, 'filename')
    filename.text = '%06d' % image_index + '.jpg'
    file_source = ET.SubElement(out, 'source')
    database = ET.SubElement(file_source, 'database')
    database.text = 'CAD Database'
    annotation = ET.SubElement(file_source, 'annotation')
    annotation.text = 'CAD'
    image = ET.SubElement(file_source, 'image')
    image.text = 'flickr'
    flickid = ET.SubElement(file_source, 'flickrid')
    flickid.text = 'Siyuan Huang'
    owner = ET.SubElement(out, 'owner')
    flickid = ET.SubElement(owner, 'flickrid')
    flickid.text = 'Siyuan Huang'
    name = ET.SubElement(owner, 'name')
    name.text = 'Siyuan Huang'
    file_size = ET.SubElement(out, 'size')
    file_width = ET.SubElement(file_size, 'width')
    file_width.text = str(640)
    file_height = ET.SubElement(file_size, 'height')
    file_height.text = str(480)
    file_depth = ET.SubElement(file_size, 'depth')
    file_depth.text = str(3)
    file_segmented = ET.SubElement(out, 'segmented')
    file_segmented.text = '0'
    for object in info:
        obj = ET.SubElement(out, 'object')
        obj_name = ET.SubElement(obj, 'name')
        obj_name.text = object
        obj_pose = ET.SubElement(obj, 'pose')
        obj_pose.text = 'unspecified'
        obj_truncated = ET.SubElement(obj, 'truncated')
        obj_truncated.text = '1'
        obj_difficult = ET.SubElement(obj, 'difficult')
        obj_difficult.text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        if info[object]['x1'] == '0' and info[object]['y1'] == '0' and info[object]['x2'] == '0' and info[object]['y2'] == '0':
            return -1
        xmin.text = str(info[object]['x1'])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(info[object]['y1'])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(info[object]['x2'])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(info[object]['y2'])
    out_tree = ET.ElementTree(out)
    out_tree.write(path)
    return 0


def labeling_parsing(path, video_id, info):
    obj_list = info[video_id]
    num_obj = len(obj_list)
    bdb = []
    for i in range(num_obj):
        with open(path + video_id + '_obj' + str(i+1) + '.txt', 'r') as f:
            frame_num = 0
            for line in f:
                if i == 0:
                    bdb.append({})
                bdb[frame_num][obj_list[i]] = {}
                line = line.split(',')
                bdb[frame_num][obj_list[i]]['x1'] = line[2]
                bdb[frame_num][obj_list[i]]['y1'] = line[3]
                bdb[frame_num][obj_list[i]]['x2'] = line[4]
                bdb[frame_num][obj_list[i]]['y2'] = line[5]
                frame_num += 1
    return bdb


def activitylabel_parsing(path):
    info = {}
    with open(path + 'activityLabel.txt', 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            line = line.split(',')
            obj_num = int(line[-2][0])
            obj_list = [line[-obj_num-1+i][2:] for i in range(obj_num)]
            info[line[0]] = obj_list
    return info


def file_sort(imagelist):
    new_imagelist = []
    for i in range(len(imagelist)):
        temp = imagelist[i].split('RGB')
        new_imagelist.append(temp[0] + 'RGB_' + str(i + 1) + '.png')
    return new_imagelist


def main():
    data_generate()
    # labeling_parsing('/home/siyuan/Documents/Dataset/CAD120/Subject1_annotations/stacking_objects/', '0510182057', {'0510182057':['plate']})


if __name__ == '__main__':
    main()
