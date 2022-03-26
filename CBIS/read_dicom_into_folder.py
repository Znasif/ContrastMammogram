from asyncore import read
import os
import json
import pydicom as dcm
import cv2
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import multiprocessing as mp

# pfile = pd.read_csv("mass_case_description_train_set.csv")
# print(pfile.head)

def check_subs_in_string(dir, sub):
    flag = True
    for i in sub:
        flag = flag and (i in dir)
    return flag

def info(dir):
    if "LEFT" in dir:
        which_breast = "LEFT"
    if "RIGHT" in dir:
        which_breast = "RIGHT"
    if "MLO" in dir:
        which_view = "MLO"
    if "CC" in dir:
        which_view = "CC"
    return which_breast, which_view

def determinet_type(file1, file2, file3):
    shape1 = dcm.dcmread(file1).pixel_array.shape
    shape2 = dcm.dcmread(file2).pixel_array.shape
    if shape1 == shape2:
        return file2, file3
    else:
        return file3, file2

def display(patient_info, patient_id):
    '''
    show left CC, right CC
         left MLO, right MLO
    overlaid with mask (no need to show cropped)
    '''
    print(patient_id)
    ims = []
    masks = []
    for j in ["CC", "MLO"]:
        for which_breast in ["LEFT", "RIGHT"]:
            file_paths = patient_info[patient_id][j][which_breast]["full"]
            for file_path in file_paths:
                if file_path != "" and "Training" in file_path:
                    read_file = dcm.dcmread(file_path)
                    im = read_file.pixel_array
                    im = cv2.resize(im, (512, 512), cv2.INTER_LINEAR)
                    ims.append(im)
                    break
            file_paths = patient_info[patient_id][j][which_breast]["mask"]
            mask = np.ones((512,512))
            for file_path in file_paths:
                if file_path != "":
                    read_file = dcm.dcmread(file_path)
                    ms = read_file.pixel_array
                    ms = cv2.resize(ms, (512, 512), cv2.INTER_LINEAR)
                    ms = 255-ms
                    mask = mask*ms
            masks.append(255-mask)
    print(len(ims))
    if(len(ims) == 4):
        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )

        count = 0
        for ax, im in zip(grid, ims):
            # Iterating over the grid returns the Axes.
            ax.imshow(im, cmap=plt.cm.gray)
            count+=1
            ax.set_axis_off()
        plt.savefig("Observed/"+patient_id+'_original.png', dpi=500, bbox_inches='tight')

        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )

        count = 0
        for ax, im in zip(grid, ims):
            # Iterating over the grid returns the Axes.
            #ax.imshow(im, cmap=plt.cm.gray)
            ax.imshow(masks[count], cmap=plt.cm.gray)
            count+=1
            ax.set_axis_off()
        plt.savefig("Observed/"+patient_id+'_mask.png', dpi=500, bbox_inches='tight')

        count = 0
        for ax, im in zip(grid, ims):
            # Iterating over the grid returns the Axes.
            ax.imshow(im, cmap=plt.cm.gray)
            ax.imshow(masks[count], cmap=plt.cm.gray, alpha=0.1)
            count+=1
            ax.set_axis_off()
        plt.savefig("Observed/"+patient_id+'_overlaid.png', dpi=500, bbox_inches='tight')

def verify(patient_id, patient_info):
    flag = True
    for i in ["MLO", "CC"]:
        for j in ["LEFT", "RIGHT"]:
            for k in ["full", "mask", "cropped"]:
                flag = flag and (patient_info[patient_id][i][j][k]!="")
    return flag

def relocate(patient_id, patient_info):
    for i in ["MLO", "CC"]:
        for j in ["LEFT", "RIGHT"]:
            for k in ["full", "mask", "cropped"]:
                file_path = patient_info[patient_id][i][j][k]
                dst = "D:\Mammogram CBIS\Extracted_CBIS"
                type_of = "Calc"
                train_or = "Test"
                if "Mass" in file_path:
                    type_of = "Mass"
                if "Training" in file_path:
                    train_or = "Training"
                dst_file = patient_id+"_"+i+"_"+j+"_"+k+".dcm"
                shutil.copy2(file_path, os.path.join(dst, type_of, train_or, dst_file))

def generate_Patient_INFO():
    parent_dir = 'CBIS-DDSM'
    dicom_lst = os.listdir(parent_dir)
    dicom_count_dict = {"Mass" : {"Training": {"full": 0, "mask": 0}, "Test" :{"full": 0, "mask": 0}}, "Calc" : {"Training": {"full": 0, "mask": 0}, "Test" : {"full": 0, "mask": 0}}}
    patient_info = {}
    shapes = {}
    multiples = 0

    for i in dicom_lst:
        cur_dir = os.path.join(parent_dir, i)
        patient_id = i.split("_P_")[1][:5]
        which_breast, which_view = info(i)
        if patient_id not in patient_info:
            patient_info[patient_id] = {"MLO": {"LEFT" : {"full" : [], "mask" : [], "cropped" : []}, "RIGHT" : {"full" : [], "mask" : [], "cropped" : []}}, "CC" : {"LEFT" : {"full" : [], "mask" : [], "cropped" : []}, "RIGHT" : {"full" : [], "mask" : [], "cropped" : []}}}
            shapes[patient_id] = {}
        for j in os.listdir(cur_dir):
            cur_sub_dir = os.path.join(cur_dir, j)
            for k in os.listdir(cur_sub_dir):
                type_of = -1
                cur_dicom_dir = os.path.join(cur_sub_dir, k)
                temp = os.listdir(cur_dicom_dir)
                dicom_cnt = len(temp)
                if check_subs_in_string(cur_dicom_dir, ["Mass", "Training", "full"]):
                    dicom_count_dict["Mass"]["Training"]["full"] += dicom_cnt
                    type_of = 0
                if check_subs_in_string(cur_dicom_dir, ["Mass", "Training", "mask"]):
                    dicom_count_dict["Mass"]["Training"]["mask"] += dicom_cnt
                    type_of = dicom_cnt
                if check_subs_in_string(cur_dicom_dir, ["Mass", "Test", "full"]):
                    dicom_count_dict["Mass"]["Test"]["full"] += dicom_cnt
                    type_of = 0
                if check_subs_in_string(cur_dicom_dir, ["Mass", "Test", "mask"]):
                    dicom_count_dict["Mass"]["Test"]["mask"] += dicom_cnt
                    type_of = dicom_cnt
                if check_subs_in_string(cur_dicom_dir, ["Calc", "Training", "full"]):
                    dicom_count_dict["Calc"]["Training"]["full"] += dicom_cnt
                    type_of = 0
                if check_subs_in_string(cur_dicom_dir, ["Calc", "Training", "mask"]):
                    dicom_count_dict["Calc"]["Training"]["mask"] += dicom_cnt
                    type_of = dicom_cnt
                if check_subs_in_string(cur_dicom_dir, ["Calc", "Test", "full"]):
                    dicom_count_dict["Calc"]["Test"]["full"] += dicom_cnt
                    type_of = 0
                if check_subs_in_string(cur_dicom_dir, ["Calc", "Test", "mask"]):
                    dicom_count_dict["Calc"]["Test"]["mask"] += dicom_cnt
                    type_of = dicom_cnt
                if(type_of == 0):
                    patient_info[patient_id][which_view][which_breast]["full"].append(os.path.join(cur_dicom_dir, temp[0]))
                elif(type_of == 1):
                    patient_info[patient_id][which_view][which_breast]["mask"].append(os.path.join(cur_dicom_dir, temp[0]))
                elif(type_of == 2):
                    file2 = os.path.join(cur_dicom_dir, temp[0])
                    file3 = os.path.join(cur_dicom_dir, temp[1])
                    if "Calc" in file2:
                        patient_info[patient_id][which_view][which_breast]["mask"].append(file3)
                        patient_info[patient_id][which_view][which_breast]["cropped"].append(file2)
                    elif "Mass" in file2:
                        patient_info[patient_id][which_view][which_breast]["mask"].append(file3)
                        patient_info[patient_id][which_view][which_breast]["cropped"].append(file2)

    with open("dicom_count_dict.json", "w+") as f:
        json.dump(dicom_count_dict, f)

    with open("patient_info_dict.json", "w+") as f:
        json.dump(patient_info, f)

def get_regions(mask_locations):
    return_dict = {}
    for i in range(len(mask_locations)):
        mask = mask_locations[i]
        read_file = dcm.dcmread(mask)
        im = read_file.pixel_array
        #print(mask)
        #print(im.shape)
        """im2 = cv2.resize(im, (512, 512), cv2.INTER_LINEAR)
        cv2.imshow("Mask", im2)
        cv2.waitKey()
        cv2.destroyAllWindow()"""
        contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(len(contours[0]))
        nps = np.array(contours[0])
        poly_x = [int(k) for k in nps[:, 0, 0]]
        poly_y = [int(k) for k in nps[:, 0, 1]]
        return_dict[str(i)] = {"shape_attributes": {"name": "polygon", "all_points_x":poly_x, "all_points_y":poly_y}, "region_attributes": {}}
        #print(return_dict)
    return return_dict

def generate_dict_for_detectron(patient_info):
    new_dict = {}
    full_dir = os.listdir("Observed")
    sp = {}
    for i in full_dir:
        selected_patients, seperator, tail = i.partition("_")
        sp[selected_patients] = None
    sp = sp.keys()    
    for i in sp:
        
        for breast_view in ["CC", "MLO"]:
            for which_breast in ["LEFT", "RIGHT"]:
                file_names = patient_info[i][breast_view][which_breast]["full"]
                full_location = file_names[0]
                for j in file_names:
                    if "Training" in j:
                        full_location = j
                new_dict[i+"_"+breast_view+"_"+which_breast+"_full_.jpg"] = {"file_name": full_location, "regions": get_regions(patient_info[i][breast_view][which_breast]["mask"])}
            

    with open("detectron_regions_.json", "w+") as f:
        json.dump(new_dict, f)


def load__Patient_INFO():
    with open("patient_info_dict.json", "r+") as f:
        return json.load(f)


def train_test_split():
    np.random.seed(0)
    train = {}
    test = {}
    full_dir_ = os.listdir("Observed")
    full_dir = []
    for i in full_dir_:
        if ('.png' in i):
            full_dir.append(i)
    sp = {}
    for i in full_dir:
        selected_patients, seperator, tail = i.partition("_")
        sp[selected_patients] = None
    sp = np.array(list(sp.keys()))
    np.random.shuffle(sp)
    f = int(sp.shape[0]*0.8)
    train_, test_ = sp[:f], sp[f:]
    with open("detectron_regions_.json", "r+") as f:
        all = json.load(f)
    for i in train_:
        for breast_view in ["CC", "MLO"]:
            for which_breast in ["LEFT", "RIGHT"]:
                name = i+"_"+breast_view+"_"+which_breast+"_full_.jpg"
                train[name] = all[name]
                px = dcm.dcmread(all[name]["file_name"]).pixel_array
                px = cv2.normalize(px, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
                cv2.imwrite("Observed/train/"+name, px)
    
    for i in test_:
        for breast_view in ["CC", "MLO"]:
            for which_breast in ["LEFT", "RIGHT"]:
                name = i+"_"+breast_view+"_"+which_breast+"_full_.jpg"
                test[name] = all[name]
                px = dcm.dcmread(all[name]["file_name"]).pixel_array
                px = cv2.normalize(px, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
                cv2.imwrite("Observed/test/"+name, px)

    
    with open("train_.json", "w+") as f:
        json.dump(train, f)
    with open("test_.json", "w+") as f:
        json.dump(test, f)


if __name__ == "__main__":
    #generate_Patient_INFO()
    patient_info = load__Patient_INFO()
    ls = []
    #display(patient_info, "00034")
    #generate_dict_for_detectron(patient_info)
    train_test_split()

    '''for i in patient_info:
        ls.append(mp.Process(target=display, args=(patient_info, i,)))
    for i in ls:
        i.start()'''
    '''lis = []
    for i in patient_info:
        if (verify(i, patient_info)):
            #lis.append(i)
            #relocate(i, patient_info)
            display(patient_info, i, "RIGHT")
            display(patient_info, i, "LEFT")
            print(i)
    print(len(lis))'''