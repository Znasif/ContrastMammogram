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
import random as rn
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

def display(patient_info, patient_id, which_breast):
    for j in ["MLO", "CC"]:
        for i in ["full", "mask", "cropped"]:
            file_path = patient_info[patient_id][j][which_breast][i]
            if file_path != "":
                read_file = dcm.dcmread(file_path)
                im = read_file.pixel_array
                if im.shape[0]>1000:
                    im = cv2.resize(im, (im.shape[1]//5, im.shape[0]//5))
                cv2.imshow(i+j, im)
    cv2.waitKey(0)

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

def indices_with_value(image):
    '''image = np.zeros_like(image_)#np.copy(image_)
    print(np.max(image_))
    condition = np.where((image_[:,:,2]>248) & (image_[:,:,1]<100) & (image_[:,:,2]<100))
    print(condition)
    image[condition]=np.array([0, 0, 255])
    return image.astype(np.uint8)'''
    image_ = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])

    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])

    lower_mask = cv2.inRange(image_, lower1, upper1)
    upper_mask = cv2.inRange(image_, lower2, upper2)

    full_mask = lower_mask + upper_mask
    #mapped = get_remapped_index(np.where(full_mask), full_mask.shape)
    result = np.zeros_like(image).astype(np.uint8)
    result[np.where(full_mask)] = 255
    return result

def get_remapped_index(x, shape):
    return (x[0]/shape[0]*3328).astype(np.uint16), (x[1]/shape[1]*4096).astype(np.uint16)

def get_mask(patient_id, jpg_location):
    img = cv2.imread(jpg_location)
    #img = np.full(img.shape, [0, 0, 255])
    img = indices_with_value(img)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img, thresh = cv2.threshold(imgray, 0, 255, 0)
    resized = cv2.resize(imgray, (3328, 4096), interpolation = cv2.INTER_AREA)
    return resized.astype(np.uint8)
    

def get_all(patient_id):
    ims = []
    masks = []
    for breast_view in ["CC", "MLO"]:
        for which_breast in ["LEFT", "RIGHT"]:
            full_location  = patient_info[patient_id][breast_view][which_breast]["full"]
            mask_location  = patient_info[patient_id][breast_view][which_breast]["mask"]
            mask = get_mask(patient_id+"_"+breast_view+"_"+which_breast, mask_location)
            info = dcm.dcmread(full_location)
            side = info['0x20', '0x62'].value
            full = info.pixel_array
            if side not in which_breast:
                full = cv2.flip(full, 1)
                mask = cv2.flip(mask, 1)
            ims.append(full)
            masks.append(mask)
    
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
    plt.savefig("Extracted/"+patient_id+'_original.png', dpi=500, bbox_inches='tight')

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
    plt.savefig("Extracted/"+patient_id+'_mask.png', dpi=500, bbox_inches='tight')

    count = 0
    for ax, im in zip(grid, ims):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap=plt.cm.gray)
        ax.imshow(masks[count], cmap=plt.cm.gray, alpha=0.1)
        count+=1
        ax.set_axis_off()
    plt.savefig("Extracted/"+patient_id+'_overlaid.png', dpi=500, bbox_inches='tight')
    plt.close('all')

def generate_Patient_INFO():
    normal_dir = 'Normal_cases'
    suspicious_dir = 'Suspicious_cases'
    patient_info = {}
    
    for parent_dir in [normal_dir, suspicious_dir]:
        dicom_lst = os.listdir(parent_dir)
        for i in dicom_lst:
            cur_dir = os.path.join(parent_dir, i)
            patient_id = i
            if os.path.isdir(cur_dir):
                if patient_id not in patient_info:
                    patient_info[patient_id] = {"MLO": {"LEFT" : {"full" : "", "mask" : "", "cropped" : ""}, "RIGHT" : {"full" : "", "mask" : "", "cropped" : ""}}, "CC" : {"LEFT" : {"full" : "", "mask" : "", "cropped" : ""}, "RIGHT" : {"full" : "", "mask" : "", "cropped" : ""}}}
                else:
                    patient_id = str(int(patient_id) + 50)
                    patient_info[patient_id] = {"MLO": {"LEFT" : {"full" : "", "mask" : "", "cropped" : ""}, "RIGHT" : {"full" : "", "mask" : "", "cropped" : ""}}, "CC" : {"LEFT" : {"full" : "", "mask" : "", "cropped" : ""}, "RIGHT" : {"full" : "", "mask" : "", "cropped" : ""}}}
                
                dcm_files = [['CC_prior.dcm','CC_recent.dcm'],  ['MLO_prior.dcm',  'MLO_recent.dcm']]
                jpg_files = [['CC_prior_GT.jpg', 'CC_recent_GT.jpg'],['MLO_prior_GT.jpg','MLO_recent_GT.jpg']]
                cur_dicom_dir = os.path.join(cur_dir, dcm_files[0][0])
                info = dcm.dcmread(cur_dicom_dir)
                which_breast_ = info['0x20', '0x62'].value
                which_view = ["CC" , "MLO"]
                for k in range(2):
                    for j in range(2):
                        if k==0:
                            which_breast = "RIGHT" if which_breast_ == "L" else "LEFT"
                        else:
                            which_breast = "LEFT" if which_breast_ == "L" else "RIGHT"
                        
                        patient_info[patient_id][which_view[j]][which_breast]["full"] = os.path.join(cur_dir, dcm_files[j][k])
                        patient_info[patient_id][which_view[j]][which_breast]["mask"] = os.path.join(cur_dir, jpg_files[j][k])
                        patient_info[patient_id][which_view[j]][which_breast]["cropped"] = os.path.join(cur_dir, jpg_files[j][k])
                patient_info[patient_id]["pathology"] = parent_dir
    
    with open("patient_info_dict.json", "w+") as f:
        json.dump(patient_info, f)



def get_regions(patient_id, mask_location):
    return_dict = {}
    #for i in range(len(mask_locations)):
    #mask = mask_locations[i]
    #read_file = dcm.dcmread(mask)
    #im = read_file.pixel_array
    mask = get_mask("anything for now", mask_location)
    if "prior" in mask_location:
        mask = cv2.flip(mask, 1)
    #print(mask)
    #print(im.shape)
    """im2 = cv2.resize(im, (512, 512), cv2.INTER_LINEAR)
    cv2.imshow("Mask", im2)
    cv2.waitKey()
    cv2.destroyAllWindow()"""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = np.zeros((mask.shape[0], mask.shape[1], 3)).astype(np.uint8)
    # Draw all contours
    # -1 signifies drawing all contours
    '''for i in contours:
        cv2.drawContours(image, [i], -1, (rn.randint(0,255), rn.randint(0,255), rn.randint(0,255)), -1)
    
    cv2.imwrite('Contours/'+ patient_id, image)'''
    
    for i in range(len(contours)):
        nps = np.array(contours[i])
        poly_x = [int(k) for k in nps[:, 0, 0]]
        poly_y = [int(k) for k in nps[:, 0, 1]]
        return_dict[str(i)] = {"shape_attributes": {"name": "polygon", "all_points_x":poly_x, "all_points_y":poly_y}, "region_attributes": {}}
        #print(return_dict)
    return return_dict

def generate_dict_for_detectron(patient_info):
    new_dict = {}
    full_dir = os.listdir("Extracted")
    sp = {}
    for i in full_dir:
        selected_patients, seperator, tail = i.partition("_")
        sp[selected_patients] = None
    sp = list(sp.keys())
    for i in sp:
        #print("pid: "+i)
        for breast_view in ["CC", "MLO"]:
            for which_breast in ["LEFT", "RIGHT"]:
                full_location = patient_info[i][breast_view][which_breast]["full"]
                name = i+"_"+breast_view+"_"+which_breast+"_full_.jpg"
                name1 = i+"_"+breast_view+"_"+which_breast+"_mask_.jpg"
                new_dict[name] = {"file_name": full_location, "regions": get_regions(name1, patient_info[i][breast_view][which_breast]["mask"])}
                #print(full_location)
                
    with open("detectron_regions_.json", "w+") as f:
        json.dump(new_dict, f)

def load__Patient_INFO():
    with open("patient_info_dict.json", "r+") as f:
        return json.load(f)

def train_test_split():
    np.random.seed(0)
    train = {}
    test = {}
    full_dir_ = os.listdir("Extracted")
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
                if "prior" in all[name]["file_name"]:
                    px = cv2.flip(px, 1)
                px = cv2.normalize(px, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
                cv2.imwrite("Extracted/train/"+name, px)
    for i in test_:
        for breast_view in ["CC", "MLO"]:
            for which_breast in ["LEFT", "RIGHT"]:
                name = i+"_"+breast_view+"_"+which_breast+"_full_.jpg"
                test[name] = all[name]
                px = dcm.dcmread(all[name]["file_name"]).pixel_array
                if "prior" in all[name]["file_name"]:
                    px = cv2.flip(px, 1)
                px = cv2.normalize(px, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
                cv2.imwrite("Extracted/test/"+name, px)
    
    '''with open("train_.json", "w+") as f:
        json.dump(train, f)
    with open("test_.json", "w+") as f:
        json.dump(test, f)'''


if __name__ == "__main__":
    #generate_Patient_INFO()
    patient_info = load__Patient_INFO()
    '''for i in patient_info:
        cv2.destroyAllWindows()
        get_all(i)'''
    #generate_dict_for_detectron(patient_info)
    train_test_split()
    # lis = []
    # for i in patient_info:
    #     if (verify(i, patient_info)):
    #         #lis.append(i)
    #         #relocate(i, patient_info)
    #         display(patient_info, i, "RIGHT")
    #         display(patient_info, i, "LEFT")
    #         print(i)
    #print(len(lis))