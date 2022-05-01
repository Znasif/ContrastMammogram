import os
from detectron2 import data
import numpy as np
import pandas as pd
import json, copy, torch, tqdm
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from src.utils.preprocessing import pixel_belongs_to, gen_bin_mask, adjacency, adjacency_, contra_close, get_output_shape
from src.utils.visualization import MammViz
from scipy import stats
from detectron2.data.detection_utils import read_image, check_image_size
import detectron2.data.transforms as T
from detectron2.config import configurable

class MammogramMapper(DatasetMapper):
  def __init__(self, cfg):
      super().__init__(cfg, True)

  def __call__(self, dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # USER: Write your own image loading if it's not from a file
    image = read_image(dataset_dict["file_name"], format=self.image_format)
    auxiliary = read_image(dataset_dict["auxiliary_file_name"], format=self.image_format)
    contralateral = read_image(dataset_dict["contralateral_file_name"], format=self.image_format)
    check_image_size(dataset_dict, image)

    aug_input = T.AugInput(image, sem_seg=None)
    transforms = self.augmentations(aug_input)
    image = aug_input.image

    aug_input = T.AugInput(auxiliary, sem_seg=None)
    transforms = self.augmentations(aug_input)
    auxiliary = aug_input.image

    aug_input = T.AugInput(contralateral, sem_seg=None)
    transforms = self.augmentations(aug_input)
    contralateral = aug_input.image

    image_shape = image.shape[:2]  # h, w
    # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
    # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
    # Therefore it's important to use torch.Tensor.
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.astype("float32").transpose(2, 0, 1)))
    dataset_dict["auxiliary"] = torch.as_tensor(np.ascontiguousarray(auxiliary.astype("float32").transpose(2, 0, 1)))
    dataset_dict["contralateral"] = torch.as_tensor(np.ascontiguousarray(contralateral.astype("float32").transpose(2, 0, 1)))
    if not self.is_train:
        # USER: Modify this if you want to keep them for some reason.
        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        return dataset_dict
    if "annotations" in dataset_dict:
      self._transform_annotations(dataset_dict, transforms, image_shape)
    return dataset_dict

def get_aux_contra(examined_path):
    examined = examined_path.split("_")
    view_options = {"MLO": "CC", "CC":"MLO"}
    breast_options = {"LEFT": "RIGHT", "RIGHT":"LEFT"}
    auxiliary = examined.copy()
    auxiliary[1] = view_options[examined[1]]
    contralateral = examined.copy()
    contralateral[2] = breast_options[examined[2]]
    auxiliary_path = '_'.join(auxiliary)
    contralateral_path = '_'.join(contralateral)
    return auxiliary_path, contralateral_path

def get_anno_obj(filename, view, breast, height, width, annos, global_mapping_array_dict, global_correspondence_size, category, dim):
    if filename in global_correspondence_size:
      return global_correspondence_size[filename]
    objs = []
    mmviz = MammViz(0, "/content/drive/MyDrive/temp_ims/ims")
    mask_gl = np.zeros(dim)
    Flag = False
    for _, anno in annos.items():
        assert not anno["region_attributes"]
        anno = anno["shape_attributes"]
        px = anno["all_points_x"]
        py = anno["all_points_y"]
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly_ = np.array([(x, y) for x, y in zip(px, py)])
        mask_arr_ex, lesion_size = gen_bin_mask(poly_, view, breast, height, width, dim)
        mask_gl += mask_arr_ex
        mask_arr_ex *= global_mapping_array_dict[filename]
        poly = [p for x in poly for p in x]
        node = 0
        try:
          node = stats.mode(mask_arr_ex[mask_arr_ex>0])[0][0]
        except:
          print(lesion_size)
          Flag = True
        obj = {
            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": category,
            "graph_examined_node": node,
            "lesion_size": lesion_size
        }
        objs.append(obj)
    if Flag:
        flag_name = filename.split("/")[-1].split(".jpg")[0]
        mmviz.vis_func((1-mask_gl)*global_mapping_array_dict[filename], flag_name)
        mmviz.vis_func((mask_gl)*255, flag_name)
    #objs = [i for i in objs if (i["graph_examined_node"]>0 and len(i["segmentation"])>5)]
    objs = sorted(objs, key=lambda x:x["lesion_size"])
    global_correspondence_size[filename] = objs
    return objs

def get_mapping_arrays(global_mapping_array_dict, record, dim):
    view_point_count = {"CC": 9, "MLO": 10}
    
    if(record["file_name"] in global_mapping_array_dict):
      pixel_belongs_arr_examined = global_mapping_array_dict[record["file_name"]]
    else:
      pixel_belongs_arr_examined = pixel_belongs_to(record["file_name"], record["view"], record["breast"], view_point_count[record["view"]], dim)
      global_mapping_array_dict[record["file_name"]] = pixel_belongs_arr_examined
    
    if(record["auxiliary_file_name"] in global_mapping_array_dict):
      pixel_belongs_arr_auxiliary = global_mapping_array_dict[record["auxiliary_file_name"]]
    else:
      aux_view = "MLO" if record["view"] == "CC" else "CC"
      pixel_belongs_arr_auxiliary = pixel_belongs_to(record["auxiliary_file_name"], aux_view, record["breast"], view_point_count[aux_view], dim)
      global_mapping_array_dict[record["auxiliary_file_name"]] = pixel_belongs_arr_auxiliary
    
    if(record["contralateral_file_name"] in global_mapping_array_dict):
      pixel_belongs_arr_contralateral = global_mapping_array_dict[record["contralateral_file_name"]]
    else:
      contra_breast = "LEFT" if record["breast"] == "RIGHT" else "RIGHT"
      pixel_belongs_arr_contralateral = pixel_belongs_to(record["contralateral_file_name"], record["view"], contra_breast, view_point_count[record["view"]], dim)
      global_mapping_array_dict[record["contralateral_file_name"]] = pixel_belongs_arr_contralateral

    record["map_examined"] = pixel_belongs_arr_examined
    record["map_auxiliary"] = pixel_belongs_arr_auxiliary
    record["map_contralateral"] = pixel_belongs_arr_contralateral
    return

def get_mammograms_dicts(data_dir, is_train = True, dim=800):
    csvs = ["mass_case_description_train_set.csv", "calc_case_description_train_set.csv", "mass_case_description_test_set.csv", "calc_case_description_test_set.csv"]
    pds = []
    if ("CBIS" in data_dir):
      for i in csvs:
          csv_data = pd.read_csv(os.path.join(data_dir, i))
          pds.append(csv_data)
    
    global_mapping_array_dict = {}
    global_correspondence_size = {}
    lesion_mapping = {"mass": 0, "calcification":1}
    
    node_point_count = {11: 101, 10: 82, 9: 65, 8:50, 7:37, 6:26, 5: 17, 4:10, 3:5, 2:2, 1:2}
    pathology_mapping = {"MALIGNANT": 1, "BENIGN_WITHOUT_CALLBACK": 0, "BENIGN":0}
    
    json_file = "train.json" if is_train else "test.json"
    image_dir = "train" if is_train else "test"

    with open(os.path.join(data_dir, json_file)) as f:
        imgs_anns = json.load(f)
    
    dataset_dicts = []
    for i in tqdm.tqdm(imgs_anns):
        npy_pth = os.path.join("/content/drive/MyDrive/temp_ims/", i+".npy")
        if os.path.exists(npy_pth):
          record = torch.load(npy_pth)
          dataset_dicts.append(record)
          outside = [i for i in record["annotations"] if (len(i["segmentation"][0])>5 and i["graph_examined_node"]>0)]
          diff = len(record["annotations"])-len(outside)
          if diff:
            record["annotations"] = outside
            torch.save(record, os.path.join("/content/drive/MyDrive/temp_ims/", i+".npy"))
            print(i, diff)
          continue
        record = {}
        v = imgs_anns[i]
        patient_info = i.split("_")

        filename = os.path.join(data_dir, image_dir, i)
        height, width = v["height"], v["width"]
        dim_ = get_output_shape((height, width), dim)
        id = patient_info[0]
        record["file_name"] = filename
        record["image_id"] = i
        record["height"] = height
        record["width"] = width
        record["view"] = patient_info[1]
        record["breast"] = patient_info[2]
        ax_f, cn_f = get_aux_contra(i)
        record["auxiliary_file_name"], record["contralateral_file_name"] = os.path.join(data_dir, image_dir, ax_f), os.path.join(data_dir, image_dir, cn_f)
        get_mapping_arrays(global_mapping_array_dict, record, dim_)
        categories_ex = []
        categories_ax = []
        categories_cn = []
        pathology, density = None, None
        aux_view = "MLO" if record["view"] == "CC" else "CC"
        contra_breast = "LEFT" if record["breast"] == "RIGHT" else "LEFT"
        if "CBIS" in data_dir:
          for j in range(4):
              curr_pd = pds[j]
              from_pd = curr_pd[(curr_pd['patient_id'] == "P_"+id) & (curr_pd["image view"] == record["view"]) & (curr_pd["left or right breast"] == record["breast"])]
              if(from_pd.shape[0]>0):
                  categories_ex.append(lesion_mapping[from_pd.iloc[0, 5]])
                  pathology = pathology_mapping[from_pd.iloc[0, 9]]
                  density = from_pd.iloc[0, 1]
              from_pd = curr_pd[(curr_pd['patient_id'] == "P_"+id) & (curr_pd["image view"] == aux_view) & (curr_pd["left or right breast"] == record["breast"])]
              if(from_pd.shape[0]>0):
                  categories_ax.append(lesion_mapping[from_pd.iloc[0, 5]])
              from_pd = curr_pd[(curr_pd['patient_id'] == "P_"+id) & (curr_pd["image view"] == record["view"]) & (curr_pd["left or right breast"] == contra_breast)]
              if(from_pd.shape[0]>0):
                  categories_cn.append(lesion_mapping[from_pd.iloc[0, 5]])
        else:
          pathology = v["pathology"]
          density = 0
          categories_ex, categories_ax, categories_cn = [1], [1], [1]
        record["pathology"] = pathology
        record["density"] = density
        annos = v["regions"]
        ex_obj = get_anno_obj(filename, record["view"], record["breast"], height, width, annos, global_mapping_array_dict, global_correspondence_size, categories_ex[0], dim_)
        dim_ = get_output_shape((imgs_anns[ax_f]["height"], imgs_anns[ax_f]["width"]), dim)
        aux_obj = get_anno_obj(record["auxiliary_file_name"], aux_view, record["breast"], imgs_anns[ax_f]["height"], imgs_anns[ax_f]["width"], imgs_anns[ax_f]["regions"], global_mapping_array_dict, global_correspondence_size, categories_ax[0], dim_)
        record["annotations"] = ex_obj
        if "CBIS" in data_dir:
          epsilon = adjacency(record["view"], ex_obj, aux_obj)
        else:
          epsilon = adjacency_(record["view"], ex_obj, aux_obj)
        if record["view"] == "MLO":
          J = contra_close(2, epsilon.shape[1])
        else:
          J = contra_close(2, epsilon.shape[0])
        record["epsilon"] = epsilon
        record["J"] = J
        torch.save(record, os.path.join("/content/drive/MyDrive/temp_ims/", i+".npy"))
        dataset_dicts.append(record)
    #torch.save(dataset_dicts, "/content/drive/MyDrive/temp_ims/dataset_"+data_dir.split("/")[-1]+".npy")
    return dataset_dicts

def registerCatalogs(parent_dir):
    lesion = parent_dir.split("/")[-1]+"_"
    DatasetCatalog.clear()
    for d in ["train", "test"]:
        bool_map = {"train": True, "test": False}
        DatasetCatalog.register(lesion + d, lambda d=d: get_mammograms_dicts(parent_dir, bool_map[d]))
        MetadataCatalog.get(lesion + d).set(thing_classes=["mass", "calc"])
    lesion_train = get_mammograms_dicts(parent_dir)
    lesion_test = get_mammograms_dicts(parent_dir, False)
    return lesion_train, lesion_test