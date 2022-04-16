import os
import numpy as np
import pandas as pd
import json
from detectron2.data import MetadataCatalog, DatasetCatalog
from src.util.preprocessing import pixel_belongs_to, gen_bin_mask
from scipy import stats


def get_mammograms_dicts(data_dir, is_train = True, dim=800):
    csvs = ["mass_case_description_train_set.csv", "calc_case_description_train_set.csv", "mass_case_description_test_set.csv", "calc_case_description_test_set.csv"]
    pds = []

    for i in csvs:
        csv_data = pd.read_csv(os.path.join(data_dir, i))
        pds.append(csv_data)

    lesion_mapping = {"mass": 0, "calcification":1}
    view_point_count = {"CC": 9, "MLO": 10}
    node_point_count = {11: 101, 10: 82, 9: 65, 8:50, 7:37, 6:26, 5: 17, 4:10, 3:5, 2:2, 1:2}
    pathology_mapping = {"MALIGNANT": 1, "BENIGN_WITHOUT_CALLBACK": 0, "BENIGN":0}
    
    json_file = "train.json" if is_train else "test.json"
    image_dir = "train" if is_train else "test"

    with open(os.path.join(data_dir, json_file)) as f:
        imgs_anns = json.load(f)
    
    dataset_dicts = []
    for i in imgs_anns:
        record = {}
        v = imgs_anns[i]
        patient_info = i.split("_")

        filename = os.path.join(data_dir, image_dir, i)
        height, width = v["height"], v["width"]

        id = patient_info[0]
        record["file_name"] = filename
        record["image_id"] = i
        record["height"] = height
        record["width"] = width
        record["view"] = patient_info[1]
        record["breast"] = patient_info[2]
        pixel_belongs_arr = pixel_belongs_to(filename, record["view"], record["breast"], view_point_count[record["view"]], dim)

        categories = {}
        pathology, density = None, None

        for j in range(4):
            curr_pd = pds[j]
            from_pd = curr_pd[(curr_pd['patient_id'] == "P_"+id) & (curr_pd["image view"] == record["view"]) & (curr_pd["left or right breast"] == record["breast"])]
            if(from_pd.shape[0]>0):
                categories.append(lesion_mapping[from_pd.iloc[0, 5]])
                pathology = pathology_mapping[from_pd.iloc[0, 9]]
                density = from_pd.iloc[0, 1]

        record["pathology"] = pathology
        record["density"] = density
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]

            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            mask_arr = gen_bin_mask(poly, height, width, dim)*pixel_belongs_arr
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": categories[0],
                "graph_node": stats.mode(mask_arr[mask_arr>0])[0]
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def registerCatalogs():
    DatasetCatalog.clear()
    for d in ["train", "test"]:
        bool_map = {"train": True, "test": False}
        DatasetCatalog.register("lesion_" + d, lambda d=d: get_mammograms_dicts("/content/drive/MyDrive/CBIS/", bool_map[d]))
        MetadataCatalog.get("lesion_" + d).set(thing_classes=["mass", "calc"])

    lesion_test_dicts = get_mammograms_dicts("/content/drive/MyDrive/CBIS/", False)
    lesion_train_dicts = get_mammograms_dicts("/content/drive/MyDrive/CBIS/")

