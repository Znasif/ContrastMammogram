from src.utils import dataloader, visualization, preprocessing
import src.model
from src.model.agrcnn import *
import os, json, cv2, torch, time, random
from google.colab.patches import cv2_imshow
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import Visualizer, ColorMode
import gc


if __name__ == "__main__":
  gc.collect()
  trainer = None
  predictor = None
  with torch.no_grad():
    torch.cuda.empty_cache()
  
  selected = "CBIS"#"Temporal" #

  parent_dir = os.path.join("/content/drive/MyDrive/", selected)
  train_dict , _, _ = dataloader.registerCatalogs(parent_dir)
  train_dir, test_dir = selected+"_train", selected+"_val"

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  trainer = AGRCNN_Trainer(train_dir, test_dir, os.path.join(parent_dir, "outputs"))
  trainer.resume_or_load(resume=True)
  trainer.train()