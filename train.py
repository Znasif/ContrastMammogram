from src.utils import dataloader, visualization, preprocessing
#from src.model import AGRCNN
import cv2
import torch
import os

def training_loop():
  pass

if __name__ == "__main__":
  selected = "Temporal" #"CBIS"#
  parent_dir = os.path.join("/content/drive/MyDrive/", selected)
  train_dict , _ = dataloader.registerCatalogs(parent_dir)

  train_dir, test_dir = selected+"_train", selected+"_test"
  #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #CBIS_trainer = AGRCNN(train, test, os.path.join(parent_dir, "outputs"))