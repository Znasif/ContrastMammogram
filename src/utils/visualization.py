import cv2
import numpy as np
import random
from google.colab.patches import cv2_imshow
import os
from datetime import datetime

class MammViz():
    def __init__(self, option, parent_dir):
      self.parent_dir = parent_dir
      self.option = option
      pass

    def visualize_landmarks(self, pseudo, dim):
        img_arr = np.zeros(dim)
        for i in pseudo:
            cv2.circle(img_arr, i, 4, 255, 1)
        self.vis_func(img_arr)
    
    def visualize_regions(self, img_arr, knn_arr):
        pseudo = np.unique(knn_arr.flatten()).shape[0]
        colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for i in range(pseudo)]
        colors[0] = (0, 0, 0)
        dim = knn_arr.shape
        overlay_arr = np.zeros_like(img_arr)
        for i in range(dim[0]):
            for j in range(dim[1]):
                overlay_arr[i][j] = colors[int(knn_arr[i][j])]
        cv2.addWeighted(overlay_arr.astype(np.uint8), 0.1, img_arr, 0.9, 0, img_arr)
        self.vis_func(img_arr)

    def vis_func(self, img_arr, prefix=""):
      if(self.option == 0):
        now = str(datetime.now().time()).replace(":", "_").replace('.', "_")
        cv2.imwrite(os.path.join(self.parent_dir, prefix+"_"+now+".jpg"), img_arr)
      if(self.option == 1):
        cv2_imshow(img_arr)