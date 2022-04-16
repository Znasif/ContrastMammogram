import cv2
import numpy as np
import random

class MammViz():
    def __init__(self, vis_func):
        self.vis_func = vis_func
        
    def visualize_landmarks(self, pseudo, dim):
        img_arr = np.zeros(dim)
        for i in pseudo:
            cv2.circle(img_arr, i, 4, 255, 1)
        self.vis_func(img_arr)
    
    def visualize_regions(self, img_arr, pseudo, knn_arr):
        colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for i in range(len(pseudo)+1)]
        dim = knn_arr.shape
        overlay_arr = np.zeros_like(img_arr)
        for i in range(dim[0]):
            for j in range(dim[1]):
                overlay_arr[i][j] = colors[int(knn_arr[i][j])]
        cv2.addWeighted(overlay_arr.astype(np.uint8), 0.5, img_arr, 0.5, 0, img_arr)
        self.vis_func(img_arr)
