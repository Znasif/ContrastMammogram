import cv2
import numpy as np
from numpy.core.numeric import count_nonzero
from scipy import ndimage

def pad_percent(img_arr, side, perc):
  img = img_arr.copy()
  if len(img_arr.shape) == 3:
    black = np.zeros(3)
  else:
    black = 0
  size_x, size_y = img_arr.shape[:2]
  size_x, size_y = int(size_x/100*perc), int(size_y/100*perc)
  img[0:size_x, ::] = black
  img[img_arr.shape[0]-size_y:img_arr.shape[0]] = black
  if side == "LEFT":
    img[::, img_arr.shape[1]-size_y:img_arr.shape[1]] = black
  if side == "RIGHT":
    img[::, 0:size_y] = black
  return img

def resize(img_arr, desired_size):
    '''
    resizes and pads the image, without distorting the original image
    '''
    old_size = img_arr.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    img_arr = cv2.resize(img_arr, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    if(len(img_arr.shape)==3):
        color = [0, 0, 0]
    else:
        color = 0
    return cv2.copyMakeBorder(img_arr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def line_to_img(st, en, dim):
    '''
    draw a line from st point to en point on an image of shape dim
    '''
    img_arr = np.zeros(dim)
    cv2.line(img_arr, st, en, 255, thickness=1)
    return img_arr

def cnt_to_img(cnt, dim):
    '''
    draw contour (minimal thickness) on an image of shape dim
    '''
    img_arr = np.zeros(dim)
    cv2.drawContours(img_arr, [cnt], -1, 255, 1)
    return img_arr

def contour_intersections(cnt, st, en, n, dim, level):
    '''
    with a given breast contour, draw a line from st to en along the contour intersection.
    Further subdivide the contour interior in n parallal lines. levels identify how many recursive
    subdivisions to perform. We only do two.
    '''
    cnt_arr = cnt_to_img(cnt, dim)
    line = line_to_img(st, en, dim)
    points = np.transpose(np.nonzero(cnt_arr*line))
    
    ext = points[0]
    opp = points[-1]
    inter_points = [(ext[1], ext[0])]
    next_points = [inter_points[0]]

    for i in range(1, n-1):
        inter_points.append((int(ext[1]+i*(opp[1]-ext[1])/(n-1)), int(ext[0]+i*(opp[0]-ext[0])/(n-1))))
        ext_ = (inter_points[i][0], 0)
        opp_ = (inter_points[i][0], dim[0]-1)
        if(level > 1):
            tmp = contour_intersections(cnt, ext_, opp_, n, dim, level-1)
            for j in tmp:
                next_points.append(j)
    inter_points.append((opp[1], opp[0]))

    if level > 1:
        return next_points
    return inter_points

def closest_node(nodes, point):
    '''
    given a pixel coordinate, identify the closest node
    '''
    closest_node = 0
    d = (nodes[0][0] - point[1])**2 + (nodes[0][1] - point[0])**2
    for i in range(len(nodes)):
        d_ = (nodes[i][0] - point[1])**2 + (nodes[i][1]-point[0])**2
        if(d_<d):
            d = d_
            closest_node = i
    return closest_node

def pseudo_to_knn(cnt, pseudo, dim):
    '''
    subdivision of the contour interior with k=1 knn
    '''
    img_arr = np.zeros((dim, dim))
    cv2.drawContours(img_arr, [cnt], -1, 1, -1)
    points = np.transpose(np.nonzero(img_arr))
    knn_arr = np.zeros((dim, dim))
    for i in points:
        knn_arr[tuple(i)] = closest_node(pseudo, i)+1
    return knn_arr.astype(np.uint8)

def pixel_belongs_to(img_path, view, breast, points_along_line, dim):
    img_arr = cv2.imread(img_path)
    img_arr = pad_percent(img_arr, breast, 3)
    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.GaussianBlur(img_gray, (65, 65), 0)
    _, thresh = cv2.threshold(kernel, 10, 255, cv2.THRESH_BINARY)
    if view == "MLO":
        if breast == "LEFT":
            thresh = ndimage.rotate(thresh, 5)
        else:
            thresh = ndimage.rotate(thresh, -30)
    
    thresh = resize(thresh, dim)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntSorted = sorted(contours, key=lambda x:cv2.contourArea(x))[-1]
    
    if breast == "RIGHT":
        nipple_point = tuple(cntSorted[cntSorted[:, :, 0].argmin()][0])
        opposite_point = (img_arr.shape[1]-1, nipple_point[1])
    else:
        nipple_point = tuple(cntSorted[cntSorted[:, :, 0].argmax()][0])
        opposite_point = (0, nipple_point[1])
    
    if view == "MLO":
        opposite_point = (opposite_point[0], opposite_point[1]-dim//2)
    
    pseudo_landmark_points = contour_intersections(cntSorted, nipple_point, opposite_point, points_along_line, dim, 2)
    pseudo_landmark_points.append(nipple_point)
    knn_arr = pseudo_to_knn(cntSorted, pseudo_landmark_points, dim)

    if view == "MLO":
        if breast == "LEFT":
            knn_arr = ndimage.rotate(knn_arr, 5)
        else:
            knn_arr = ndimage.rotate(knn_arr, -30)       
    return knn_arr

def gen_bin_mask(poly, height, width, dim):
    bin_arr = np.zeros((height, width))
    cv2.drawContours(bin_arr, np.array(poly)-0.5, 0, 1, -1)
    return resize(bin_arr, dim)