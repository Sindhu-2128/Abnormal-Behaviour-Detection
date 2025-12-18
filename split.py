# split.py
import numpy as np

class Spliter(object):
    """
    A class to intelligently split large connected components (blobs) into
    smaller, person-sized bounding boxes. This is a high-performance version
    that avoids slow operations in loops.
    """
    normal = 120
    heightNorm = 20
    widthNorm = 6

    def __init__(self, discardFloor=0.5, splitCeil=2.0):
        self.floor = discardFloor * Spliter.normal
        self.ceil = splitCeil * Spliter.normal

    def split(self, pos, fg_img, weight):
        posArea, heights, widths = self.areaHeightWidthCompute(pos, weight)
        
        # Use a fast Python list for building results
        realPos_list = []

        for ind, area in enumerate(posArea):
            height = heights[ind]
            width = widths[ind]

            if area < self.floor:
                continue

            if area > self.ceil:
                n_h = int(round(height[0] / Spliter.heightNorm))
                if n_h == 0: n_h = 1

                n_w = int(round(width[0] / Spliter.widthNorm))
                if n_w == 0: n_w = 1
                
                n = min(int(round(area[0] / Spliter.normal)), n_w * n_h)
                if n == 0: n = 1

                box_height = pos[ind][0] - pos[ind][1]
                box_width = pos[ind][2] - pos[ind][3]
                if box_height <= 0 or box_width <= 0:
                    continue

                step_y = box_height / n_h
                step_x = box_width / n_w

                res = []
                for i in range(n_h):
                    pos1 = int(pos[ind][1] + i * step_y)
                    pos0 = int(pos[ind][1] + (i + 1) * step_y)
                    for j in range(n_w):
                        pos3 = int(pos[ind][3] + j * step_x)
                        pos2 = int(pos[ind][3] + (j + 1) * step_x)
                        
                        roi = fg_img[pos1:pos0, pos3:pos2]
                        if roi.size == 0: continue
                        res.append([pos0, pos1, pos2, pos3, roi.sum(), roi.mean()])
                
                if not res: continue

                res.sort(key=lambda x: x[-1], reverse=True)
                
                num_to_keep = min(n, len(res))
                for i in range(num_to_keep):
                    # Slice to get the 5 elements: 4 coords + area
                    new_box = res[i][:-1]
                    realPos_list.append(new_box)
            else:
                realPos_list.append(pos[ind].tolist())
        
        # Convert to a NumPy array once at the end
        if not realPos_list:
            return np.zeros((0, 5))
        return np.array(realPos_list)

    def areaHeightWidthCompute(self, pos, weight):
        area = np.zeros((pos.shape[0], 1))
        height = np.zeros((pos.shape[0], 1))
        width = np.zeros((pos.shape[0], 1))
        
        max_weight_index = len(weight) - 1

        for ind, eachPos in enumerate(pos):
            y_index = int((eachPos[0] + eachPos[1]) // 2)
            y_index = max(0, min(y_index, max_weight_index))
            
            w = weight[y_index]
            if w < 0: w = 0 # Ensure weight is not negative
            area[ind] = eachPos[-1] * w
            height[ind] = (eachPos[0] - eachPos[1]) * np.sqrt(w)
            width[ind] = (eachPos[2] - eachPos[3]) * np.sqrt(w)
            
        return np.nan_to_num(area), np.nan_to_num(height), np.nan_to_num(width)