import re
import numpy as np
import os
from xmlLoader_generator import Poi_handle

class Weight_matrix:

    def __init__(self, ref_data_path=None, frame_height=None, n=3):
        """
        Initializes the Weight Matrix.
        Args:
            ref_data_path (str): The absolute path to the 'ref_data' directory.
            frame_height (int): The height of the video frames. THIS IS REQUIRED.
            n (int): The picture number to use for calibration from poi.xml.
        """
        try:
            if ref_data_path is None:
                raise ValueError("A path to the 'ref_data' directory must be provided.")
            if frame_height is None:
                raise ValueError("The frame_height must be provided to initialize the weight matrix.")

            poi_path = os.path.join(ref_data_path, 'poi.xml')
            connect_path = os.path.join(ref_data_path, 'connectedFieldImg.txt')

            if not os.path.exists(connect_path) or not os.path.exists(poi_path):
                 raise FileNotFoundError("poi.xml or connectedFieldImg.txt not found for perspective correction.")

            tps = self.diff(Poi_handle(path=poi_path).searchPic(n))
            connect = np.loadtxt(connect_path, delimiter=',')

            self.y1 = tps[0][0]
            self.y2 = tps[-1][0]
            self.ab = tps[0][1]
            self.cd = tps[-1][1]
            self.h1 = connect[8][0] - connect[8][1]
            self.h2 = connect[15][0] - connect[15][1]

            self.compute_weight_matrix(frame_height)

        except Exception as e:
            print(f"[WARNING] Could not initialize Weight_matrix from data: {e}")
            print(f"[WARNING] Using a default identity weight matrix. Perspective correction will be disabled.")
            self.weight_matrix = np.ones(frame_height)

    def diff(self, pic):
        """Helper function to parse calibration data from the XML."""
        res = []
        if pic is not None:
            for y in pic:
                points = y.findall('point')
                if len(points) < 2: continue
                x1 = int(re.findall(r'\((\d+),', points[0].text)[0])
                x2 = int(re.findall(r'\((\d+),', points[-1].text)[0])
                res.append((int(y.get('val')), abs(x1 - x2)))
        return res

    def y_weight(self, y):
        """Calculates the perspective weight for a given y-coordinate."""
        if self.y1 == self.y2: return 1.0
        w1 = (y - self.y2) / (self.y1 - self.y2)
        w2 = (self.y1 - y) / (self.y1 - self.y2)
        return (w1 * (self.cd / self.ab) + w2) * (w1 * (self.h2 / self.h1) + w2)

    def compute_weight_matrix(self, frame_height):
        """Generates the weight matrix and ensures no values are negative."""
        # Calculate the weights using the formula
        raw_weights = np.vectorize(self.y_weight)(np.arange(frame_height))
        
        # --- THIS IS THE FIX ---
        # Clamp the weights: any value in the array that is less than 0 will be set to 0.
        # This prevents the np.sqrt() function from receiving negative numbers.
        self.weight_matrix = np.maximum(0, raw_weights)
        # --- END FIX ---

    def get_weight_matrix(self):
        """Returns the computed weight matrix."""
        return self.weight_matrix