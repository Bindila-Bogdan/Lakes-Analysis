import sys
import cv2
import numpy as np
from tqdm import tqdm

from scipy.ndimage import binary_dilation, binary_closing

DILATION_SIZE = 3


def get_largest_lake_mask(lake_detections):
    # find the mask of the lake when it has the largest area
    largest_area_overall = -sys.maxsize
    largest_area_mask_overall = None
    dilated_detections = []

    # iterate over each observation of the lake
    for i, _ in enumerate(lake_detections):
        largest_area = -sys.maxsize
        largest_area_mask = None

        # dilate the image and store it
        diltated_img = binary_dilation(
            lake_detections[i],
            structure=np.ones((DILATION_SIZE, DILATION_SIZE), np.bool_),
        )
        diltated_img_cast = diltated_img.astype(np.uint8) * 255
        dilated_detections.append(diltated_img)

        # get the connected components
        ret, labels = cv2.connectedComponents(diltated_img_cast)

        # find the biggest connected component from the current observation
        for label in range(1, ret):
            mask = np.zeros(labels.shape, dtype=np.uint8)
            mask[labels == label] = 1
            mask_area = mask.sum()

            if mask_area > largest_area:
                largest_area = mask_area
                largest_area_mask = mask

        # if the lake has an area larger than the previous largest area, update the mask
        if largest_area > largest_area_overall:
            largest_area_overall = largest_area
            largest_area_mask_overall = largest_area_mask

    return largest_area_mask_overall, largest_area_overall, dilated_detections


def remove_adjacent_water_bodies(lake_detections, largest_area_mask_overall):
    # remove all water detections that do not intersect with the largest found mask
    isolated_detections = []
    filled_isolated_detections = []

    # iterate over each observation
    for i in tqdm(range(len(lake_detections))):
        # prepare the data
        detection = lake_detections[i]
        detection_cast = detection.astype(np.uint8) * 255

        # get the connected components
        ret, labels = cv2.connectedComponents(detection_cast)

        # initiate the mask
        mask = np.array(labels, dtype=np.uint32)

        # remove all connected components that do not intersect with the largest found mask
        for label in range(1, ret):
            if (
                ((mask == label).astype(np.uint8) + largest_area_mask_overall) == 2
            ).sum() > 0:
                mask[labels == label] = ret
            else:
                mask[labels == label] = 0

        # store these isolated detections into an array
        mask = (mask / ret).astype(np.uint8)
        isolated_detections.append(mask)

        # fill the holes in the image by applying a morphological closing operation
        filled_mask = binary_closing(mask).astype(np.uint8)
        filled_isolated_detections.append(filled_mask)

    return isolated_detections, filled_isolated_detections
