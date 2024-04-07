import sys
import copy
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score


MISSING_DATA_THRESHOLD = 0.5


def get_water_index_values(data):
    # get the global min and max values of the water index across the dataset
    min_value = sys.maxsize
    max_value = -sys.maxsize

    water_index_values = []

    for bands in data.values():
        min_value = min(min_value, bands[0].flatten().min())
        max_value = max(max_value, bands[0].flatten().max())

        # store all water index values
        water_index_values.extend(list(bands[0].flatten()))

    return min_value, max_value, water_index_values


def preprocess_ground_truth(ground_truth):
    ground_truth_copy = copy.deepcopy(ground_truth)

    # replace no data 0 with 3
    ground_truth_copy[ground_truth_copy == 0] = 3

    # replace water detection 2 with 0 for water
    ground_truth_copy[ground_truth_copy == 2] = 0

    # represent water with 1 and no water with 0
    ground_truth_copy = 1 - ground_truth_copy

    return ground_truth_copy


def compute_scores_per_fold(data, data_indices, threshold):
    # initialize values
    f1 = 0
    kappa = 0
    number_of_images = 0

    # iterate over data that have an index from the passed list
    for index in data_indices:
        water_detection = (list(data.values())[index][0] > threshold).flatten()
        ground_truth = list(data.values())[index][1]

        # consider the ground truth only if it has less than 50% missing data
        if (
            sum(ground_truth.flatten() == 0) / len(ground_truth.flatten())
            <= MISSING_DATA_THRESHOLD
        ):
            preprocessed_ground_truth = preprocess_ground_truth(ground_truth).flatten()

            # get the indices of pixels with data
            empty_indices = np.where(preprocessed_ground_truth == -2)[0]

            # if the ground truth contains any pixels with info compute the f1 score
            if len(empty_indices) > 0:
                number_of_images += 1
                f1 += f1_score(
                    np.delete(preprocessed_ground_truth, empty_indices),
                    np.delete(water_detection, empty_indices),
                    average="weighted",
                )
                kappa += cohen_kappa_score(
                    np.delete(preprocessed_ground_truth, empty_indices),
                    np.delete(water_detection, empty_indices),
                )

    return f1, kappa, number_of_images


def compute_cross_validation_scores(data, data_indices, threshold):
    f1_score_cum = 0
    kappa_score_cum = 0
    number_of_images_cum = 0

    # compute metrics for each fold
    for data_indices_subset in data_indices:
        f1, kappa, number_of_images = compute_scores_per_fold(
            data, data_indices_subset, threshold
        )
        f1_score_cum += f1
        kappa_score_cum += kappa
        number_of_images_cum += number_of_images

    # compute the average metric for each threshold
    return f1_score_cum / number_of_images_cum, kappa_score_cum / number_of_images_cum


def predict(data, threshold):
    # create a list that stores the detections obtained with the best threshold
    detections = []

    for date in data.keys():
        detections.append(data[date][0] > threshold)

    return detections
