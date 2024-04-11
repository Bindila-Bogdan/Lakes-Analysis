import os
import sys
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, cohen_kappa_score


EPSILON = 0.00000001
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


def compute_scores_per_fold_th(data, data_indices, threshold):
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


def compute_cross_validation_scores_th(data, data_indices, threshold, test=False):
    f1_score_cum = 0
    kappa_score_cum = 0
    number_of_images_cum = 0

    # compute metrics for each fold
    for test_data_indices_subset in data_indices:
        train_data_indices_subset = [
            i for i in range(len(data)) if i not in test_data_indices_subset
        ]

        if test:
            f1, kappa, number_of_images = compute_scores_per_fold_th(
                data, test_data_indices_subset, threshold
            )
        else:
            f1, kappa, number_of_images = compute_scores_per_fold_th(
                data, train_data_indices_subset, threshold
            )
        f1_score_cum += f1
        kappa_score_cum += kappa
        number_of_images_cum += number_of_images

    # compute the average metric for each threshold
    return f1_score_cum / number_of_images_cum, kappa_score_cum / number_of_images_cum


def train_test_rf(data, data_indices, rf_classifier=None):
    # get imagees from fold
    data_fold = [list(data.values())[i] for i in data_indices]
    fold_points = None

    # iterate over images
    for i, _ in enumerate(data_fold):
        # sample water and non-water points
        example = np.array(data_fold[i])
        reshaped_example = example.reshape(example.shape[0], example.shape[1] * example.shape[2])

        water_indices = np.where(reshaped_example[1, :] == 2)[0]
        np.random.shuffle(water_indices)
        water_points = reshaped_example[:, water_indices]

        non_water_indices = np.where(reshaped_example[1, :] == 1)[0]
        np.random.shuffle(non_water_indices)
        non_water_points = reshaped_example[:, non_water_indices]

        # add the points to the previous ones
        current_fold_points = np.concatenate((water_points, non_water_points), axis=1)

        if fold_points is None:
            fold_points = current_fold_points
        else:
            fold_points = np.concatenate((fold_points, current_fold_points), axis=1)

        # create data set
        fold_points_df = pd.DataFrame(fold_points.T)
        y = fold_points_df.iloc[:, 1].astype(int).values
        x = fold_points_df.drop(1, axis=1).to_numpy()

        # train model
        if rf_classifier is None:
            rf_classifier = RandomForestClassifier(n_jobs=os.cpu_count())
            rf_classifier.fit(x, y)

        # predict the values and compute the metrics
        y_pred = rf_classifier.predict(x)
        f1 = f1_score(y, y_pred, average="weighted") * len(data_fold)
        kappa = cohen_kappa_score(y, y_pred) * len(data_fold)

        return rf_classifier, f1, kappa


def cross_validation_rf(data, data_indices, rf_classifier_input=None):
    f1_score_cum = 0
    kappa_cum = 0
    f1_score_cum_val = 0
    kappa_cum_val = 0
    num_images = 0.000001
    num_images_val = 0.00000001

    # compute metrics for each fold
    for test_data_indices_subset in tqdm(data_indices):
        train_data_indices_subset = [
            i for i in range(len(data)) if i not in test_data_indices_subset
        ]
        # account for the case when we want to train with the entire dataset
        do_not_test = False

        if len(train_data_indices_subset) == 0:
            do_not_test = True
            train_data_indices_subset = data_indices[0]

        # train or test the Random Forest model
        rf_classifier, f1, kappa = train_test_rf(data, train_data_indices_subset, rf_classifier_input)
        f1_score_cum += f1
        kappa_cum += kappa
        num_images += len(train_data_indices_subset)

        # account for the case when we want to train with the entire dataset
        if not do_not_test:
            _, f1_val, kappa_val = train_test_rf(data, test_data_indices_subset, rf_classifier)
            f1_score_cum_val += f1_val
            kappa_cum_val += kappa_val
            num_images_val += len(test_data_indices_subset)

    # compute the average metrics
    return f1_score_cum / num_images, kappa_cum / num_images, f1_score_cum_val / num_images_val, kappa_cum_val / num_images_val, rf_classifier


def predict_th(data, threshold):
    # create a list that stores the detections obtained with the best threshold
    detections = []

    for date in data.keys():
        detections.append(data[date][0] > threshold)

    return detections


def predict_rf(data, rf_classifier):
    # get imagees from fold
    data_fold = list(data.values())
    predictions = []

    # iterate over images
    for i, _ in enumerate(tqdm(data_fold)):
        # sample water and non-water points
        example = np.array(data_fold[i])
        height = example.shape[1]
        width = example.shape[2]
        reshaped_example = example.reshape(example.shape[0], height * width)

        # create data set
        x = pd.DataFrame(reshaped_example.T).drop(1, axis=1).to_numpy()
        y_pred = rf_classifier.predict(x)

        predictions.append(y_pred.reshape(height, width) - 1)

    return predictions