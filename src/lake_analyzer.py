import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ipywidgets import interact, widgets, fixed

from .lake_post_processor import *

FIRST_DATE = "1984-03"
LAST_DATE = "2021-12"
DILATION_SIZE = 3
SPATIAL_RESOLUTION = 30
sns.set()


def estimate_area_by_month(datections, lake_name, all_dates):
    dates = [date[-10:] for date in all_dates if lake_name in date]
    indices = [index for index, date in enumerate(all_dates) if lake_name in date]
    lake_detections = np.array(datections)[indices]

    # create a data frame that stores the lake area in square kilometers for every month
    areas = []

    # pixel area in square kilometers
    pixel_area = SPATIAL_RESOLUTION**2 / (1000**2)

    for detection in lake_detections:
        areas.append(detection.sum() * pixel_area)
    known_area_by_month = pd.DataFrame({"date": dates, "area": areas})

    # create a data frame that contains the months with missing observations
    missing_dates = []
    ground_truth_months_trimmed = [date[:-3] for date in dates]

    for year in range(int(FIRST_DATE[:4]), int(LAST_DATE[:4]) + 1):
        for month in range(1, 13):
            date = f"{year}-{str(month).zfill(2)}"

            if (
                FIRST_DATE <= date <= LAST_DATE
            ) and date not in ground_truth_months_trimmed:
                missing_dates.append(date)

    missing_area_by_month = pd.DataFrame(
        {"date": missing_dates, "area": [np.nan] * len(missing_dates)}
    )

    # store in one data frame the lake area estimation for each month
    area_by_month = pd.concat([known_area_by_month, missing_area_by_month]).sort_values(
        "date"
    )

    return area_by_month


def plot_lake_area_by_month(detections, data, lake_name):
    # get estimated lake area by month
    area_by_month = estimate_area_by_month(detections, lake_name, list(data.keys()))

    # display lake area evolution
    fig = px.line(
        area_by_month[~area_by_month["area"].isna()],
        x="date",
        y="area",
        markers=True,
        labels={"area": "area in square kilometers"},
        title=f"Observations of area evolution for lake {lake_name.capitalize()}",
    )
    fig.update_traces(line_color='#06c2ac')
    fig.show()


def display_detection(
    index,
    dates,
    water_indices,
    ground_truths,
    detections,
    dilated_detections,
    largest_area,
    isolated_detections,
    filled_isolated_detections,
):
    # display several versions of a detection
    fig, ax = plt.subplots(1, 7)
    fig.set_size_inches(24, 6)

    images = [
        water_indices[index],
        ground_truths[index],
        detections[index],
        dilated_detections[index],
        largest_area,
        isolated_detections[index],
        filled_isolated_detections[index],
    ]
    titles = [
        "Water index",
        "Ground truth",
        "Detection",
        "Dilated detection",
        "Detection mask",
        "Isolated detection",
        "Filled isolated detection",
    ]

    for i, _ in enumerate(images):
        ax[i].imshow(images[i], cmap="Greys")
        ax[i].set_title(titles[i])
        ax[i].axis("off")
        ax[i]

    fig.suptitle(f"Year-month: {dates[index]}")

    plt.show()
    plt.close()


def prepare_visualization(
    data, all_dates, lake_name, filled_isolated_detections, largest_area_mask_overall
):
    indices = [index for index, date in enumerate(all_dates) if lake_name in date]
    lake_data = np.array(list(data.values()))[indices]

    # prepare detections and terrain images to be plotted
    terrain_images = []
    explained_detections = []

    for i, _ in enumerate(filled_isolated_detections):
        detection = filled_isolated_detections[i].flatten()
        ground_truth = lake_data[i][1].flatten()

        # false positives
        false_positives = np.intersect1d(
            np.intersect1d(np.where(detection == 1), np.where(ground_truth == 1)),
            np.where(largest_area_mask_overall.flatten() == 1),
        )

        # false negative
        false_negatives = np.intersect1d(
            np.intersect1d(np.where(detection == 0), np.where(ground_truth == 2)),
            np.where(largest_area_mask_overall.flatten() == 1),
        )

        # add correct detections, false positives and false negatives
        explained_detection = detection
        explained_detection[false_positives] = 2
        explained_detection[false_negatives] = 3

        # store the expalined detection
        explained_detections.append(
            explained_detection.reshape(largest_area_mask_overall.shape)
        )

        # create the terrain image by adding the RGB bands
        terrain_image = np.dstack((lake_data[i][4], lake_data[i][3], lake_data[i][2]))

        # scale the bands to have values between 0 and 1
        for channel_index in range(3):
            channel_data = terrain_image[:, :, channel_index]
            channel_data_scaled = (channel_data - channel_data.min()) / (
                channel_data.max() - channel_data.min()
            )
            terrain_image[:, :, channel_index] = channel_data_scaled
        terrain_images.append(terrain_image)
    return explained_detections, terrain_images


def display_predictions(
    index, lake_name, current_dates, terrain_images, explained_detections
):
    # display the lake  detection on top of the original satellite image
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10, 8)

    date = current_dates[index]

    ax[0].imshow(terrain_images[index])
    ax[0].axis("off")

    ax[1].imshow(terrain_images[index])
    ax_1 = ax[1].imshow(
        explained_detections[index],
        alpha=1.0 * (explained_detections[index] > 0),
        cmap=ListedColormap(["royalblue", "turquoise", "orangered"]),
    )
    cbar = plt.colorbar(
        ax_1, ticks=[1, 2, 3], orientation="vertical", fraction=0.05, pad=0.04
    )
    cbar.ax.set_yticklabels(["False negatives", "Water", "False positives"])
    ax[1].axis("off")

    fig.suptitle(f"Prediction for lake {lake_name.capitalize()} {date}")
    plt.tight_layout()

    plt.show()
    plt.close()


def analyze_lake(lake_name, data, detections):
    # put together all the plotting functions for a selected lake
    current_dates = [date[-10:] for date in list(data.keys()) if lake_name in date]
    indices = [
        index for index, date in enumerate(list(data.keys())) if lake_name in date
    ]
    lake_detections = np.array(detections)[indices]
    ground_truths = [bands[1] for bands in np.array(list(data.values()))[indices]]
    water_indices = [bands[0] for bands in np.array(list(data.values()))[indices]]

    # get region of interest
    largest_area_mask_overall, _, dilated_detections = get_largest_lake_mask(
        lake_detections
    )

    # get only detections that intersect the region of interest
    isolated_detections, filled_isolated_detections = remove_adjacent_water_bodies(
        lake_detections, largest_area_mask_overall
    )

    # plot area evolution by month
    plot_lake_area_by_month(detections, data, lake_name)

    # display interactive plot with several versions of a detection
    interact(
        display_detection,
        index=widgets.IntSlider(min=0, max=len(indices) - 1, step=1, value=0),
        dates=fixed(current_dates),
        ground_truths=fixed(ground_truths),
        water_indices=fixed(water_indices),
        detections=fixed(lake_detections),
        dilated_detections=fixed(dilated_detections),
        largest_area=fixed(largest_area_mask_overall),
        isolated_detections=fixed(isolated_detections),
        filled_isolated_detections=fixed(filled_isolated_detections),
    )

    # prepare data for the plot that shows the final detections
    explained_detections, terrain_images = prepare_visualization(
        data,
        list(data.keys()),
        lake_name,
        filled_isolated_detections,
        largest_area_mask_overall,
    )

    # display interactive plot with the final detections
    interact(
        display_predictions,
        index=widgets.IntSlider(min=0, max=len(current_dates) - 1, step=1, value=0),
        lake_name=fixed(lake_name),
        current_dates=fixed(current_dates),
        terrain_images=fixed(terrain_images),
        explained_detections=fixed(explained_detections),
    )


def plot_feature_importance(trained_rf_classifier):
    # get and sort feature importances
    importances = trained_rf_classifier.feature_importances_
    feature_names = [
        "water index",
        "blue band",
        "green band",
        "red band",
        "near infrared band",
        "shortwave infrared 1 band",
        "shortwave infrared 2 band",
    ]
    feature_importance = {
        k: v
        for k, v in sorted(
            dict(zip(feature_names, importances)).items(),
            reverse=True,
            key=lambda x: x[1],
        )
    }

    # plot a bar plot with feature importances given by the Random Forest
    sns.barplot(
        y=feature_importance.keys(), x=feature_importance.values(), color="turquoise"
    )
    plt.ylabel("feature")
    plt.xlabel("importance")
    plt.title("Random Forest feature importance")
    plt.show()
