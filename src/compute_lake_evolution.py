import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


FIRST_DATE = "1984-03"
LAST_DATE = "2012-04"
DILATION_SIZE = 3
SPATIAL_RESOLUTION = 30


def estimate_area_by_month(datections, lake_name, all_dates):
    dates = [date[-10:] for date in all_dates if lake_name in date]
    indices = [index for index, date in enumerate(all_dates) if lake_name in date]
    lake_detections = np.array(datections)[indices]

    # create a data frame that stores the lake area in square kilometers for every month with observations

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
    fig.show()


def display_image(
    index,
    dates,
    water_indices,
    ground_truths,
    detections,
    dilated_detections,
    isolated_detections,
    filled_isolated_detections,
):

    fig, ax = plt.subplots(1, 6)
    fig.set_size_inches(24, 6)

    ax[0].imshow(water_indices[index])
    ax[0].set_title("Water index")
    ax[0].axis("off")

    ax[1].imshow(ground_truths[index])
    ax[1].set_title("Ground truth")
    ax[1].axis("off")

    ax[2].imshow(detections[index])
    ax[2].set_title("Detection")
    ax[2].axis("off")

    ax[3].imshow(dilated_detections[index])
    ax[3].set_title("Dilated detection")
    ax[3].axis("off")

    ax[4].imshow(isolated_detections[index])
    ax[4].set_title("Isolated detection")
    ax[4].axis("off")

    ax[5].imshow(filled_isolated_detections[index])
    ax[5].set_title("Filled isolated detection")
    ax[5].axis("off")

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

    for i in range(len(filled_isolated_detections)):
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

        terrain_image = np.dstack((lake_data[i][2], lake_data[i][3], lake_data[i][4]))

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
