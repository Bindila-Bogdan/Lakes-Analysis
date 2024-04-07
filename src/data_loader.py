import os
import rasterio


DATA_PATH = "./data/{lake_name}/"
BLUE_BAND = 1
GREEN_BAND = 2
RED_BAND = 3
WATER_INDEX_BAND = 20
GROUND_TRUTH_BAND = 21


def load_lakes(lake_names):
    # create a sorted dictionary by date that stores for each date of the specified lakes all bands
    names_dates = []
    bands = []

    for lake_name in lake_names:
        # read data for current lake
        dates = [date for date in os.listdir(DATA_PATH.format(lake_name=lake_name))]
        maps = [
            rasterio.open(DATA_PATH.format(lake_name=lake_name) + date)
            for date in dates
        ]

        names_dates_current_lake = [lake_name + "_" + date[:-4] for date in dates]

        # select bands of interest
        bands_current_lake = [
            [
                map.read(band_index)
                for band_index in [
                    WATER_INDEX_BAND,
                    GROUND_TRUTH_BAND,
                    RED_BAND,
                    GREEN_BAND,
                    BLUE_BAND,
                ]
            ]
            for map in maps
        ]

        # store the data for the current lake
        names_dates.extend(names_dates_current_lake)
        bands.extend(bands_current_lake)

    # build the dictioanry and return it
    data = {
        k: v
        for k, v in sorted(dict(zip(names_dates, bands)).items(), key=lambda x: x[0])
    }

    # split data indices into folds, so that in each subset are indices of 2 lakes
    data_indices = [
        [
            index
            for index, name_date in enumerate(list(data.keys()))
            if lake_name not in name_date
        ]
        for lake_name in lake_names
    ]

    return data, data_indices
