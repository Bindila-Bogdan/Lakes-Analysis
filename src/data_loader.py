import os
import rasterio
from tqdm import tqdm


DATA_PATH = "./data/{lake_name}/"


def load_lakes(lake_names):
    # create a sorted dictionary by date that stores for each date of the specified lakes all bands
    names_dates = []
    bands = []

    for lake_name in tqdm(lake_names):
        # read data for current lake
        dates = [date for date in os.listdir(DATA_PATH.format(lake_name=lake_name))]
        maps = [
            rasterio.open(DATA_PATH.format(lake_name=lake_name) + date)
            for date in dates
        ]

        names_dates_current_lake = [lake_name + "_" + date[:-4] for date in dates]

        # select bands of interest
        bands_current_lake = []

        for current_map in maps:
            number_of_bands = current_map.count

            if number_of_bands == 8:
                bands_current_lake.append(
                    [
                        current_map.read(band_index)
                        for band_index in [7, 8, 1, 2, 3, 4, 5, 6]
                    ]
                )
            elif number_of_bands == 9:
                bands_current_lake.append(
                    [
                        current_map.read(band_index)
                        for band_index in [8, 9, 2, 3, 4, 5, 6, 7]
                    ]
                )
            else:
                raise ValueError(f"Unrecognized number of bands: {number_of_bands}")

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
