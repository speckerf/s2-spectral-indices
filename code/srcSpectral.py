import concurrent.futures
import os
import time

import ee
import pandas as pd
from config import get_config
from loguru import logger
from tqdm import tqdm
from utilsCloudfree import apply_cloudScorePlus_mask
from utilsSpectral import apply_spectral_indices, collapse_and_cast_spectral_indices
from utilsTiles import get_epsg_code_from_mgrs, get_s2_indices_filtered

CONFIG = get_config()

service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(
    service_account, "auth/gem-eth-analysis-24fe4261f029.json"
)
ee.Initialize(credentials, project="ee-speckerfelix")
# ee.Initialize()


def export_mgrs_tile(mgrs_tile: str, year: int) -> None:
    version = CONFIG["PIPELINE_PARAMS"]["VERSION"]
    output_resolution = CONFIG["PIPELINE_PARAMS"]["OUTPUT_RESOLUTION"]

    # # check if the export has already been done
    # imgc_folder = (
    #     CONFIG["GEE_FOLDERS"]["ASSET_FOLDER"]
    #     + f"/spectral-indices_s2-srf-yearly_{output_resolution}m_{version}/"
    # )
    # system_index = f"spectral-indices_s2-srf-yearly_m_{output_resolution}m_s_{year}0101_{year}1231_T{mgrs_tile}_epsg-{get_epsg_code_from_mgrs(mgrs_tile)}_{version}"
    # if (
    #     ee.data.getInfo(imgc_folder + system_index) is not None
    #     and ee.data.getInfo(imgc_folder + system_index)["type"] == "Image"
    # ):
    #     logger.info(f"Export already done for mgrs_tile: {mgrs_tile}")
    #     return

    logger.info(f"Exporting mgrs_tile: {mgrs_tile}")
    start_date = ee.Date(f"{year}-01-01")
    end_date = ee.Date(f"{year}-12-31")

    # list all sentinel-2 tiles in this mgrs tile
    all_mgrs_tiles = pd.read_csv(
        os.path.join(
            "data",
            "tiles",
            "mgrs_tiles",
            "mgrs_tiles_all_land_ecoregions.csv",
        )
    )
    current_mgrs_tiles = list(
        set(
            all_mgrs_tiles[all_mgrs_tiles["mgrs_tile_3"] == mgrs_tile][
                "mgrs_tile"
            ].tolist()
        )
    )

    # save s2_indices_filtered for later use
    s2_indices_filename = f"s2-indices_{year}_mgrs-tile-{mgrs_tile}_{version}.txt"
    if os.path.exists(
        os.path.join(
            "data",
            "tiles",
            "s2_indices_per_mgrs_tile",
            s2_indices_filename,
        )
    ):
        logger.debug(f"Loading s2_indices_filtered from file: {s2_indices_filename}")
        with open(
            os.path.join(
                "data",
                "tiles",
                "s2_indices_per_mgrs_tile",
                s2_indices_filename,
            ),
            "r",
        ) as f:
            s2_indices_filtered = f.read().splitlines()
    else:
        s2_indices_filtered = get_s2_indices_filtered(
            mgrs_tiles=current_mgrs_tiles, start_date=start_date, end_date=end_date
        )
        logger.debug(f"Saving s2_indices_filtered to file: {s2_indices_filename}")
        # save s2_indices_filtered for later use
        with open(
            os.path.join(
                "data",
                "tiles",
                "s2_indices_per_mgrs_tile",
                s2_indices_filename,
            ),
            "w",
        ) as f:
            for item in s2_indices_filtered:
                f.write("%s\n" % item)

    if len(s2_indices_filtered) == 0:
        logger.error(
            f"Sentinel-2 collection empty after filter for mgrs_tile: {mgrs_tile}"
        )
        return

    imgc = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filter(
        ee.Filter.inList("system:index", s2_indices_filtered)
    )

    imgc = imgc.map(lambda img: img.divide(10000))

    # determine intersecting output geometry
    output_geometry_bbox = imgc.geometry().bounds()

    # apply cloud mask
    imgc_cloudfree = apply_cloudScorePlus_mask(imgc)

    # predict spectral indices
    preds = apply_spectral_indices(imgc_cloudfree).select(CONFIG["INDICES"])

    # collapse to mean and stddev
    output_image = collapse_and_cast_spectral_indices(preds)

    ocean_mask = ee.ImageCollection("ESA/WorldCover/v200").first().unmask(199).neq(199)
    output_image = output_image.updateMask(ocean_mask)

    # Set export parameters
    year_start_string = str(year) + "0101"
    year_end_string = str(year) + "1231"
    epsg_code = get_epsg_code_from_mgrs(mgrs_tile)
    epsg_code_gee = f"EPSG:{epsg_code}"
    epsg_string = f"epsg-{epsg_code}"

    system_index = f"spectral-indices_s2-srf-yearly_m_{output_resolution}m_s_{year_start_string}_{year_end_string}_T{mgrs_tile}_{epsg_string}_{version}"

    output_image = (
        output_image.set("system:time_start", ee.Date.fromYMD(int(year), 1, 1).millis())
        .set("system:time_end", ee.Date.fromYMD(int(year), 12, 31).millis())
        .set("year", ee.String(year))
        .set("version", ee.String(version))
        .set("system:index", system_index)
        .set("mgrs_tile", ee.String(mgrs_tile))
        .set("int8_scale", ee.Number(CONFIG["PIPELINE_PARAMS"]["INT8_SCALE"]))
    )

    # Define bit positions from CONFIG
    bit_position_valid_s2_input = CONFIG["GAPFILLING"]["BITMASK_VALID_INPUT"]  # 0
    bit_position_valid_vi = CONFIG["GAPFILLING"]["BITMASK_VALID_VI"]  # 1

    # Create bitmask for valid Sentinel-2 input
    mask_valid_input = (imgc.mosaic().select("B8").mask().gt(0)).multiply(
        2**bit_position_valid_s2_input
    )

    # Create bitmask for valid VI (NDVI)
    mask_valid_vi = (output_image.select(CONFIG["INDICES"][0]).mask().gt(0)).multiply(
        2**bit_position_valid_vi
    )

    # Combine the bitmasks using bitwise OR (addition works for non-overlapping bits)
    bit_mask = mask_valid_input.add(mask_valid_vi).rename("QA_GAPFILL").toUint8()

    # Add the QA bitmask as a new band
    output_image = output_image.addBands(bit_mask)

    imgc_folder = (
        CONFIG["GEE_FOLDERS"]["ASSET_FOLDER"]
        + f"/spectral-indices_s2-srf-yearly_{output_resolution}m_{version}/"
    )

    task = ee.batch.Export.image.toAsset(
        image=output_image,
        description=system_index,
        crs=epsg_code_gee,
        assetId=imgc_folder + system_index,
        region=output_geometry_bbox,
        scale=output_resolution,
        maxPixels=1e11,
    )
    task.start()
    time.sleep(0.1)


def subset_mgrs_years_tiles(mgrs_tiles: list, years: list, max_tasks: int):
    total_tasks = len(mgrs_tiles) * len(years)
    if total_tasks <= max_tasks:
        return mgrs_tiles, years

    logger.debug(
        f"Check number of total exports that are still to be executed: {len(mgrs_tiles) * len(years)}"
    )
    output_resolution = CONFIG["PIPELINE_PARAMS"]["OUTPUT_RESOLUTION"]
    version = CONFIG["PIPELINE_PARAMS"]["VERSION"]

    # check if the export has already been done
    imgc_folder = (
        CONFIG["GEE_FOLDERS"]["ASSET_FOLDER"]
        + f"/spectral-indices_s2-srf-yearly_{output_resolution}m_{version}"
    )
    system_indices_all = [
        f"spectral-indices_s2-srf-yearly_m_{output_resolution}m_s_{year}0101_{year}1231_T{mgrs_tile}_epsg-{get_epsg_code_from_mgrs(mgrs_tile)}_{version}"
        for mgrs_tile in mgrs_tiles
        for year in years
    ]
    system_indices_done = (
        ee.ImageCollection(imgc_folder).aggregate_array("system:index").getInfo()
    )

    system_indices_todo = list(set(system_indices_all) - set(system_indices_done))

    if len(system_indices_todo) <= max_tasks:
        logger.debug(f"Number of tasks to be executed: {len(system_indices_todo)}")
        mgrs_tiles_todo = list(
            set(
                [
                    system_index.split("_")[-3][1:5]
                    for system_index in system_indices_todo
                ]
            )
        )
        return mgrs_tiles_todo, years

    logger.debug(f"Not export all mgrs_tiles and years, only a subset.")
    # ensure mgrs_tiles_subset * years_subset <= max_tasks
    mgrs_tiles_subset = list(
        set([system_index.split("_")[-3][1:5] for system_index in system_indices_todo])
    )
    while len(mgrs_tiles_subset) * len(years) > max_tasks:
        mgrs_tiles_subset.pop()

    return mgrs_tiles_subset, years


def global_export_mgrs_tiles(years: list):
    mgrs_tiles = pd.read_csv(
        os.path.join(
            "data",
            "tiles",
            "mgrs_tiles",
            "mgrs_tiles_all_land_ecoregions.csv",
        )
    )
    mgrs_tiles_list = list(set(mgrs_tiles["mgrs_tile_3"].tolist()))
    include = ["19F", "19E", "20F"]
    mgrs_tiles_list = list(set([*mgrs_tiles_list, *include]))

    logger.debug(f"Exporting mgrs_tiles: {mgrs_tiles_list} for years {years}")

    if len((mgrs_tiles_list) * len(years)) > 3000:
        logger.warning(
            f"Too many tasks to start: {len((mgrs_tiles_list) * len(years))} > 3000. Only exporting a subset."
        )
        mgrs_tiles_list, years = subset_mgrs_years_tiles(
            mgrs_tiles_list, years, max_tasks=2900
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(export_mgrs_tile, mgrs_tile, year)
            for mgrs_tile in mgrs_tiles_list
            for year in years
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            try:
                future.result()  # If the task raised an exception, this will raise it here
            except Exception as e:
                logger.error(f"Error exporting mgrs_tile: {e}")

    logger.info(f"All mgrs_tile export tasks started for years {years}")


def subset_export_mgrs_tiles(years: list | int):
    if isinstance(years, int) or isinstance(years, str):
        years = [years]
    mgrs_tiles = pd.read_csv(
        os.path.join(
            "data",
            "tiles",
            "mgrs_tiles",
            "mgrs_tiles_all_land_ecoregions.csv",
        )
    )
    mgrs_tiles_list = list(set(mgrs_tiles["mgrs_tile_3"].tolist()))
    include = ["19F", "19E", "20F"]
    mgrs_tiles_list = list(set([*mgrs_tiles_list, *include]))

    logger.debug(f"Exporting mgrs_tiles: {mgrs_tiles_list} for years {years}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(export_mgrs_tile, mgrs_tile, int(year))
            for mgrs_tile in mgrs_tiles_list
            for year in years
            if mgrs_tile in CONFIG["PIPELINE_PARAMS"]["MGRS_SUBSET_EXPORT"]
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            try:
                future.result()  # If the task raised an exception, this will raise it here
            except Exception as e:
                logger.error(f"Error exporting mgrs_tile: {e}")

    logger.info(f"All mgrs_tile export tasks started for years {years}")


# def test_single_export():
#     mgrs_tile = "18L"
#     # mgrs_tile = "34T"
#     export_mgrs_tile(mgrs_tile)


if __name__ == "__main__":

    logger.info(f"Output resolution: {CONFIG['GAPFILLING']['INPUT_RESOLUTION']}m")
    logger.info(f"Version: {CONFIG['GAPFILLING']['VERSION']}")

    # wait 1 hour
    # time.sleep(3 * 3600)
    # subset_export_mgrs_tiles(years=CONFIG["PIPELINE_PARAMS"]["YEARS"])
    global_export_mgrs_tiles(years=CONFIG["PIPELINE_PARAMS"]["YEARS"])
    # subset_export_mgrs_tiles(years=CONFIG["PIPELINE_PARAMS"]["YEARS"])
    # subset_export_mgrs_tiles(years=CONFIG["PIPELINE_PARAMS"]["YEARS"])
    # subset_export_mgrs_tiles(years=CONFIG["PIPELINE_PARAMS"]["YEARS"])
    # subset_export_mgrs_tiles(years=CONFIG["PIPELINE_PARAMS"]["YEARS"])
    # subset_export_mgrs_tiles(years=CONFIG["PIPELINE_PARAMS"]["YEARS"])
