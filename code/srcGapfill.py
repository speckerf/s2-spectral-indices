import concurrent.futures
from typing import Optional, Tuple

import ee
from config import get_config
from loguru import logger
from tqdm import tqdm
from utilsGapfill import (
    get_projection_info,
    get_to_fill_mask,
    spatially_fill_image,
    temporally_fill_image,
)

CONFIG = get_config()
service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(
    service_account, "auth/gem-eth-analysis-24fe4261f029.json"
)
ee.Initialize(credentials, project="ee-speckerfelix")


def load_img_imgc_from_idx(
    system_index: str, input_collection: str = "raw"
) -> Tuple[ee.Image, ee.ImageCollection, str]:
    assert input_collection in ["raw", "spatial", "temporal"]

    resolution = CONFIG["GAPFILLING"]["INPUT_RESOLUTION"]
    version = CONFIG["GAPFILLING"]["VERSION"]

    if input_collection == "raw":
        imgc_path = f"projects/ee-speckerfelix/assets/oemc-spectral/spectral-indices_s2-srf-yearly_{resolution}m_{version}"
    elif input_collection == "spatial":
        imgc_path = f"projects/ee-speckerfelix/assets/oemc-spectral/spectral-indices_s2-srf-yearly-spatfill_{resolution}m_{version}"
    elif input_collection == "temporal":
        imgc_path = f"projects/ee-speckerfelix/assets/oemc-spectral/spectral-indices_s2-srf-yearly-tempfill_{resolution}m_{version}"

    imgc = ee.ImageCollection(imgc_path)

    assert (
        imgc.filter(ee.Filter.eq("system:index", system_index)).size().getInfo() == 1
    ), f"Image with system index {system_index} not found in ImageCollection."
    img = imgc.filter(ee.Filter.eq("system:index", system_index)).first()

    return img, imgc, imgc_path


def load_imgc_and_path(input_collection: str) -> Tuple[ee.ImageCollection, str]:
    assert input_collection in ["raw", "spatial", "temporal"]

    resolution = CONFIG["GAPFILLING"]["INPUT_RESOLUTION"]
    version = CONFIG["GAPFILLING"]["VERSION"]

    if input_collection == "raw":
        imgc_path = f"projects/ee-speckerfelix/assets/oemc-spectral/spectral-indices_s2-srf-yearly_{resolution}m_{version}"
    elif input_collection == "spatial":
        imgc_path = f"projects/ee-speckerfelix/assets/oemc-spectral/spectral-indices_s2-srf-yearly-spatfill_{resolution}m_{version}"
    elif input_collection == "temporal":
        imgc_path = f"projects/ee-speckerfelix/assets/oemc-spectral/spectral-indices_s2-srf-yearly-tempfill_{resolution}m_{version}"
    imgc = ee.ImageCollection(imgc_path)

    return imgc, imgc_path


def export_spatial_gapfill(
    system_index: str,
    input_collection: str = "raw",
    export: bool = True,
    img_temp_gapfilled: ee.Image = None,
) -> Optional[ee.Image]:
    if img_temp_gapfilled is None:
        img, imgc, imgc_path = load_img_imgc_from_idx(system_index, input_collection)
    else:
        img = img_temp_gapfilled

    to_fill_mask = get_to_fill_mask(img)
    img_filled = spatially_fill_image(img.select(CONFIG["INDICES"]), to_fill_mask)

    bit_position_spatial_fill = CONFIG["GAPFILLING"]["BITMASK_SPAT_FILL"]
    spatially_filled_mask = to_fill_mask.And(
        img_filled.select(CONFIG["INDICES"][0]).mask()
    ).gt(0)
    spatially_filled_add_bitmask = spatially_filled_mask.multiply(
        2**bit_position_spatial_fill
    )

    img_output = ee.Image.cat(
        [
            img_filled.toInt8(),
            img.select("QA_GAPFILL").add(spatially_filled_add_bitmask).toUint8(),
        ]
    )

    img_output = ee.Image(
        img_output.copyProperties(img)
        .set("system:time_start", img.get("system:time_start"))
        .set("system:time_end", img.get("system:time_end"))
        .set("spatial_gapfilling", True)
    )

    export_crs, export_crs_transform = get_projection_info(img)

    if temporally_fill_image:
        output_imgc_path = f"projects/ee-speckerfelix/assets/oemc-spectral/spectral-indices_s2-srf-yearly-gapfill_{CONFIG['GAPFILLING']['INPUT_RESOLUTION']}m_{CONFIG['GAPFILLING']['VERSION']}"
    else:
        output_imgc_path = f"projects/ee-speckerfelix/assets/oemc-spectral/spectral-indices_s2-srf-yearly-spatfill_{CONFIG['GAPFILLING']['INPUT_RESOLUTION']}m_{CONFIG['GAPFILLING']['VERSION']}"

    if export:
        task = ee.batch.Export.image.toAsset(
            image=img_output,
            description=f"Spatial Gapfilling: {system_index}",
            assetId=f"{output_imgc_path}/{system_index}",
            region=img.geometry().bounds(),
            crs=export_crs,
            crsTransform=export_crs_transform,
        )
        task.start()
        return None
    else:
        return img_output


def export_temp_fill(
    system_index: str, input_collection: str = "raw", export: bool = True
) -> Optional[ee.Image]:
    img, imgc, imgc_path = load_img_imgc_from_idx(system_index, input_collection)

    imgc_filtered = imgc.filter(ee.Filter.eq("mgrs_tile", img.get("mgrs_tile")))
    if imgc_filtered.size().getInfo() != 6:
        logger.warning(
            f"ImageCollection for mgrs tile {img.get('mgrs_tile').getInfo()} has {imgc_filtered.size().getInfo()} images."
        )

    img_filled = temporally_fill_image(
        img.select(CONFIG["INDICES"]),
        imgc_filtered,
        decay_factor=CONFIG["GAPFILLING"]["TEMPORAL_EXP_DECAY"],
    )

    to_fill_mask = get_to_fill_mask(img)

    bit_position_spatial_fill = CONFIG["GAPFILLING"]["BITMASK_TEMP_FILL"]
    temporally_filled_mask = to_fill_mask.And(
        img_filled.select(CONFIG["INDICES"][0]).mask()
    ).gt(0)
    temporally_filled_add_bitmask = temporally_filled_mask.multiply(
        2**bit_position_spatial_fill
    )

    img_output = ee.Image.cat(
        [
            img_filled.select(CONFIG["INDICES"]).toInt8(),
            img.select("QA_GAPFILL").add(temporally_filled_add_bitmask).toUint8(),
        ]
    )

    img_output = ee.Image(
        img_output.copyProperties(img)
        .set("system:time_start", img.get("system:time_start"))
        .set("system:time_end", img.get("system:time_end"))
        .set("temporal_gapfilling", True)
    )

    export_crs, export_crs_transform = get_projection_info(img)

    output_imgc_path = f"projects/ee-speckerfelix/assets/oemc-spectral/spectral-indices_s2-srf-yearly-tempfill_{CONFIG['GAPFILLING']['INPUT_RESOLUTION']}m_{CONFIG['GAPFILLING']['VERSION']}"

    if export:
        task = ee.batch.Export.image.toAsset(
            image=img_output,
            description=f"Temporal Gapfilling: {system_index}",
            assetId=f"{output_imgc_path}/{system_index}",
            region=img.geometry().bounds(),
            crs=export_crs,
            crsTransform=export_crs_transform,
        )
        task.start()
        return None
    else:
        return img_output


def export_sequential_gapfill(system_index: str) -> None:
    assert (
        CONFIG["GAPFILLING"]["STRATEGY_ORDER"][0] == "temporal"
    ), "Sequential gapfilling is only supported when temporal gapfilling is first."

    logger.info(f"Starting sequential gapfilling for {system_index}.")
    img_temp_filled = export_temp_fill(
        system_index, input_collection="raw", export=False
    )
    export_spatial_gapfill(
        system_index,
        export=True,
        img_temp_gapfilled=img_temp_filled,
    )


def complete_temporal_gapfill() -> None:
    if CONFIG["GAPFILLING"]["STRATEGY_ORDER"][0] == "temporal":
        imgc_input, imgc_input_path = load_imgc_and_path(input_collection="raw")
        imgc_output_path = imgc_input_path.replace(
            "spectral-indices_s2-srf-yearly", "spectral-indices_s2-srf-yearly-tempfill"
        )
        input_collection = "raw"
    elif CONFIG["GAPFILLING"]["STRATEGY_ORDER"][1] == "temporal":
        imgc_input, imgc_input_path = load_imgc_and_path(input_collection="spatial")
        imgc_output_path = imgc_input_path.replace(
            "spectral-indices_s2-srf-yearly-spatfill",
            "spectral-indices_s2-srf-yearly-tempfill",
        )
        input_collection = "spatial"

    imgc_output = ee.ImageCollection(imgc_output_path)
    system_indices_input = imgc_input.aggregate_array("system:index").getInfo()
    system_indices_output = imgc_output.aggregate_array("system:index").getInfo()
    # find all indices that need to be gapfilled
    indices_to_be_processed = list(
        set(system_indices_input) - set(system_indices_output)
    )

    logger.info(
        f"Starting temporal gapfilling for {len(indices_to_be_processed)} images."
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(export_temp_fill, system_idx, input_collection)
            for system_idx in indices_to_be_processed
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            try:
                future.result()  # If the task raised an exception, this will raise it here
            except Exception as e:
                logger.error(f"Error exporting temporal gapfilling: {e}")


def complete_spatial_gapfill() -> None:
    if CONFIG["GAPFILLING"]["STRATEGY_ORDER"][0] == "spatial":
        imgc_input, imgc_input_path = load_imgc_and_path(input_collection="raw")
        imgc_output_path = imgc_input_path.replace(
            "spectral-indices_s2-srf-yearly", "spectral-indices_s2-srf-yearly-spatfill"
        )
        input_collection = "raw"
    elif CONFIG["GAPFILLING"]["STRATEGY_ORDER"][1] == "spatial":
        imgc_input, imgc_input_path = load_imgc_and_path(input_collection="temporal")
        imgc_output_path = imgc_input_path.replace(
            "spectral-indices_s2-srf-yearly-tempfill",
            "spectral-indices_s2-srf-yearly-spatfill",
        )
        input_collection = "temporal"
    imgc_output = ee.ImageCollection(imgc_output_path)
    system_indices_input = imgc_input.aggregate_array("system:index").getInfo()
    system_indices_output = imgc_output.aggregate_array("system:index").getInfo()
    # find all indices that need to be gapfilled
    indices_to_be_processed = list(
        set(system_indices_input) - set(system_indices_output)
    )

    logger.info(
        f"Starting spatial gapfilling for {len(indices_to_be_processed)} images."
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(export_spatial_gapfill, system_idx, input_collection)
            for system_idx in indices_to_be_processed
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            try:
                future.result()  # If the task raised an exception, this will raise it here
            except Exception as e:
                logger.error(f"Error exporting temporal gapfilling: {e}")


def complete_sequential_gapfill() -> None:
    assert (
        CONFIG["GAPFILLING"]["STRATEGY_ORDER"][0] == "temporal"
    ), "Sequential gapfilling is only supported when temporal gapfilling is first."
    imgc_input, imgc_input_path = load_imgc_and_path(input_collection="raw")
    imgc_output_path = imgc_input_path.replace(
        "spectral-indices_s2-srf-yearly", "spectral-indices_s2-srf-yearly-gapfill"
    )
    input_collection = "raw"

    imgc_output = ee.ImageCollection(imgc_output_path)
    system_indices_input = imgc_input.aggregate_array("system:index").getInfo()
    system_indices_output = imgc_output.aggregate_array("system:index").getInfo()
    # find all indices that need to be gapfilled
    indices_to_be_processed = list(
        set(system_indices_input) - set(system_indices_output)
    )

    logger.info(
        f"Starting sequential gapfilling for {len(indices_to_be_processed)} images."
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(export_sequential_gapfill, system_idx)
            for system_idx in indices_to_be_processed
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            try:
                future.result()  # If the task raised an exception, this will raise it here
            except Exception as e:
                logger.error(f"Error exporting temporal gapfilling: {e}")


if __name__ == "__main__":
    # pass
    # export_spatial_gapfill(
    #     system_index="spectral-indices_s2-srf-yearly_m_1000m_s_20190101_20191231_T54M_epsg-32754_v01"
    # )
    # export_temp_fill(
    #     system_index="spectral-indices_s2-srf-yearly_m_1000m_s_20210101_20211231_T55M_epsg-32755_v01"
    # )
    # export_sequential_gapfill(
    #     system_index="spectral-indices_s2-srf-yearly_m_1000m_s_20210101_20211231_T55M_epsg-32755_v01"
    # )
    # complete_temporal_gapfill()
    complete_sequential_gapfill()
    # complete_spatial_gapfill()
    # temporal_gapfilling_test()
    # spatial_gapfilling_test()
