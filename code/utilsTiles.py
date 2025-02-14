import ast
import os
import string

import ee
import geopandas as gpd
import pandas as pd
from config import get_config
from loguru import logger
from pyproj import CRS
from utilsCloudfree import apply_cloudScorePlus_mask

CONFIG = get_config()


def return_mgrs_bounding_box(mgrs_tiles: list) -> ee.Geometry.BBox:
    zone_number = int(mgrs_tiles[0][0:2])
    zone_letter = mgrs_tiles[0][2]

    longitudes = {
        "C": [-80, -72],
        "D": [-72, -64],
        "E": [-64, -56],
        "F": [-56, -48],
        "G": [-48, -40],
        "H": [-40, -32],
        "J": [-32, -24],
        "K": [-24, -16],
        "L": [-16, -8],
        "M": [-8, 0],
        "N": [0, 8],
        "P": [8, 16],
        "Q": [16, 24],
        "R": [24, 32],
        "S": [32, 40],
        "T": [40, 48],
        "U": [48, 56],
        "V": [56, 64],
        "W": [64, 72],
        "X": [72, 84],
    }

    west = -180 + (zone_number - 1) * 6
    south = longitudes[zone_letter][0]
    east = west + 6
    north = longitudes[zone_letter][1]

    return ee.Geometry.BBox(west, south, east, north)


def get_s2_indices_filtered(
    start_date: ee.Date,
    end_date: ee.Date,
    mgrs_tiles: list,
) -> pd.DataFrame:
    # load s2 data
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    bounding_box = return_mgrs_bounding_box(mgrs_tiles)
    imgc = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(bounding_box.buffer(1e6))
        .filter(ee.Filter.inList("MGRS_TILE", mgrs_tiles))
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                CONFIG["CLOUD_FILTERING"]["CLOUDY_PIXEL_PERCENTAGE"],
            )
        )
        .filter(
            ee.Filter.And(
                ee.Filter.eq("GENERAL_QUALITY", "PASSED"),
                ee.Filter.eq("GEOMETRIC_QUALITY", "PASSED"),
                ee.Filter.gt("system:asset_size", 1000000),
            )
        )
        .select([*bands, "SCL"])
    )

    # filter imgc by grouping by mgrs tile and orbit number
    s2_indices_filtered = groupby_mgrs_orbit_pandas(
        imgc,
    )
    return s2_indices_filtered


def add_group(image):
    orbit = image.getNumber("SENSING_ORBIT_NUMBER")
    tile = image.getString("MGRS_TILE")
    group = ee.String(orbit.format()).cat("_").cat(tile)
    return image.set("group", group)


def get_epsg_code_from_mgrs(mgrs_zone_number: str):
    """
    Get the EPSG code for a UTM zone from its MGRS zone number.

    Parameters:
    - zone_number (int): The MGRS zone number. Four digit string, e.g. 'T40K'
        - if last digit is from N to Z: Northern hemisphere
        - if last digit is from A to M: Southern hemisphere

    Returns:
    - int: The EPSG code.
    """
    if len(mgrs_zone_number) == 4 and mgrs_zone_number[0] == "T":
        mgrs_zone_number = mgrs_zone_number[1:]
    if mgrs_zone_number[-1] in string.ascii_uppercase[13:26]:
        southern_hemisphere = False
    elif mgrs_zone_number[-1] in string.ascii_uppercase[0:13]:
        southern_hemisphere = True
    else:
        raise ValueError("Invalid MGRS zone number.")
    zone_number = int(mgrs_zone_number[0:2])
    crs = CRS.from_dict(
        {
            "proj": "utm",
            "zone": zone_number,
            "south": southern_hemisphere,
            "datum": "WGS84",
            "units": "m",
        }
    )

    return crs.to_epsg()


def add_vegetion_index_weight(image: ee.Image) -> ee.Image:
    # use SCL to mask out all snow and water pixels, also defective pixels: mask out 0, 1, 2, 6, 11
    scl = image.select("SCL")
    scl_mask = scl.remap([0, 1, 2, 6, 11], [0, 0, 0, 0, 0], 1)

    # then mask based on esa world cover
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
    natural_classes = [10, 20, 30, 60, 70, 90, 95, 100]
    natural_mask = worldcover.remap(
        natural_classes, ee.List.repeat(1, len(natural_classes)), 0
    )

    if CONFIG["S2_FILTERING"]["VI_INDEX"] == "EVI":
        vi = image.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {
                "NIR": image.select("B8").divide(10000),
                "RED": image.select("B4").divide(10000),
                "BLUE": image.select("B2").divide(10000),
            },
        ).rename("vegetation_index")

    elif CONFIG["S2_FILTERING"]["VI_INDEX"] == "NDVI":
        vi = image.expression(
            "(NIR - RED) / (NIR + RED)",
            {
                "NIR": image.select("B8").divide(10000),
                "RED": image.select("B4").divide(10000),
            },
        ).rename("vegetation_index")
    else:
        raise ValueError("Index not recognized.")
    # mask out pixels that should not contribute to evi calculation
    vi = vi.updateMask(natural_mask).updateMask(scl_mask)

    mean_vi = vi.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=image.geometry(), scale=500
    ).getNumber("vegetation_index")
    mean_vi = ee.Algorithms.If(ee.Algorithms.IsEqual(mean_vi, None), -1, mean_vi)

    cloudy_pixel_percentage = image.getNumber("CLOUDY_PIXEL_PERCENTAGE").divide(100)
    vi_weight = ee.Number(1).subtract(mean_vi).multiply(2)
    cloud_pheno_weight_combined = cloudy_pixel_percentage.add(vi_weight)

    system_asset_size = image.getNumber("system:asset_size")
    water_percentage = image.getNumber("WATER_PERCENTAGE").divide(100)
    snow_ice_percentage = image.getNumber("SNOW_ICE_PERCENTAGE").divide(100)
    vegetation_percentage = image.getNumber("VEGETATION_PERCENTAGE").divide(100)

    return image.set(
        "cloud_pheno_image_weight",
        cloud_pheno_weight_combined,
        "mean_vegetation_index",
        mean_vi,
        "vegetation_index_weight",
        vi_weight,
        "cloudy_pixel_percentage",
        cloudy_pixel_percentage,
        "system_asset_size",
        system_asset_size,
        "water_percentage",
        water_percentage,
        "snow_ice_percentage",
        snow_ice_percentage,
        "vegetation_percentage",
        vegetation_percentage,
    )


def groupby_mgrs_orbit_pandas(
    imgc: ee.ImageCollection,
) -> ee.List:

    imgc = imgc.map(add_group)

    # mask out clouds and shadows using cloudscore plus
    imgc = apply_cloudScorePlus_mask(imgc)

    # add pheno distance weight / new with evi instead of ndvi
    imgc = imgc.map(add_vegetion_index_weight)

    # convert imagecollection to pandas data frame: with system:index, group, CLOUDY_PIXEL_PERCENTAGE, and pheno_weoght

    def extract_properties(img):
        return ee.Feature(
            None,
            {
                "s2_index": img.getString("system:index"),
                "group": img.getString("group"),
                "mgrs_tile": img.getString("MGRS_TILE"),
                "orbit": img.getNumber("SENSING_ORBIT_NUMBER"),
                "cloud_pheno_image_weight": img.getNumber("cloud_pheno_image_weight"),
                "cloudy_pixel_percentage": img.getNumber("cloudy_pixel_percentage"),
                "pheno_distance_weight": img.getNumber("pheno_distance_weight"),
                "mean_vegetation_index": img.getNumber("mean_vegetation_index"),
                "vegetation_index_weight": img.getNumber("vegetation_index_weight"),
                "system_asset_size": img.getNumber("system_asset_size"),
                "water_percentage": img.getNumber("water_percentage"),
                "snow_ice_percentage": img.getNumber("snow_ice_percentage"),
                "vegetation_percentage": img.getNumber("vegetation_percentage"),
            },
        )

    fc = imgc.map(extract_properties)
    fc_computed = ee.data.computeFeatures({"expression": fc})

    total_features_retrieved = len(fc_computed.get("features", {}))
    if total_features_retrieved == 0:
        logger.error(f"Empty feature collection; after grouping.")
        return []
    if "nextPageToken" in fc_computed:
        logger.debug(
            f"Partial retrieval of {total_features_retrieved} features. Fetching the rest."
        )
        fc_list = [fc_computed]
        while "nextPageToken" in fc_computed:
            fc_computed = ee.data.computeFeatures(
                {"expression": fc, "pageToken": fc_computed["nextPageToken"]}
            )
            fc_list.append(fc_computed)
            total_features_retrieved += len(fc_computed["features"])
            logger.debug(
                f"Partial retrieval of {total_features_retrieved} features. Fetching the rest."
            )
        logger.debug(f"Total features retrieved: {total_features_retrieved}")
        fc_concat = {
            "type": "FeatureCollection",
            "features": [feature for fc in fc_list for feature in fc["features"]],
        }
    else:
        logger.debug(f"Total features retrieved: {total_features_retrieved}")
        fc_concat = fc_computed

    # parse dict to pandas dataframe

    # convert feature collection to pandas dataframe # drop geometry column
    df = gpd.GeoDataFrame.from_features(fc_concat).drop(columns="geometry")
    # # cast orbit to int (not np.int64)
    df["orbit"] = df["orbit"].astype(int)

    # filter each mgrs tile by the relevant orbits
    relevant_orbits_df = (
        pd.read_csv(
            os.path.join("data", "tiles", "s2_orbits_per_mgrs_tile_merged.csv"),
        )
        .drop(columns=["system:index", ".geo"])
        .rename(columns={"MGRS_TILE": "mgrs_tile"})
    )
    # Parse the columns as lists
    list_columns = ["ORBITS_TO_KEEP", "ALL_ORBITS", "ORBIT_MEAN_NODATA_PERCENTAGE"]
    for col in list_columns:
        relevant_orbits_df[col] = relevant_orbits_df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    # merge the two dataframes
    df = df.merge(relevant_orbits_df, on="mgrs_tile")

    # Filter rows where the 'orbit' is in the 'ORBITS_TO_KEEP' list for each row
    df = df[df.apply(lambda row: row["orbit"] in row["ORBITS_TO_KEEP"], axis=1)]

    # filter by group and sort by cloud_pheno_image_weight, limit to 10 images per group
    df_sorted = df.sort_values(
        by=["group", "cloud_pheno_image_weight"], ascending=[True, True]
    )

    # filter out all evi larger than 1.0 and smaller than -1.0
    df_sorted_1 = df_sorted[
        (df_sorted["mean_vegetation_index"] <= 1.0)
        & (df_sorted["mean_vegetation_index"] > -1.0)
    ]

    df_sorted_3 = df_sorted_1  # legacy code
    # group by group and date of acquisition, only keep largest image (by system_asset_size)
    df_sorted_3.loc[:, "acquisition_date"] = df_sorted_3["s2_index"].str.slice(0, 8)
    df_sorted_4 = (
        df_sorted_3.groupby(["group", "acquisition_date"])
        .apply(lambda x: x.nlargest(1, "system_asset_size"))
        .reset_index(drop=True)
    )

    # per group, drop images if their mean_evi is below 0.9 quantile - MAX_EVI_DIFFERENCE
    df_sorted_4["mean_vi_diff"] = df_sorted_4.groupby("group")[
        "mean_vegetation_index"
    ].transform(lambda x: x.quantile(CONFIG["S2_FILTERING"]["VI_MAX_PERCENTILE"]) - x)
    df_sorted_5 = df_sorted_4[
        df_sorted_4["mean_vi_diff"] <= CONFIG["S2_FILTERING"]["MAX_VI_DIFFERENCE"]
    ]

    # from the remaining images, select the top MAX_IMAGES_PER_GROUP images per group
    df_filtered = (
        df_sorted_5.groupby("group")
        .apply(
            lambda x: x.nsmallest(
                CONFIG["S2_FILTERING"]["MAX_IMAGES_PER_GROUP"],
                "cloud_pheno_image_weight",
            )
        )
        .reset_index(drop=True)
    )

    logger.debug(f"imgc size after grouping and filtering: {df_filtered.shape[0]}")

    # return list of system:index
    return df_filtered["s2_index"].tolist()


def save_mgrs_tiles_for_ecoregion(eco_id):
    # get sentinel-2 mgrs tiles in this ecoregion
    ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    ecoregion_geometry = ecoregions.filter(ee.Filter.eq("ECO_ID", eco_id)).geometry()

    s2_imgc = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate("2023-06-01", "2023-06-30")
        .filterBounds(ecoregion_geometry)
    )

    mgrs_tiles = s2_imgc.aggregate_array("MGRS_TILE").getInfo()
    unique_mgrs_tiles = list(set(mgrs_tiles))

    # save to csv
    df = pd.DataFrame(unique_mgrs_tiles, columns=["mgrs_tile"])

    filename = f"mgrs_tiles_ecoregion_{eco_id}.csv"
    foldername = os.path.join("data", "tiles", "mgrs_tiles")

    os.makedirs(foldername, exist_ok=True)

    with open(os.path.join(foldername, filename), "w") as f:
        df.to_csv(f, index=False)


def save_mgrs_tiles_all_ecoregions():
    from concurrent.futures import ThreadPoolExecutor

    ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    eco_ids = ecoregions.aggregate_array("ECO_ID").getInfo()

    # Use ThreadPoolExecutor to execute the loop in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(save_mgrs_tiles_for_ecoregion, eco_ids)


# def get_all_land_mgrs_tiles():
#     # load all csv file in data/gee_pipeline/outputs/mgrs_tiles, add eco_id as column
#     foldername = os.path.join("data", "gee_pipeline", "outputs", "mgrs_tiles_per_eco")
#     files = [f for f in os.listdir(foldername) if f.endswith(".csv")]
#     df = pd.DataFrame()
#     for file in files:
#         eco_id = file.split("_")[-1].split(".")[0]
#         df_temp = pd.read_csv(os.path.join(foldername, file))
#         df_temp["eco_id"] = eco_id
#         df = pd.concat([df, df_temp])

#     # now read file data/phenology_pipeline/outputs/ecoregions_to_exclude_all.csv
#     df_exclude = pd.read_csv(
#         os.path.join(
#             "data", "phenology_pipeline", "outputs", "ecoregions_to_exclude_all.csv"
#         )
#     )

#     # filter out the ecoregions to exclude
#     df = df[~df["eco_id"].isin(df_exclude["ECO_ID"])]

#     # from 5 letter mgrs tile, get the 3 letter mgrs tile
#     df["mgrs_tile_3"] = df["mgrs_tile"].str[:3]

#     # save file
#     df.to_csv(
#         os.path.join(foldername, "mgrs_tiles_all_land_ecoregions.csv"), index=False
#     )

#     return df


if __name__ == "__main__":

    ee.Initialize(project="ee-speckerfelix")

    # get_all_land_mgrs_tiles()
    # save_mgrs_tiles_all_ecoregions()
    # get_s2_indices_filtered(
