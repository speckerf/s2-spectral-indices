import ee
from config import get_config

CONFIG_GEE_PIPELINE = get_config()


def apply_cloudScorePlus_mask(s2imgC: ee.ImageCollection) -> ee.ImageCollection:
    # code adopted from https://code.earthengine.google.com/5693305b63e347c83028ee1b6030b755
    csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    # Link S2 and CloudScore+ Mask.
    linkedCollection = s2imgC.linkCollection(
        csPlus, [CONFIG_GEE_PIPELINE["CLOUD_FILTERING"]["CLOUD_SCORE_PLUS_BAND"]]
    )

    s2CollectionMasked = linkedCollection.map(
        lambda image: image.updateMask(
            image.select(
                CONFIG_GEE_PIPELINE["CLOUD_FILTERING"]["CLOUD_SCORE_PLUS_BAND"]
            ).gte(CONFIG_GEE_PIPELINE["CLOUD_FILTERING"]["CLOUD_SCORE_PLUS_THRESHOLD"])
        )
    )

    return s2CollectionMasked


if __name__ == "__main__":
    ee.Initialize()
    # polygon 10km around ([8.50107079297198, 47.39400732763448])
    geometry = ee.Geometry.Polygon(
        [
            [
                [8.50107079297198 - 0.1, 47.39400732763448 - 0.1],
                [8.50107079297198 + 0.1, 47.39400732763448 - 0.1],
                [8.50107079297198 + 0.1, 47.39400732763448 + 0.1],
                [8.50107079297198 - 0.1, 47.39400732763448 + 0.1],
            ]
        ]
    )
    imgc = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate("2022-06-01", "2022-06-30")
    )
    imgc = apply_cloudScorePlus_mask(imgc)

    # create mean composite and test export
    composite = imgc.mean()
    task = ee.batch.Export.image.toAsset(
        image=composite,
        description="test_cloudfree_export",
        assetId="projects/ee-speckerfelix/assets/tests/test_cloudfree_export_10",
        region=geometry,
        scale=10,
    )

    task.start()
