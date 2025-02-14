from typing import Dict, Optional

import ee
from config import get_config
from loguru import logger

CONFIG = get_config()


class SpectralIndices:
    """
    A class to store expressions for computing spectral indices in Earth Engine.
    """

    INDICES: Dict[str, str] = {
        "NDVI": "(NIR - RED) / (NIR + RED)",
        "EVI": "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        "EVI2": "2.4 * ((NIR - RED) / (NIR + RED + 1))",  # https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/evi2/
        "NDWI": "(GREEN - NIR) / (GREEN + NIR)",
        "NDMI": "(NIR - SWIR1) / (NIR + SWIR1)",
        "SAVI": "1.5 * ((NIR - RED) / (NIR + RED + 0.5))",
        "NBRI": "(NIR - SWIR2) / (NIR + SWIR2)",  # Normalized Burned Ratio Index: https://www.geo.university/pages/spectral-indices-with-multispectral-satellite-data
        "BSI": "((SWIR1 + RED) - (NIR + BLUE))/((SWIR1 + RED) + (NIR + BLUE))",  # Bare Soil Index: https://www.geo.university/pages/spectral-indices-with-multispectral-satellite-data
    }

    BANDS: Dict[str, Dict[str, str]] = {
        "NDVI": {"NIR": "B8", "RED": "B4"},
        "EVI": {"NIR": "B8", "RED": "B4", "BLUE": "B2"},
        "EVI2": {"NIR": "B8", "RED": "B4"},
        "NDWI": {"GREEN": "B3", "NIR": "B8"},
        "NDMI": {"NIR": "B8", "SWIR1": "B11"},
        "SAVI": {"NIR": "B8", "RED": "B4"},
        "NBRI": {"NIR": "B8", "SWIR2": "B12"},
        "BSI": {"SWIR1": "B11", "RED": "B4", "NIR": "B8", "BLUE": "B2"},
    }

    @classmethod
    def compute_index(cls, img: ee.Image, index_name: str) -> ee.Image:
        """Computes a spectral index on a given image."""
        expression = cls.INDICES.get(index_name)
        bands = cls.BANDS.get(index_name)

        if expression and bands:
            return img.expression(
                expression, {k: img.select(v) for k, v in bands.items()}
            ).rename(index_name)
        else:
            logger.warning(f"Index {index_name} not found.")
            return img


def apply_spectral_indices(imgc: ee.ImageCollection) -> ee.ImageCollection:
    """Applies spectral indices to an ImageCollection."""

    # assert that all indices in CONFIG_SPECTRAL are implemented
    for index_name in CONFIG["INDICES"]:
        if index_name not in SpectralIndices.INDICES:
            logger.error(f"Index {index_name} not implemented in SpectralIndices.")

    def compute_indices(img: ee.Image) -> ee.Image:
        for index_name in CONFIG["INDICES"]:
            img = img.addBands(SpectralIndices.compute_index(img, index_name))
        return img

    return imgc.map(compute_indices)


def collapse_and_cast_spectral_indices(imgc: ee.ImageCollection) -> ee.Image:
    """Collapses an ImageCollection to a single image with mean spectral indices."""
    mean_img = imgc.mean()

    # check if cast to uint8 is necessary
    if CONFIG["PIPELINE_PARAMS"]["CAST_TO_INT8"]:
        return mean_img.multiply(CONFIG["PIPELINE_PARAMS"]["INT8_SCALE"]).int8()
    else:
        return mean_img


def test():
    # Zurich
    test_point = ee.Geometry.Point(8.5417, 47.3769)

    s2_imgc = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(test_point)
        .filterDate("2021-06-01", "2021-08-31")
    )

    # divide by 10000 to get reflectance
    s2_imgc = s2_imgc.map(lambda img: img.divide(10000))

    # Apply cloud mask
    s2_imgc = apply_cloudScorePlus_mask(s2_imgc)

    # Predict spectral indices
    preds = apply_spectral_indices(s2_imgc).select(CONFIG["INDICES"])
    final_img = collapse_and_cast_spectral_indices(preds)

    # test export
    bounds = s2_imgc.geometry().bounds()
    task = ee.batch.Export.image.toAsset(
        image=final_img,
        description="test_spectral_indices_export",
        assetId="projects/ee-speckerfelix/assets/tests/test_spectral_indices_export_7",
        region=bounds,
        scale=100,
    )
    task.start()


if __name__ == "__main__":
    from utilsCloudfree import apply_cloudScorePlus_mask

    ee.Initialize()
    test()
