from collections import Counter
from itertools import accumulate
from typing import Tuple

import ee
from config import get_config
from loguru import logger

CONFIG = get_config()


# Define exponential weighting function
def exp_weight(delta, decay_factor) -> ee.Image:
    return ee.Image(
        ee.Number(delta).abs().multiply(ee.Number(decay_factor).multiply(-1)).exp()
    )


def get_prime_factors(n: int) -> list:
    """Returns the list of prime factors of n with multiplicities."""
    factors = []
    i = 2
    while i * i <= n:  # Check up to sqrt(n)
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 1
    if n > 1:  # If there's any prime factor greater than sqrt(n)
        factors.append(n)
    return factors


def get_common_primefactors(a: int, b: int) -> list:
    """Returns the common prime factors of a and b, considering multiplicities."""
    a_factors = Counter(get_prime_factors(a))
    b_factors = Counter(get_prime_factors(b))

    # Find the minimum multiplicity of common factors
    common_factors = []
    for factor in a_factors.keys() & b_factors.keys():
        common_factors.extend([factor] * min(a_factors[factor], b_factors[factor]))

    common_factors.sort()
    return common_factors


def get_cumulative_products(factors: list) -> list:
    """Returns a list of cumulative products from the given prime factors."""
    return list(accumulate(factors, lambda x, y: x * y))


def get_projection_info(img: ee.Image) -> Tuple[str, list]:
    """Returns the CRS and transform of the given image."""
    proj = img.projection()
    return proj.crs().getInfo(), proj.getInfo()["transform"]


def compute_reduced_transforms(scale, export_crs_transform):
    """Determine resolution reduction factors and compute reduced transforms."""
    if scale not in [20, 100, 1000]:
        raise ValueError("Invalid output scale. Must be one of [20, 100, 1000].")
    resolution_factors = {20: [2, 2, 5, 5, 5], 100: [2, 2, 5, 5], 1000: [2, 5]}
    factors = resolution_factors.get(scale, [])
    reduced_scales = [i * scale for i in get_cumulative_products(factors)]
    return [
        [s, 0, export_crs_transform[2], 0, -s, export_crs_transform[5]]
        for s in reduced_scales
    ]


def bitwiseExtract(input, fromBit, toBit):
    maskSize = ee.Number(1).add(toBit).subtract(fromBit)
    mask = ee.Number(1).leftShift(maskSize).subtract(1)
    return input.rightShift(fromBit).bitwiseAnd(mask)


def get_to_fill_mask(img: ee.Image) -> ee.Image:
    """Returns the mask of pixels to fill."""
    # needs to be not ocean and masked in input image / and have had valid input S2 data for that pixel (stored in QA)
    ocean_mask = (
        ee.ImageCollection("ESA/WorldCover/v200").first().unmask(199).neq(199)
    )  # -> ocean = 0, no ocean = 1
    # create bitmask for QA band at bit position 0
    bits = bitwiseExtract(img.select("QA_GAPFILL"), 0, 0).eq(1)
    return img.select(CONFIG["INDICES"][0]).mask().Not().And(ocean_mask).And(bits).gt(0)


def spatially_fill_image(img: ee.Image, to_fill_mask: ee.Image) -> ee.Image:
    img = img.select(CONFIG["INDICES"])
    export_crs, export_crs_transform = get_projection_info(img)

    scale = export_crs_transform[0]
    reduced_transforms = compute_reduced_transforms(scale, export_crs_transform)

    # to_fill_mask = get_to_fill_mask(img)

    fills = ee.List(reduced_transforms).map(
        lambda tr: img.reduceResolution(
            reducer=ee.Reducer.mean(), bestEffort=True, maxPixels=128
        )
        .reproject(
            crs=export_crs,
            crsTransform=tr,
        )
        .updateMask(to_fill_mask)
    )

    list_for_mosaic = ee.List([img]).cat([fills]).flatten().reverse()
    img_filled = ee.ImageCollection(list_for_mosaic).mosaic()

    return img_filled


def temporally_fill_image(
    img: ee.Image,
    imgc_filtered: ee.ImageCollection,
    decay_factor: float,
) -> ee.Image:
    """
    Gapfill a single image using temporal gapfilling. Weights are computed using an exponential decay function.

    Args:
        img: Image to fill. Should not have QA_GAPFILL band, resp. selects only the indices.
        imgc: ImageCollection to use for gapfilling.
        decay_factor: Exponential decay factor.
    """
    img = img.select(CONFIG["INDICES"])
    imgc_filtered = imgc_filtered.select(CONFIG["INDICES"])

    year = ee.Number(img.get("year"))
    years = imgc_filtered.aggregate_array("year").sort()

    logger.debug(f"Year of img: {year.getInfo()}; Years in imgc: {years.getInfo()}")

    def get_fill_image(other_yr: ee.Number) -> ee.Image:
        other_yr = ee.Number(other_yr)
        img_other = imgc_filtered.filterDate(ee.String(other_yr.toInt())).first()
        other_year_weight = exp_weight(ee.Number(other_yr).subtract(year), decay_factor)
        year_fill_img = img_other.multiply(other_year_weight)
        return year_fill_img.toFloat()

    def get_fill_weight_image(other_yr: ee.Number) -> ee.Image:
        other_yr = ee.Number(other_yr)
        img_other = imgc_filtered.filterDate(ee.String(other_yr.toInt())).first()
        img_other_mask = img_other.mask()
        other_year_weight = exp_weight(ee.Number(other_yr).subtract(year), decay_factor)
        return img_other_mask.multiply(other_year_weight).toFloat()

    years_without_self = years.remove(year)
    filter_years_apart = ee.Filter.And(
        ee.Filter.gte("item", year.subtract(CONFIG["GAPFILLING"]["MAX_YEARS_APART"])),
        ee.Filter.lte("item", year.add(CONFIG["GAPFILLING"]["MAX_YEARS_APART"])),
    )
    years_without_self_filtered = years_without_self.filter(filter_years_apart)
    years_fill_images = years_without_self_filtered.map(get_fill_image)
    weights_fill_images = years_without_self_filtered.map(get_fill_weight_image)

    vi_weighted_sum = ee.ImageCollection(years_fill_images).reduce(ee.Reducer.sum())
    weight_total = ee.ImageCollection(weights_fill_images).reduce(ee.Reducer.sum())
    filled = ee.Image(vi_weighted_sum).divide(weight_total)

    img_filled = img.unmask(filled).toInt8()
    return img_filled
