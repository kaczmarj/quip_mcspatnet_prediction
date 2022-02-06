"""Separate a whole slide image into PNG patches."""

import argparse
from pathlib import Path

import numpy as np
import openslide
from PIL import Image


def _get_parsed_args(args=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input-slide", required=True, help="Whole slide image path.")
    p.add_argument("-o", "--output-dir", required=True, help="Output directory.")

    p.add_argument(
        "-s",
        "--patch-size",
        type=int,
        required=True,
        help="Patch size in pixels at target magnification.",
    )
    p.add_argument(
        "-m",
        "--mag",
        required=True,
        type=float,
        help="Target magnification.",
    )
    p.add_argument("--mag-40x-magic-number", type=float, required=True, help="???")

    p.add_argument(
        "--use-mask",
        dest="use_mask",
        action="store_true",
        help="Calculate a mask to only extract patches containing tissue.",
    )
    p.add_argument("--no-use-mask", dest="use_mask", action="store_false")
    p.set_defaults(feature=True)

    p.add_argument("--no-use-mask", action="store_false")

    # We convert this to a dict so that invalid keys will raise a KeyError.
    return vars(p.parse_args(args))


def _get_mag(oslide, mag_40x_magic_number):
    if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
        mag = mag_40x_magic_number / float(
            oslide.properties[openslide.PROPERTY_NAME_MPP_X]
        )
    elif "XResolution" in oslide.properties:
        mag = mag_40x_magic_number / float(oslide.properties["XResolution"])
    # for Multiplex IHC WSIs, .tiff images
    elif "tiff.XResolution" in oslide.properties:
        mag = mag_40x_magic_number / float(oslide.properties["tiff.XResolution"])
    else:
        mag = mag_40x_magic_number / float(0.254)
    return mag


def _get_mask(oslide: openslide.OpenSlide):
    from skimage import color, filters, morphology

    level = oslide.level_count - 1
    size = oslide.level_dimensions[level]
    thumb = oslide.read_region(0, 0, level=level, size=size)
    thumb = np.asarray(thumb)

    thumb = color.rgb2gray(thumb)
    thresh = filters.threshold_otsu(thumb)
    mask = thumb > thresh
    mask = morphology.remove_small_objects(mask == 0, min_size=16 * 16, connectivity=2)
    mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)
    mask = morphology.binary_dilation(mask, morphology.disk(16))
    return mask


def _generate_patches(arg_dict):
    oslide = openslide.OpenSlide(arg_dict["input_slide"])
    mag = _get_mag(oslide, arg_dict["mag_40x_magic_number"])

    pw = int(arg_dict["patch_size"] * mag / arg_dict["mag"])
    width, height = oslide.dimensions[:2]

    mask = None
    if arg_dict["use_mask"]:
        mask = _get_mask(oslide)
        # Numpy array images have shape (height, width, ?channels).
        downsample_ratio = mask.shape[1] / width

    for x in range(1, width, pw):
        for y in range(1, height, pw):
            if x + pw > width:
                pw_x = width - x
            else:
                pw_x = pw
            if y + pw > height:
                pw_y = height - y
            else:
                pw_y = pw

            if (
                (int(arg_dict["patch_size"] * pw_x / pw) <= 0)
                or (int(arg_dict["patch_size"] * pw_y / pw) <= 0)
                or (pw_x <= 0)
                or (pw_y <= 0)
            ):
                continue

            # Skip if patch is not in the binary mask.
            # TODO: FIX THIS.
            if mask is not None:
                if mask[downsample_ratio * y, downsample_ratio * x]:
                    continue

            patch = oslide.read_region((x, y), 0, (pw_x, pw_y))
            # shahira: skip where the alpha channel is zero
            patch_arr = np.asarray(patch)
            if patch_arr[:, :, 3].max() == 0:
                continue
            patch = patch.resize(
                (
                    int(arg_dict["patch_size"] * pw_x / pw),
                    int(arg_dict["patch_size"] * pw_y / pw),
                ),
                Image.LANCZOS,
            )
            name = f"{x}_{y}_{pw}_{arg_dict['patch_size']}.png"
            yield patch, name


def main(args=None):
    ns = _get_parsed_args(args)

    output_dir = Path(ns["output_dir"])
    if output_dir.exists():
        print(f"Output directory exists: {output_dir}")
        print("Not doing anything... and exiting.")
        return

    for i, (img, name) in _generate_patches(ns):
        img.save(output_dir / name)
        if i % 100 == 0:
            print(f"Saving tile {i}")


if __name__ == "__main__":
    main()
