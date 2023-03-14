from __future__ import annotations

# Needs:
# - pip install pillow tqdm numpy click matplotlib


from typing import Sequence
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.contrib.concurrent import process_map
import click
import pathlib
from .spm import NanoscopeFile

# A reasonable list of fonts that are likely to found *somewhere*:

FONTS_TO_TRY = [
    "Helvetica",
    "Arial",
    "arial",
    "FiraSans-Regular",
    "LiberationSans-Regular",
    "DejaVuSans",
]

USABLE_FONTNAME = None
for font in FONTS_TO_TRY:
    try:
        ImageFont.truetype(font)
        USABLE_FONTNAME = font
    except OSError:
        continue


def proc_spm(
    fname: str | NanoscopeFile,
    add_title: bool = True,
    add_bar: bool = True,
    font: None | str = None,
    cmap: str | None = None,
) -> Image.Image:
    if isinstance(fname, NanoscopeFile):
        f = fname
    else:
        f = NanoscopeFile(fname)

    height = f.first_height_image()

    x, y, s = height._metadata["Scan Size"].split(" ")
    sss = (float(x), float(y), s.replace("~", "µ"))

    dat = height.processed_data()

    ds = dat.flatten()
    ds.sort()
    vmin = ds[int(0.002 * len(ds))] - 1
    vmax = ds[int(0.998 * len(ds))] + 1

    ylen, xlen = dat.shape

    if isinstance(font, str):
        fnt = ImageFont.truetype(font, int(26 * xlen / 1024))
    else:
        if USABLE_FONTNAME is None:
            raise OSError("Can't find a usable font.")
        fnt = ImageFont.truetype(USABLE_FONTNAME, int(26 * xlen / 1024))

    # fig, ax = plt.subplots(constrained_layout=True, figsize=(6,6))
    # ax.axis("off")
    # ax.set_position([0,0,1,1])
    # ax.imshow(dat, cmap="bone", vmin=vmin, vmax=vmax)
    # scalebar = ScaleBar(sss[0]/xlen, sss[2], length_fraction=0.25, frameon=False, color='white', location='lower right')
    # ax.add_artist(scalebar)

    iv = np.maximum(np.minimum(((dat - vmin) / (vmax - vmin) * 255), 255), 0).astype(
        np.uint8
    )
    # create pillow image from array
    if cmap is not None:
        import matplotlib.cm as cm

        img = Image.fromarray(
            (cm.get_cmap(cmap, lut=256)(iv)[:, :, :3] * 255).astype(np.uint8),
            mode="RGB",
        )
    else:
        img = Image.fromarray(iv, mode="L")
    an_color = "white"

    if add_bar:
        d = ImageDraw.Draw(img)
        yspos = int(0.95 * ylen)
        xsend = int(0.95 * xlen)
        if sss[2] == "nm":
            nmpx = sss[0] / xlen
        elif sss[2] == "µm":
            nmpx = sss[0] / xlen * 1000
        possible = np.array([10, 100, 250, 500, 1000, 2000, 4000])
        texts = ["10 nm", "100 nm", "250 nm", "500 nm", "1 µm", "2 µm", "4 µm"]
        frac = (possible / nmpx) / xlen
        choice = np.nonzero(frac > 0.1)[0][0]
        size = possible[choice] / nmpx
        d.line(
            [(xsend - size, yspos), (xsend, yspos)],
            fill=an_color,
            width=int(10 * xlen / 1024),
        )
        d.text(
            (xsend - size // 2, yspos + 10),
            texts[choice],
            fill=an_color,
            font=fnt,
            align="center",
            anchor="ma",
        )
    if add_title:
        img = add_filename(img, xlen, str(fname), fnt, an_color=an_color)
    return img


def add_filename(img: Image.Image, xlen: int, fname: str, fnt=None, an_color=255):
    d = ImageDraw.Draw(img)
    d.text((xlen // 2, 10), fname, fill=an_color, font=fnt, align="center", anchor="ma")
    return img


import glob

toconvert = glob.glob("*.spm")


def proc_and_save(v):
    fname, bar, title, cmap = v
    img = proc_spm(fname, add_title=title, add_bar=bar, cmap=cmap)
    img.save(fname.replace(".spm", ".png"))


@click.command()
@click.option(
    "--title/--no-title", default=True, help="Add filename as a title to the image"
)
@click.option("--bar/--no-bar", default=True, help="Add a scale bar to the image")
@click.option("-r", "--recursive", is_flag=True, help="Search for files recursively")
@click.option("--cmap", "-c", type=str, help="Matplotlib colormap to use")
@click.argument("PATH", type=click.Path(exists=True), nargs=-1)
def main(title, bar, recursive, path: Sequence[str], cmap: str | None = None):
    toconvert: list[pathlib.Path | str] = []

    if len(path) == 0:
        click.echo(
            "No paths or filenames provided: converting SPM files in current directory."
        )
        path = ("./",)

    for p in path:
        pp = pathlib.Path(p)
        if pp.is_dir():
            if recursive:
                toconvert.extend(pp.glob("**/*.spm"))
            else:
                toconvert.extend(pp.glob("*.spm"))
        else:
            toconvert.append(pp)

    process_map(proc_and_save, [(str(x), bar, title, cmap) for x in toconvert])


if __name__ == "__main__":
    main()
