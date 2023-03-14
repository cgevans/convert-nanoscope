# Needs:
# - pip install pillow tqdm numpy click matplotlib

fontname = "FiraSans-Regular.ttf"

from os import PathLike
import pathlib
from typing import Any, Sequence
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont
from tqdm.contrib.concurrent import process_map
from functools import cached_property
import click
import matplotlib

class NanoscopeFile:
    def __init__(self, path: PathLike):
        self._path = pathlib.Path(path)
        self._handle = self._path.open('rb')
        dlist = {'Ciao image': []}
        vl = {}
        for line in self._handle:
            if line.startswith(b"\\*File list end"):
                break
            elif m := re.match(r"\\\*(.*) list\r\n", line.decode('windows-1252')):
                vl = {}
                if m.group(1) != "Ciao image":
                    dlist[m.group(1)] = vl
                else:
                    dlist[m.group(1)].append(vl)
            elif line.startswith(b"\\"):
                l = line[1:].decode('windows-1252')
                k, v = l.split(": ", 1)
                vl[k] = v.strip()
        self.images = [NanoscopeImage(self, x) for x in dlist['Ciao image']]

    def first_height_image(self):
        return next(x for x in self.images if x.data_type == 'Height')
        
class NanoscopeImage:
    def __init__(self, file: NanoscopeFile, metadata: dict[str, Any]):
        self._file = file
        self._metadata = metadata

    @cached_property
    def shape(self):
        x = int(self._metadata['Number of lines'])
        y = int(self._metadata['Samps/line'])
        return x, y
    
    @cached_property
    def data_type(self):
        d = re.search(r"\[([^]]+)\]", self._metadata['@2:Image Data'])
        if d:
            return d.group(1)
        else:
            return None
        
    @cached_property
    def raw_data(self):
        self._file._handle.seek(int(self._metadata['Data offset']))
        return np.fromfile(self._file._handle, count=int(self._metadata['Data length'])//4, dtype=np.int32).reshape(self.shape)[::-1,:]

    def processed_data(self, planefit: bool = True, linemedian: bool = True):
        dat = self.raw_data
        if planefit:
            xv = dat.mean(axis=1)
            xpa = np.polyfit(np.arange(len(xv)), xv - xv[0], 1)[0]
            yv = dat.mean(axis=0)
            ypa = np.polyfit(np.arange(len(yv)), yv - yv[0], 1)[0]
            dat = dat - xpa * np.arange(len(xv))[:,None] - ypa * np.arange(len(yv))[None,:]
        if linemedian:
            dm = np.median(dat, axis=1)
            dm -= dm.mean()
            dat = dat - dm[:,None]
        return dat

def proc_spm(fname: str, add_title: bool = True, add_bar: bool = True, fnt = None, cmap: str | None = None) -> Image:
    f = NanoscopeFile(fname)

    height = f.first_height_image()

    x,y,s = height._metadata["Scan Size"].split(" ")
    sss = (float(x), float(y), s.replace("~", "µ"))

    dat = height.processed_data()

    ds = dat.flatten()
    ds.sort()
    vmin = ds[int(0.002 * len(ds))]-1
    vmax = ds[int(0.998 * len(ds))]+1

    ylen, xlen = dat.shape

    fnt = ImageFont.truetype(fontname, int(26 * xlen / 1024))

    # fig, ax = plt.subplots(constrained_layout=True, figsize=(6,6))
    # ax.axis("off")
    # ax.set_position([0,0,1,1])
    # ax.imshow(dat, cmap="bone", vmin=vmin, vmax=vmax)
    # scalebar = ScaleBar(sss[0]/xlen, sss[2], length_fraction=0.25, frameon=False, color='white', location='lower right')
    # ax.add_artist(scalebar)

    iv = np.maximum(np.minimum(((dat-vmin)/(vmax-vmin)*255), 255), 0).astype(np.uint8)
    # create pillow image from array
    if cmap is not None:
        import matplotlib.cm as cm
        img = Image.fromarray((cm.get_cmap(cmap, lut=256)(iv)[:, :, :3]*255).astype(np.uint8), mode="RGB")
        an_color = "white"
    else:
        img = Image.fromarray(iv,  mode="L")
        an_color = 255


    if add_bar:
        d = ImageDraw.Draw(img)
        yspos = int(0.95 * ylen)
        xsend = int(0.95 * xlen)
        if sss[2] == "nm":
            nmpx = sss[0]/xlen
        elif sss[2] == "µm":
            nmpx = sss[0]/xlen * 1000
        possible = np.array([10, 100, 250, 500, 1000, 2000, 4000])
        texts = ["10 nm", "100 nm", "250 nm", "500 nm", "1 µm", "2 µm", "4 µm"]
        frac = (possible / nmpx) / xlen
        choice = np.nonzero(frac > 0.1)[0][0]
        size = possible[choice]/nmpx
        d.line([(xsend-size,yspos), (xsend,yspos)], fill=an_color, width=int(10 * xlen/1024))
        d.text((xsend-size//2,yspos+10), texts[choice], fill=an_color, font=fnt, align='center', anchor='ma')
    if add_title:
        img = add_filename(img, xlen, fname, fnt, an_color=an_color)
    return img

def add_filename(img: Image, xlen: int, fname: str, fnt=None, an_color=255):
    d = ImageDraw.Draw(img)
    d.text((xlen//2, 10), fname, fill=an_color, font=fnt, align='center', anchor='ma')
    return img

import glob

toconvert = glob.glob("*.spm")

def proc_and_save(v):
    fname, bar, title, cmap = v
    img = proc_spm(fname, add_title=title, add_bar=bar, cmap=cmap)
    img.save(fname.replace(".spm", ".png"))


@click.command()
@click.option('--title/--no-title', default=True, help='Add filename as a title to the image')
@click.option('--bar/--no-bar', default=True, help='Add a scale bar to the image')
@click.option('-r', '--recursive', is_flag=True, help='Search for files recursively')
@click.option('--cmap', '-c', type=str, help='Matplotlib colormap to use')
@click.argument('PATH', type=click.Path(exists=True), nargs=-1)
def main(title, bar, recursive, path: Sequence[str], cmap: str | None = None):
    toconvert = []

    if len(path) == 0:
        click.echo("No paths or filenames provided: converting SPM files in current directory.")
        path = ('./',)

    for p in path:
        pp = pathlib.Path(p)
        if pp.is_dir():
            if recursive:
                toconvert.extend(pp.glob('**/*.spm'))
            else:
                toconvert.extend(pp.glob('*.spm'))
        else:
            toconvert.append(pp)

    process_map(proc_and_save, [(str(x), bar, title, cmap) for x in toconvert])

if __name__ == '__main__':
    main()