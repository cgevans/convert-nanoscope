from __future__ import annotations
import re
from functools import cached_property
import numpy as np
from os import PathLike
import pathlib
from typing import Any


class NanoscopeFile:
    def __init__(self, path: str | PathLike):
        self._path = pathlib.Path(path)
        self._handle = self._path.open("rb")
        dlist: dict[str, Any] = {"Ciao image": []}
        vl: dict[str, Any] = {}
        for line in self._handle:
            if line.startswith(b"\\*File list end"):
                break
            elif m := re.match(r"\\\*(.*) list\r\n", line.decode("windows-1252")):
                vl = {}
                if m.group(1) != "Ciao image":
                    dlist[m.group(1)] = vl
                else:
                    dlist[m.group(1)].append(vl)
            elif line.startswith(b"\\"):
                l = line[1:].decode("windows-1252")
                k, v = l.split(": ", 1)
                vl[k] = v.strip()
        self.images = [NanoscopeImage(self, x) for x in dlist["Ciao image"]]

    def first_height_image(self):
        return next(x for x in self.images if (x.data_type == "Height") or (x.data_type == "ZSensor"))


class NanoscopeImage:
    def __init__(self, file: NanoscopeFile, metadata: dict[str, Any]):
        self._file = file
        self._metadata = metadata

    @cached_property
    def shape(self):
        x = int(self._metadata["Number of lines"])
        y = int(self._metadata["Samps/line"])
        return x, y

    @cached_property
    def data_type(self):
        d = re.search(r"\[([^]]+)\]", self._metadata["@2:Image Data"])
        if d:
            return d.group(1)
        else:
            return None

    @cached_property
    def raw_data(self):
        self._file._handle.seek(int(self._metadata["Data offset"]))
        return np.fromfile(
            self._file._handle,
            count=int(self._metadata["Data length"]) // 4,
            dtype=np.int32,
        ).reshape(self.shape)[::-1, :]

    def processed_data(self, planefit: bool = True, linemedian: bool = True):
        dat = self.raw_data
        if planefit:
            xv = dat.mean(axis=1)
            xpa = np.polyfit(np.arange(len(xv)), xv - xv[0], 1)[0]
            yv = dat.mean(axis=0)
            ypa = np.polyfit(np.arange(len(yv)), yv - yv[0], 1)[0]
            dat = (
                dat
                - xpa * np.arange(len(xv))[:, None]
                - ypa * np.arange(len(yv))[None, :]
            )
        if linemedian:
            dm = np.median(dat, axis=1)
            dm -= dm.mean()
            dat = dat - dm[:, None]
        return dat
