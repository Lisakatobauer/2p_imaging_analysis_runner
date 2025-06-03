import numpy as np
import re

import subprocess
from pathlib import Path

from scipy import interpolate


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def detect_bidi_offset(plane, maxoff=5, zrange=(0, -1)):
    """
    # Function written by Joe Donovan for bidirectional scan offsets correction
    :param zrange:
    :param maxoff:
    :param plane:
    :return:
    """

    aplane = plane[zrange[0]:zrange[1]].mean(0)
    offsets = np.arange(-maxoff + 1, maxoff)
    x = np.arange(aplane.shape[-1])
    minoff = []
    for row in range(2, plane.shape[-2] - 1, 2):
        f = interpolate.interp1d(x, (aplane[row + 1, :] + aplane[row - 1, :]) * .5, fill_value='extrapolate')

        offsetscores = np.asarray([np.mean(np.abs(aplane[row, :] - f(x + offset))) for offset in offsets])
        minoff.append(offsetscores.argmin())
    bestoffset = offsets[np.bincount(minoff).argmax()]
    print(f'the best bidirectional scan offset correction for this plane is {bestoffset}')
    return bestoffset


def bidi_offset_correction_plane(plane, offset=None, maxoff=5, zrange=(0, -1)):
    if not offset:
        offset = detect_bidi_offset(plane, maxoff=maxoff, zrange=zrange)
    plane[:, 1::2, :] = np.roll(plane[:, 1::2, :], -offset, -1)  # .astype(aplane.dtype)
    return plane


def get_git_root() -> Path:
    return Path(
        subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()
    )
