"""Minimal stand-in for `xinshuo_visualization`.

AB3DMOT's `AB3DMOT_libs/vis.py` imports `random_colors` at module load time, and
`model.py` imports `vis.py` unconditionally. This evaluation runs with
visualization disabled (`cfg.vis = False`), so the drawing code is never called.
Rather than vendoring the whole Xinshuo_PyToolbox for one unused helper, we
provide an equivalent `random_colors` implementation here.

Only used to satisfy the import; if you enable AB3DMOT's visualization, install
the real toolbox instead: https://github.com/xinshuoweng/Xinshuo_PyToolbox
"""
import colorsys
import random


def random_colors(N, bright=True, seed=0):
    """N visually distinct RGB colors in [0,255], evenly spaced in HSV."""
    brightness = 1.0 if bright else 0.7
    hsv = [(i / float(N), 1, brightness) for i in range(max(int(N), 1))]
    colors = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(*h)) for h in hsv]
    random.Random(seed).shuffle(colors)
    return colors
