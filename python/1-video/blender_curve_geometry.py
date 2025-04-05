from math import *  # pylint: disable=wildcard-import, unused-wildcard-import, redefined-builtin
from mathutils import *  # pylint: disable=wildcard-import, unused-wildcard-import
import inspect
import os.path
import csv
import bpy
import numpy as np

C = bpy.context
D = bpy.data

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))


def main():
    edges = D.collections["Edges"].all_objects
    print(f"Num edges: {len(edges)}")

    for obj in edges:
        # https://blenderartists.org/t/how-to-copy-curve-geometry/520958/5
        if obj.type != "CURVE":
            continue

        obj.data.bevel_mode = "ROUND"
        obj.data.bevel_depth = np.random.uniform(0.001, 0.0025)
        obj.data.use_fill_caps = True


main()
