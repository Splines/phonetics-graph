from math import *  # pylint: disable=wildcard-import, unused-wildcard-import, redefined-builtin
from mathutils import *  # pylint: disable=wildcard-import, unused-wildcard-import
import inspect
import os.path
import csv
import bpy

C = bpy.context
D = bpy.data

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))


def ensure_collection(collection_name: str) -> bpy.types.Collection:
    """Ensures the collection exists and returns it."""
    if collection_name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(collection_name)
        C.scene.collection.children.link(new_collection)
    else:
        new_collection = bpy.data.collections[collection_name]
    return new_collection


def main():
    """Main function to add edges between every two words in the collection."""
    # Ensure the "Edges" collection exists
    edges_collection = ensure_collection("Edges")

    # Iterate over all objects labeled as "Word_{i}"
    word_objects = [obj for obj in D.objects if obj.name.startswith("Word_")]
    word_objects = word_objects[:10]  # Limit to the first 10 objects

    for i, obj1 in enumerate(word_objects):
        for obj2 in word_objects[i + 1 :]:
            # Create a new curve for the edge
            curve_data = D.curves.new(name=f"Edge_{obj1.name}_{obj2.name}", type="CURVE")
            curve_data.dimensions = "3D"
            spline = curve_data.splines.new(type="POLY")
            spline.points.add(1)  # Add two points (start and end)

            offset = 0.2
            start = obj1.location + Vector((offset, offset, offset))
            end = obj2.location + Vector((-offset, -offset, -offset))

            spline.points[0].co = (*start, 1.0)  # w=1.0 for homogeneous coordinates
            spline.points[1].co = (*end, 1.0)

            # Create a new object for the curve and link it to the "Edges" collection
            curve_obj = D.objects.new(name=f"Edge_{obj1.name}_{obj2.name}", object_data=curve_data)
            edges_collection.objects.link(curve_obj)

    print("Edges created successfully.")


main()
