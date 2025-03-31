# https://b3d.interplanety.org/en/using-microsoft-visual-studio-code-as-external-ide-for-writing-blender-scripts-add-ons/
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

INPUT_FILE = os.path.join(path, "../../data/graph/nodes-first-random.csv")
COLLECTION_NAME = "Texts"
FONT_NAME = "Vermiglione Regular"


def get_word_list() -> list[str]:
    """Gets a list of words from the input file."""
    words = []
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            words.extend(row)  # Assuming words are separated by commas
            if len(words) >= 500:
                break

    # "word (pronunciation)" -> "word"
    words = [word.split(" (")[0] for word in words]

    return words


def ensure_collection(collection_name: str) -> bpy.types.Collection:
    """Ensures the collection exists and returns it."""
    if collection_name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection)
    else:
        new_collection = bpy.data.collections[collection_name]
    return new_collection


def main():
    """Main function to import words as text objects in Blender."""
    words = get_word_list()
    print(f"Words: {words}")

    texts_collection = ensure_collection(COLLECTION_NAME)

    font = D.fonts.get(FONT_NAME)
    if not font:
        raise ValueError(f"Font '{FONT_NAME}' not found in Blender.")

    for i, word in enumerate(words):
        text_data = D.curves.new(name=f"Word_{i}", type="FONT")
        text_data.body = word.strip()
        text_data.font = font
        text_object = D.objects.new(name=f"Word_{i}", object_data=text_data)
        texts_collection.objects.link(text_object)

        # Convert text object to mesh
        C.view_layer.objects.active = text_object
        text_object.select_set(True)
        bpy.ops.object.convert(target="MESH")
        bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS", center="BOUNDS")

        text_object.select_set(False)


main()
