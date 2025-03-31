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


def get_word_list() -> list[str]:
    """Gets a list of words from the input file."""
    words = []
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            words.extend(row)  # Assuming words are separated by commas
            if len(words) >= 10:
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

    # Add each word as a text object
    for i, word in enumerate(words):
        text_data = bpy.data.curves.new(name=f"Text_{i}", type="FONT")
        text_data.body = word.strip()  # Remove any extra whitespace
        text_object = bpy.data.objects.new(name=f"Text_{i}", object_data=text_data)
        texts_collection.objects.link(text_object)


main()
