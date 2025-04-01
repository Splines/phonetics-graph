import bpy
import os
from contextlib import contextmanager


# https://blender.stackexchange.com/a/306087/
@contextmanager
def console_override():
    try:
        area = next(a for a in bpy.context.screen.areas if a.type == "CONSOLE")
        with bpy.context.temp_override(area=area):
            yield area
    except StopIteration:
        yield None


def print(*texts):
    text = " ".join(str(i) for i in texts)
    with console_override() as area:
        if area is None:
            return
        for line in text.split("\n"):
            bpy.ops.console.scrollback_append(text=line, type="OUTPUT")


filepath = os.path.join(
    "Z:", "home", "dominic", "dev", "phonetics-graph", "python", "1-video", "blender_import_text.py"
)

try:
    exec(compile(open(filepath).read(), filepath, "exec"))
except Exception as e:
    print(f"Error: {e}")
