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

PARTICLE_SYSTEM_OBJECTS = [
    "Words1",
    "Words2",
    "Words3",
    "Words4",
    "Words5",
]

NUM_PARTICLES = 20


def ensure_collection(collection_name: str) -> bpy.types.Collection:
    """Ensures the collection exists and returns it."""
    if collection_name not in D.collections:
        new_collection = D.collections.new(collection_name)
        C.scene.collection.children.link(new_collection)
    else:
        new_collection = D.collections[collection_name]
    return new_collection


def get_particle_systems() -> list[bpy.types.ParticleSystem]:
    """Gets a list of particle systems from the specified objects."""
    particle_systems = []
    for obj in D.objects:
        if not obj.name in PARTICLE_SYSTEM_OBJECTS or not obj.particle_systems:
            continue

        deps_graph = C.evaluated_depsgraph_get()
        evaluated_obj = obj.evaluated_get(deps_graph)
        particle_system = evaluated_obj.particle_systems[0]
        particle_systems.append(particle_system)

    assert len(particle_systems) == len(PARTICLE_SYSTEM_OBJECTS)
    return particle_systems


def main():
    """Main function to add edges between every two words in the collection."""
    # Ensure the "Edges" collection exists
    edges_collection = ensure_collection("Edges")
    particle_systems = get_particle_systems()

    for num_system, system in enumerate(particle_systems):
        particles = system.particles[:NUM_PARTICLES]

        subcollection = D.collections.new(f"Edges_{num_system + 1}")
        edges_collection.children.link(subcollection)

        print(f"Creating edges for particle system: {system.settings.name}")
        for i, p1 in enumerate(particles):
            for j, p2 in enumerate(particles[i + 1 :]):
                edge_name = f"ps{num_system + 1} Edge_{i}_{j}"

                curve_data = D.curves.new(name=f"{edge_name}-curve", type="CURVE")
                curve_data.dimensions = "3D"
                spline = curve_data.splines.new(type="POLY")
                spline.points.add(1)
                spline.points[0].co = (*p1.location, 1.0)
                spline.points[1].co = (*p2.location, 1.0)

                curve_obj = D.objects.new(name=edge_name, object_data=curve_data)
                subcollection.objects.link(curve_obj)

    print("Edges created and constrained to particle locations successfully.")


def keyframe_edges():
    """Keyframe edges to follow particle locations."""
    particle_systems = get_particle_systems()

    print("Updating edges for every frame...")
    scene = C.scene
    for frame in range(2000, 2600):
        scene.frame_set(frame)
        for num_system, system in enumerate(particle_systems):
            particles = system.particles[:NUM_PARTICLES]
            for i, p1 in enumerate(particles):
                for j, p2 in enumerate(particles[i + 1 :]):
                    spline = D.objects[f"ps{num_system + 1} Edge_{i}_{j}"].data.splines[0]
                    spline.points[0].co = (*p1.location, 1.0)
                    spline.points[0].keyframe_insert(data_path="co", frame=frame)
                    spline.points[1].co = (*p2.location, 1.0)
                    spline.points[1].keyframe_insert(data_path="co", frame=frame)

    print("Edges updated for every frame successfully.")


# main()
keyframe_edges()
