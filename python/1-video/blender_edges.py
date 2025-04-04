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


def ensure_collection(collection_name: str) -> bpy.types.Collection:
    """Ensures the collection exists and returns it."""
    if collection_name not in D.collections:
        new_collection = D.collections.new(collection_name)
        C.scene.collection.children.link(new_collection)
    else:
        new_collection = D.collections[collection_name]
    return new_collection


def main():
    """Main function to add edges between every two words in the collection."""
    # Ensure the "Edges" collection exists
    edges_collection = ensure_collection("Edges")

    particle_system_objs = [obj for obj in D.objects if obj.name in PARTICLE_SYSTEM_OBJECTS]
    assert len(particle_system_objs) == len(PARTICLE_SYSTEM_OBJECTS)

    deps_graph = C.evaluated_depsgraph_get()

    particle_systems = []
    for obj in particle_system_objs:
        if not obj.particle_systems:
            print(f"Object {obj.name} has no particle systems. Aborting.")
            return

        particle_system = obj.evaluated_get(deps_graph).particle_systems[0]
        particle_systems.append(particle_system)
        print(f"Object: {obj.name}, Particle System: {particle_system.settings.name}")

    # Iterate over all particle systems to create edges
    for num_system, system in enumerate(particle_systems):
        particles = system.particles[:5]

        print(f"Creating edges for particle system: {system.settings.name}")

        for i, p1 in enumerate(particles):
            for j, p2 in enumerate(particles[i + 1 :]):
                print(f"Creating edge: {i} -- {j}")
                edge_name = f"particle_system{num_system + 1} Edge_{i}_{j}"

                curve_data = D.curves.new(name=f"{edge_name}-curve", type="CURVE")
                curve_data.dimensions = "3D"
                spline = curve_data.splines.new(type="POLY")
                spline.points.add(1)
                spline.points[0].co = (*p1.location, 1.0)
                spline.points[1].co = (*p2.location, 1.0)

                curve_obj = D.objects.new(name=edge_name, object_data=curve_data)
                edges_collection.objects.link(curve_obj)

                # Add drivers to dynamically update the edge positions
                # for k, particle in enumerate([p1, p2]):
                #     for axis in range(3):  # x, y, z axes
                #         driver = spline.points[k].driver_add("co", axis).driver
                #         driver.type = "SCRIPTED"
                #         driver.expression = f'particle_systems["{psys.name}"].particles[{particle.index}].location[{axis}]'
                #         var = driver.variables.new()
                #         var.name = "particle_systems"
                #         var.targets[0].id = system.settings

    print("Edges created and constrained to particle locations successfully.")


main()
