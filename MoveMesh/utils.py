import bpy


def get_scene_objs():
    print("场景所有物体：")
    for o in bpy.context.scene.objects:
        print(o.name)
