import bpy
from .utils import get_scene_objs
from .py_props import my_dict, my_list


class Test_Ops(bpy.types.Operator):
    bl_label = '物体移动按钮'
    bl_description = "这是一个可以移动物体的按钮"
    bl_idname = 'a_test.test_ops'

    # 可以让操作执行后左下角显示一个提示

    bl_options = {"REGISTER", "UNDO"}
    move_axis: bpy.props.EnumProperty(items=(("X", "X", "沿着X轴移动", 0),
                                             ("Y", "Y", "沿着Y轴移动", 1),
                                             ("Z", "Z", "沿着Z轴移动", 2)))
    move_step: bpy.props.FloatProperty(default=10.0, name="移动距离")

    def func(self):
        for o in bpy.context.selected_objects:
            if self.move_axis == "X":
                o.location.x += self.move_step
            elif self.move_axis == "Y":
                o.location.y += self.move_step
            else:
                o.location.z += self.move_step

    def execute(self, context):
        print("Int ops--->func:")
        self.func()
        print("Utils--->get_scene_objs:")
        get_scene_objs()
        print("Py-Props:")
        print(my_list)
        print(my_dict)
        return {"FINISHED"}
