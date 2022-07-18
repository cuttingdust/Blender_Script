import random
import bpy
from math import cos
from math import sin
bl_info = {
    "name": "Object Random",
    "description": "Object Random functions",
    "author": "CuttingDust",
    "version": (0, 0, 1),
    "blender": (2, 90, 0),
    "location": "View3D",
    "warning": "This addon is still in development.插件还在开发中.",
    "wiki_url": "",
    "category": "Object"}


class Prop_eg1(bpy.types.PropertyGroup):
    axis_x: bpy.props.BoolProperty(name='X', default=False)
    axis_y: bpy.props.BoolProperty(name='Y', default=False)
    axis_z: bpy.props.BoolProperty(name='Z', default=False)
    axis_min: bpy.props.FloatProperty(name='Min', default=0.0)
    axis_max: bpy.props.FloatProperty(name='Max', default=1.0)
    func_type: bpy.props.EnumProperty(items=[('sin', '正弦', "", 0),
                                             ('cos', '余弦', "", 1),
                                             ('power', '平方', "", 2)], name='函数')


class Ops1_eg1(bpy.types.Operator):
    bl_idname = "eg.ops1"
    bl_label = "测试操作符1"
    char_list = [str(i) for i in range(10)] + ['a', 'b', 'c', 'd', 'e', 'f']

    rand_type: bpy.props.StringProperty(default="axis")

    def random_location(self):
        axis_min = bpy.context.scene.eg1.axis_min
        axis_max = bpy.context.scene.eg1.axis_max
        for o in bpy.context.scene.objects:
            if bpy.context.scene.eg1.axis_x:
                o.location.x = random.uniform(axis_min, axis_max)
            if bpy.context.scene.eg1.axis_y:
                o.location.y = random.uniform(axis_min, axis_max)
            if bpy.context.scene.eg1.axis_z:
                o.location.z = random.uniform(axis_min, axis_max)

    def random_name(self):
        for o in bpy.context.scene.objects:
            o.name = "".join(random.choices(self.char_list, k=10))
        return

    def execute(self, context):
        if self.rand_type == "axis":
            self.random_location()
        else:
            self.random_name()
        return {"FINISHED"}


class Ops2_eg1(bpy.types.Operator):
    bl_idname = "eg.ops2"
    bl_label = "测试操作符2"

    def execute(self, context):
        for o in bpy.context.scene.objects:
            if bpy.context.scene.eg1.func_type == 'sin':
                o.location.y = sin(o.location.x)
            if bpy.context.scene.eg1.func_type == 'cos':
                o.location.y = cos(o.location.x)
            if bpy.context.scene.eg1.func_type == 'power':
                o.location.y = o.location.x ** 2
        return {"FINISHED"}


class Panel_eg1(bpy.types.Panel):
    bl_idname = "VIEW_PT_EG1"
    bl_label = "Addon eg1 Object random"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ADD-EG"

    def draw(self, context):
        layout = self.layout
        # 随机物体位置
        row = layout.row(align=True)
        row.prop(bpy.context.scene.eg1, "axis_x", toggle=True)
        row.prop(bpy.context.scene.eg1, "axis_y", toggle=True)
        row.prop(bpy.context.scene.eg1, "axis_z", toggle=True)
        row = layout.row(align=True)
        row.prop(bpy.context.scene.eg1, "axis_min", toggle=True)
        row.prop(bpy.context.scene.eg1, "axis_max", toggle=True)
        layout.operator(Ops1_eg1.bl_idname, text="随机物体位置").rand_type = "axis"

        # 随机物体名称
        layout.operator(Ops1_eg1.bl_idname, text="随机物体名称").rand_type = "name"

        # 函数控制物体属体
        layout.prop(bpy.context.scene.eg1, "func_type")
        layout.operator(Ops2_eg1.bl_idname, text="执行")


def register():
    bpy.utils.register_class(Panel_eg1)
    bpy.utils.register_class(Ops1_eg1)
    bpy.utils.register_class(Prop_eg1)
    bpy.types.Scene.eg1 = bpy.props.PointerProperty(type=Prop_eg1)
    bpy.utils.register_class(Ops2_eg1)


def unregister():
    bpy.utils.unregister_class(Panel_eg1)
    bpy.utils.unregister_class(Ops1_eg1)
    bpy.utils.unregister_class(Prop_eg1)
    del bpy.types.Scene.eg1
    bpy.utils.unregister_class(Ops2_eg1)
