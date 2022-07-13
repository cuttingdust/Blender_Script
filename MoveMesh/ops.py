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
                                             ("Z", "Z", "沿着Z轴移动", 2)), 
                                      name= "移动轴向")
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
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'move_axis')
        print("DRAW")
        
class Test_Ops2(bpy.types.Operator):
    bl_label = '测试执行顺序'
    bl_idname = 'a_test.test_ops2'
    bl_description = "这是一个可以测试event的按钮"
    bl_options = {"REGISTER", "UNDO"}

    def __init__(self) -> None:
        print("Init---")

    def __del__(self):
        print("Del---")

    @classmethod
    def poll(cls, context):
        # 判断 场景必须有激活物体
        return context.object
        # 判断场景激活物体 必须为网格
        if context.object:
            return context.object.type == "MESH"
        return False

    def invoke(self, context, event):
        print("INVOKE")
        # return {"FINISHED"}
        # 如果是普通操作符
        # return self.execute(context)
        # 如果是模态操作符
        self.objname = ""
        if bpy.context.object:
            self.objname = bpy.context.object.name
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        print("EXE")
        # 提示 信息
        self.report({"INFO"}, "Info message!")
        # 警告 信息
        self.report({"WARNING"}, "Warning message!")
        # 错误 信息
        self.report({"ERROR"}, "Error message!")
        return {"FINISHED"}

    def draw(self, context):
        print("DRAW")
        self.layout.label(text="弹出面板")

    def modal(self, context, event):
        if event.type != "MOUSEMOVE":
            print(f"按键:{event.type},状态:{event.value}")
        if event.type == "ESC":
            return {"FINISHED"}
        # 当鼠标选中新物体之后 将名称打印到控制台
        # FINISHED CANCELLED PASS_THROUGH INTERFACE RUNNING_MODAL

        return {"RUNNING_MODAL"}

