bl_info = {
    "name": "插件名称 上天下地插件",
    "author": "插件作者名称 会飞的键盘侠",
    "version": (1, 0),  # 插件版本
    "blender": (2, 83, 0),  # 支持blender版本
    "location": "关于插件位于哪个面板的描述 3D视窗+N面板",
    "description": "插件的描述信息 这个插件可以上天入地",
    "warning": "插件的警告信息 上天入地插件未写完 尚处于测试阶段！",
    "doc_url": "插件的网页链接",
    "tracker_url": "bug 汇报",
    "category": "插件的分类 Object",
}
import bpy
from .ops import Test_Ops
from .ui import Test_Panel


class Test_AddonPref(bpy.types.AddonPreferences):
    bl_idname = __package__
    root: bpy.props.StringProperty(name="Asset root directory",
                                   default="C:/tmp/new_assets",
                                   description="Path to Root Asset Directory",
                                   subtype="DIR_PATH"
                                   )

    def draw(self, context):
        self.layout.row().prop(self, "root", text="预设的根目录")


def register():
    print("上天入地插件 注册~")
    bpy.utils.register_class(Test_Ops)
    bpy.utils.register_class(Test_Panel)
    bpy.utils.register_class(Test_AddonPref)


def unregister():
    print("上天入地插件 卸载~")
    bpy.utils.unregister_class(Test_AddonPref)
    bpy.utils.unregister_class(Test_Panel)
    bpy.utils.unregister_class(Test_Ops)
