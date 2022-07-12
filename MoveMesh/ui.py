import bpy
from .ops import Test_Ops


class Test_Panel(bpy.types.Panel):
    # 标签
    bl_label = 'a-test'
    bl_idname = 'A_TEST_PL'
    # 面板所属区域
    bl_space_type = "VIEW_3D"
    # 显示面板的地方
    bl_region_type = "UI"
    # 显示面板的地方的归类
    bl_category = "A-TEST"

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.operator(Test_Ops.bl_idname)
