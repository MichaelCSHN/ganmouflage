import sys
import numpy as np
import open3d as o3d
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QDoubleSpinBox,
    QGroupBox, QSlider, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QMutex, QWaitCondition
import time
import os

# 全局变量用于存储点云和OBJ网格
global_point_cloud = None
global_obj_mesh = None
global_obj_original_mesh = None # 用于存储OBJ网格的原始状态，以便重置变换

# 用于线程间同步的锁和条件变量
viewer_mutex = QMutex()
viewer_ready_condition = QWaitCondition()

# --- Open3D 查看器线程 ---
class Open3DViewerThread(QThread):
    """
    一个独立的线程来处理Open3D可视化。
    使用Open3D.Visualizer实现持久化显示和动态更新。
    """
    # 信号：通知GUI线程Open3D查看器已初始化并准备就绪
    viewer_initialized = pyqtSignal()
    # 信号：通知GUI线程更新几何体
    update_geometries_signal = pyqtSignal(list, object) # ([PCL, OBJ], type_of_update)
    # 信号：通知GUI线程某个特定几何体已更新
    update_single_geometry_signal = pyqtSignal(object) # geometry_object

    def __init__(self):
        super().__init__()
        self.visualizer = None
        self.geometries_in_viewer = [] # 存储当前在viewer中的几何体，用于管理
        self.running = True

    def run(self):
        """
        运行Open3D可视化循环。
        """
        # 修正：QMutex 不支持 'with' 语句，需要显式 lock/unlock
        viewer_mutex.lock()
        try:
            self.visualizer = o3d.visualization.Visualizer()
            self.visualizer.create_window(window_name="点云与OBJ模型", width=1024, height=768, left=50, top=50)
            self.visualizer.get_render_option().mesh_show_back_face = True # 显示网格两面

            # 将信号连接到线程内的槽函数
            self.update_geometries_signal.connect(self._handle_update_geometries)
            self.update_single_geometry_signal.connect(self._handle_update_single_geometry)

            self.viewer_initialized.emit() # 通知主线程查看器已准备就绪
            viewer_ready_condition.wakeAll() # 唤醒等待的线程
        finally:
            viewer_mutex.unlock()

        # Open3D的事件循环，保持窗口响应
        while self.running and self.visualizer.poll_events():
            self.visualizer.update_renderer()
            # 短暂休眠以避免100% CPU占用
            time.sleep(0.01) # 10ms

        self.visualizer.destroy_window()
        self.visualizer = None
        self.running = False
        print("Open3D查看器线程已退出。")

    def stop(self):
        """
        请求线程优雅地停止。
        """
        self.running = False

    def _handle_update_geometries(self, geometries_list, type_of_update):
        """
        槽函数：处理更新所有几何体的请求。
        """
        if self.visualizer is None:
            return

        # 清除现有几何体
        for geom in self.geometries_in_viewer:
            self.visualizer.remove_geometry(geom)
        self.geometries_in_viewer.clear()

        # 添加新几何体
        for geom in geometries_list:
            if geom: # 确保几何体存在
                self.visualizer.add_geometry(geom)
                self.geometries_in_viewer.append(geom)
        
        # 视口自动调整到新几何体
        if geometries_list and type_of_update == "initial":
            self.visualizer.reset_view_point(True) # 首次加载或完全刷新时重置视角

        self.visualizer.update_renderer()

    def _handle_update_single_geometry(self, geometry_object):
        """
        槽函数：处理更新单个几何体的请求（例如OBJ变换）。
        """
        if self.visualizer is None or geometry_object is None:
            return
        
        # 查找并更新已存在的几何体
        # 注意：Open3D的update_geometry需要你传入的几何体是之前add_geometry的那个实例
        # 否则它会创建一个新的。这里我们依赖于global_obj_mesh的引用被更新。
        # 最稳妥的方法是 remove_geometry 然后 add_geometry，但这样会导致闪烁。
        # 对于变换，如果只是改变内部数据，update_geometry_data() 是更好的。
        # 幸运的是，transform_mesh 会修改几何体内部的顶点数据，
        # 所以我们可以直接调用 update_geometry。
        
        # Open3D 0.10+ 版本：update_geometry() 方法用于更新已经添加到Visualizer中的几何体。
        # 它要求传入的是与之前add_geometry时同一个对象实例。
        # 在我们的代码中，global_obj_mesh 在变换后被替换为新的TriangleMesh实例，
        # 这意味着旧的实例不在Visualizer中。
        # 解决方案：移除旧的，添加新的。或者，如果变换直接修改了顶点数据，可以尝试 update_geometry。
        
        # 鉴于我们的变换是创建一个新网格，然后替换全局变量，
        # 稳妥的更新方式是移除旧的，添加新的。
        # 为了避免重新遍历所有几何体并重新添加点云（如果点云没变），
        # 我们只更新OBJ。

        # 尝试更新OBJ，需要找到它在 self.geometries_in_viewer 中的引用
        found_and_updated = False
        for i, geom in enumerate(self.geometries_in_viewer):
            # 简单判断是否是OBJ（可以通过类型或ID，这里简单判断是否是TriangleMesh）
            if isinstance(geom, o3d.geometry.TriangleMesh):
                # 移除旧的OBJ实例
                self.visualizer.remove_geometry(geom)
                # 添加新的（已变换的）OBJ实例
                self.visualizer.add_geometry(geometry_object)
                self.geometries_in_viewer[i] = geometry_object # 更新列表中的引用
                found_and_updated = True
                break
        
        if not found_and_updated:
            # 如果OBJ不在列表中（可能从未添加过），则直接添加
            self.visualizer.add_geometry(geometry_object)
            self.geometries_in_viewer.append(geometry_object)


        self.visualizer.update_renderer()


# --- 主GUI类 ---
class PointCloudViewerApp(QWidget):
    """
    用于显示和变换3D模型的主应用程序窗口。
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("点云与OBJ模型查看器")
        self.viewer_thread = Open3DViewerThread() # 预先创建线程实例
        self.viewer_thread.viewer_initialized.connect(self.on_viewer_initialized)
        
        self.init_ui()

        # 初始化变换值
        self.current_scale = 1.0
        self.current_translation = np.array([0.0, 0.0, 0.0])
        self.current_rotation_angles_deg = np.array([0.0, 0.0, 0.0]) # 欧拉角 (X, Y, Z) 单位为度

        self.is_viewer_ready = False # 标志：Open3D查看器是否已完全初始化并准备接受指令

        self.update_controls_enabled_state() # 设置控件的初始状态

        # 启动Open3D查看器线程
        self.viewer_thread.start()
        
        # 等待Open3D查看器线程初始化完成
        # 注意：这里使用QWaitCondition会阻塞GUI初始化，可以考虑在启动后立即返回，
        # 并在 on_viewer_initialized 中启用按钮。
        # 为了简化，此处先同步等待。
        viewer_mutex.lock() # 修正：显式锁定
        try:
            if not self.is_viewer_ready:
                viewer_ready_condition.wait(viewer_mutex)
        finally:
            viewer_mutex.unlock() # 修正：显式解锁
        print("GUI: Open3D查看器已准备就绪。")


    def init_ui(self):
        """
        初始化用户界面元素。
        """
        main_layout = QHBoxLayout()
        controls_layout = QVBoxLayout() # 左侧用于控件

        # --- 文件加载组 ---
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()

        self.load_bundle_btn = QPushButton("加载点云 (bundle.out)")
        self.load_bundle_btn.clicked.connect(self.load_bundle_file)
        file_layout.addWidget(self.load_bundle_btn)

        self.load_obj_btn = QPushButton("加载OBJ模型 (.obj)")
        self.load_obj_btn.clicked.connect(self.load_obj_file)
        file_layout.addWidget(self.load_obj_btn)

        self.display_models_btn = QPushButton("显示/更新 3D 模型")
        self.display_models_btn.clicked.connect(self.display_current_models)
        file_layout.addWidget(self.display_models_btn)

        self.export_obj_btn = QPushButton("导出OBJ模型 (.obj)")
        self.export_obj_btn.clicked.connect(self.export_obj_file)
        self.export_obj_btn.setEnabled(False) # 在加载OBJ之前禁用
        file_layout.addWidget(self.export_obj_btn)

        file_group.setLayout(file_layout)
        controls_layout.addWidget(file_group)

        # --- 变换控件组 ---
        transform_group = QGroupBox("OBJ模型变换")
        transform_layout = QVBoxLayout()

        # 缩放控制
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("缩放:"))
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setRange(0.01, 100.0) # 范围从0.01倍到100倍
        self.scale_spinbox.setSingleStep(0.1)
        self.scale_spinbox.setValue(1.0)
        self.scale_spinbox.valueChanged.connect(self.update_transformation)
        scale_layout.addWidget(self.scale_spinbox)
        transform_layout.addLayout(scale_layout)

        # 平移控制 (X, Y, Z)
        self.translation_x_layout, self.translation_x_spinbox = self._create_spinbox("平移 X:", -100.0, 100.0, 0.1, 0.0)
        self.translation_y_layout, self.translation_y_spinbox = self._create_spinbox("平移 Y:", -100.0, 100.0, 0.1, 0.0)
        self.translation_z_layout, self.translation_z_spinbox = self._create_spinbox("平移 Z:", -100.0, 100.0, 0.1, 0.0)
        transform_layout.addLayout(self.translation_x_layout)
        transform_layout.addLayout(self.translation_y_layout)
        transform_layout.addLayout(self.translation_z_layout)

        # 旋转控制 (X, Y, Z 使用滑块+SpinBox)
        self.rotation_x_layout, self.rotation_x_slider, self.rotation_x_spinbox, self.rotation_x_label = self._create_angle_control("旋转 X (俯仰):")
        self.rotation_y_layout, self.rotation_y_slider, self.rotation_y_spinbox, self.rotation_y_label = self._create_angle_control("旋转 Y (偏航):")
        self.rotation_z_layout, self.rotation_z_slider, self.rotation_z_spinbox, self.rotation_z_label = self._create_angle_control("旋转 Z (翻滚):")
        transform_layout.addLayout(self.rotation_x_layout)
        transform_layout.addLayout(self.rotation_y_layout)
        transform_layout.addLayout(self.rotation_z_layout)

        # 重置变换按钮
        self.reset_transform_btn = QPushButton("重置OBJ变换")
        self.reset_transform_btn.clicked.connect(self.reset_obj_transformation)
        self.reset_transform_btn.setEnabled(False) # 在加载OBJ之前禁用
        transform_layout.addWidget(self.reset_transform_btn)

        transform_group.setLayout(transform_layout)
        controls_layout.addWidget(transform_group)

        controls_layout.addStretch(1) # 将所有控件推到顶部

        # 组装主布局
        main_layout.addLayout(controls_layout)
        self.setLayout(main_layout)

    def on_viewer_initialized(self):
        """
        Open3D查看器线程初始化完成时调用的槽函数。
        """
        self.is_viewer_ready = True
        # 可以在这里启用某些只在查看器准备好后才能操作的GUI控件
        # 例如：self.display_models_btn.setEnabled(True)
        print("GUI: Open3D查看器初始化信号接收。")


    def _create_spinbox(self, label_text, min_val, max_val, step, default_val):
        """辅助函数，用于创建QLabel和QDoubleSpinBox对。"""
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel(label_text))
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        spinbox.setValue(default_val)
        spinbox.valueChanged.connect(self.update_transformation)
        h_layout.addWidget(spinbox)
        return h_layout, spinbox

    def _create_angle_control(self, label_text):
        """创建带滑块和SpinBox的角度控制，二者联动。"""
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(-180, 180)
        slider.setValue(0)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        slider.setTickInterval(30)
        slider.setTickPosition(QSlider.TicksBelow)
        spinbox = QDoubleSpinBox()
        spinbox.setRange(-180, 180)
        spinbox.setSingleStep(1)
        spinbox.setValue(0)
        angle_display = QLabel("0°")
        # 联动
        slider.valueChanged.connect(lambda v: spinbox.setValue(v))
        spinbox.valueChanged.connect(lambda v: slider.setValue(int(v)))
        slider.valueChanged.connect(lambda value, display=angle_display: display.setText(f"{value}°"))
        spinbox.valueChanged.connect(lambda value, display=angle_display: display.setText(f"{int(value)}°"))
        # 变换
        slider.valueChanged.connect(self.update_transformation)
        spinbox.valueChanged.connect(self.update_transformation)
        h_layout.addWidget(slider)
        h_layout.addWidget(spinbox)
        h_layout.addWidget(angle_display)
        return h_layout, slider, spinbox, angle_display

    def update_controls_enabled_state(self):
        """
        根据当前是否加载了OBJ模型来启用/禁用变换和导出控件。
        """
        obj_loaded = global_obj_mesh is not None
        self.export_obj_btn.setEnabled(obj_loaded)
        self.reset_transform_btn.setEnabled(obj_loaded)
        self.scale_spinbox.setEnabled(obj_loaded)
        self.translation_x_spinbox.setEnabled(obj_loaded)
        self.translation_y_spinbox.setEnabled(obj_loaded)
        self.translation_z_spinbox.setEnabled(obj_loaded)
        self.rotation_x_slider.setEnabled(obj_loaded)
        self.rotation_x_spinbox.setEnabled(obj_loaded)
        self.rotation_y_slider.setEnabled(obj_loaded)
        self.rotation_y_spinbox.setEnabled(obj_loaded)
        self.rotation_z_slider.setEnabled(obj_loaded)
        self.rotation_z_spinbox.setEnabled(obj_loaded)


    # --- 文件加载回调 ---
    def load_bundle_file(self):
        """
        打开文件对话框以选择并加载bundle.out点云文件。
        """
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "选择bundle.out文件", "", "Bundle Files (*.out);;所有文件 (*)", options=options)
        if filepath:
            print(f"加载bundle.out文件: {filepath}")
            try:
                point_cloud_data = self._read_bundle_out(filepath)
                global global_point_cloud
                global_point_cloud = point_cloud_data
                QMessageBox.information(self, "加载成功", "点云模型已加载。请点击 '显示/更新 3D 模型' 查看。")
                self.display_current_models(update_view_point=True) # 触发更新
            except Exception as e:
                QMessageBox.critical(self, "加载错误", f"读取bundle.out文件时出错: {e}")
                print(f"读取bundle.out文件时出错: {e}")

    def load_obj_file(self):
        """
        打开文件对话框以选择并加载OBJ 3D模型文件。
        """
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "选择OBJ模型文件", "", "OBJ Files (*.obj);;所有文件 (*)", options=options)
        if filepath:
            print(f"加载OBJ模型文件: {filepath}")
            try:
                mesh = o3d.io.read_triangle_mesh(filepath)
                if not mesh.has_vertices() or not mesh.has_triangles():
                    raise ValueError("OBJ文件未包含有效的顶点或面。")
                mesh.compute_vertex_normals() # 确保计算法线以正确渲染光照
                
                # 检查OBJ文件中是否存在现有材质
                if not mesh.has_vertex_colors() and not mesh.has_triangle_materials():
                    # 如果没有颜色/材质，则分配一个默认的灰色
                    mesh.paint_uniform_color([0.7, 0.7, 0.7]) # 中性灰色

                global global_obj_mesh
                global global_obj_original_mesh
                global_obj_mesh = mesh
                global_obj_original_mesh = o3d.geometry.TriangleMesh(mesh) # 存储一个副本以供重置
                self.reset_transformation_values() # 重置GUI控件为默认值
                self.update_controls_enabled_state() # 启用变换控件
                QMessageBox.information(self, "加载成功", "OBJ模型已加载。请点击 '显示/更新 3D 模型' 查看或直接进行变换。")
                self.display_current_models(update_view_point=True) # 触发更新
            except Exception as e:
                QMessageBox.critical(self, "加载错误", f"读取OBJ文件时出错: {e}")
                print(f"读取OBJ文件时出错: {e}")

    def export_obj_file(self):
        """
        导出当前OBJ模型到文件。
        同时生成对应的.npy文件。
        """
        if global_obj_mesh is None:
            QMessageBox.warning(self, "警告", "没有可导出的OBJ模型。")
            return

        # 获取保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存OBJ文件", "", "OBJ Files (*.obj)"
        )
        if not file_path:
            return

        try:
            # 确保文件扩展名为.obj
            if not file_path.lower().endswith('.obj'):
                file_path += '.obj'

            # 获取顶点和法线
            vertices = np.asarray(global_obj_mesh.vertices)
            normals = np.asarray(global_obj_mesh.vertex_normals)
            triangles = np.asarray(global_obj_mesh.triangles)

            # 写入OBJ文件
            with open(file_path, 'w') as f:
                # 写入文件头
                f.write("####\n")
                f.write("#\n")
                f.write("# OBJ File Generated by GANmouflage\n")
                f.write("#\n")
                f.write("####\n")
                f.write(f"# Object {os.path.basename(file_path)}\n")
                f.write("#\n")
                f.write(f"# Vertices: {len(vertices)}\n")
                f.write(f"# Faces: {len(triangles)}\n")
                f.write("#\n")
                f.write("####\n")

                # 写入法线和顶点
                for i in range(len(vertices)):
                    # 写入法线
                    f.write(f"vn {normals[i][0]:.6f} {normals[i][1]:.6f} {normals[i][2]:.6f}\n")
                    # 写入顶点（包含颜色信息）
                    f.write(f"v {vertices[i][0]:.6f} {vertices[i][1]:.6f} {vertices[i][2]:.6f} 1.000000 1.000000 1.000000\n")

                # 写入面信息（注意：OBJ文件中的索引是从1开始的）
                for triangle in triangles:
                    f.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")

            # 生成对应的.npy文件
            npy_path = os.path.splitext(file_path)[0] + '.npy'
            
            # 随机采样2048个点
            if len(vertices) > 2048:
                indices = np.random.choice(len(vertices), 2048, replace=False)
                sampled_vertices = vertices[indices]
                sampled_normals = normals[indices]
            else:
                # 如果点数不足，重复采样
                indices = np.random.choice(len(vertices), 2048, replace=True)
                sampled_vertices = vertices[indices]
                sampled_normals = normals[indices]

            # 合并点和法线
            combined = np.concatenate([sampled_vertices, sampled_normals], axis=1)
            
            # 保存为npy文件
            np.save(npy_path, combined.astype(np.float64))

            QMessageBox.information(self, "成功", f"OBJ模型已保存到：\n{file_path}\n\n对应的NPY文件已保存到：\n{npy_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存文件时出错：\n{str(e)}")

    # --- 变换回调 ---
    def update_transformation(self):
        """
        将GUI中当前的缩放、平移和旋转值应用于OBJ模型，并更新Open3D查看器。
        """
        if global_obj_original_mesh is None:
            QMessageBox.warning(self, "操作失败", "请先加载OBJ模型才能进行变换。")
            return

        if not self.is_viewer_ready:
            QMessageBox.warning(self, "Open3D未准备就绪", "Open3D查看器尚未完全启动。请稍候或点击 '显示/更新 3D 模型' 按钮。")
            return

        # 从GUI控件获取值
        self.current_scale = self.scale_spinbox.value()
        self.current_translation = np.array([
            self.translation_x_spinbox.value(),
            self.translation_y_spinbox.value(),
            self.translation_z_spinbox.value()
        ])
        # 旋转角度从SpinBox获取
        self.current_rotation_angles_deg = np.array([
            self.rotation_x_spinbox.value(),
            self.rotation_y_spinbox.value(),
            self.rotation_z_spinbox.value()
        ])
        
        # 从原始网格创建一个新的副本，以便独立地应用变换
        transformed_mesh = o3d.geometry.TriangleMesh(global_obj_original_mesh)

        # 按顺序应用变换: 缩放，旋转，平移
        transformed_mesh.scale(self.current_scale, center=transformed_mesh.get_center())
        rotation_angles_rad = np.deg2rad(self.current_rotation_angles_deg)
        R = transformed_mesh.get_rotation_matrix_from_xyz(rotation_angles_rad)
        transformed_mesh.rotate(R, center=transformed_mesh.get_center()) # 围绕其自身中心旋转
        transformed_mesh.translate(self.current_translation)

        # 更新全局网格实例
        global global_obj_mesh
        global_obj_mesh = transformed_mesh
        
        # 通知查看器线程更新OBJ模型
        self.viewer_thread.update_single_geometry_signal.emit(global_obj_mesh)
        print("OBJ模型已变换并更新视图。")

    def reset_obj_transformation(self):
        """
        将OBJ模型重置为其原始加载状态，并重置GUI控件。
        """
        if global_obj_original_mesh:
            global global_obj_mesh
            global_obj_mesh = o3d.geometry.TriangleMesh(global_obj_original_mesh) # 重置为原始状态
            self.reset_transformation_values() # 重置GUI控件为默认值
            self.update_transformation() # 应用（现在是单位）变换
            print("OBJ模型变换已重置。")
            QMessageBox.information(self, "重置成功", "OBJ模型变换已重置到初始状态。")
        else:
            QMessageBox.warning(self, "重置失败", "没有加载OBJ模型可供重置。")
            print("没有加载OBJ模型可供重置。")

    def reset_transformation_values(self):
        """
        将所有变换GUI控件（微调框、滑块、SpinBox）重置为其默认值。
        """
        self.scale_spinbox.setValue(1.0)
        self.translation_x_spinbox.setValue(0.0)
        self.translation_y_spinbox.setValue(0.0)
        self.translation_z_spinbox.setValue(0.0)
        self.rotation_x_slider.setValue(0)
        self.rotation_x_spinbox.setValue(0)
        self.rotation_y_slider.setValue(0)
        self.rotation_y_spinbox.setValue(0)
        self.rotation_z_slider.setValue(0)
        self.rotation_z_spinbox.setValue(0)
        self.rotation_x_label.setText("0°")
        self.rotation_y_label.setText("0°")
        self.rotation_z_label.setText("0°")


    def display_current_models(self, update_view_point=False):
        """
        通知Open3D查看器线程显示或更新当前的全局模型。
        """
        if not self.is_viewer_ready:
            QMessageBox.warning(self, "Open3D未准备就绪", "Open3D查看器尚未完全启动。请稍候。")
            return

        geometries_to_display = []
        if global_point_cloud:
            geometries_to_display.append(global_point_cloud)
        if global_obj_mesh:
            geometries_to_display.append(global_obj_mesh)

        if not geometries_to_display:
            QMessageBox.information(self, "信息", "没有可显示的点云或OBJ模型。请先加载文件。")
            print("没有几何体可以显示。")
            return
        
        # 通过信号通知查看器线程更新几何体
        if update_view_point:
            self.viewer_thread.update_geometries_signal.emit(geometries_to_display, "initial")
        else:
            self.viewer_thread.update_geometries_signal.emit(geometries_to_display, "update")

        print("GUI: 已请求Open3D查看器更新模型。")


    # --- bundle.out 解析辅助函数 ---
    def _read_bundle_out(self, filepath):
        """
        读取bundle.out文件并将其解析为Open3D PointCloud对象。
        格式与data/utils.py中的read_bundler函数保持一致。
        """
        points = []
        colors = []
        try:
            with open(filepath, 'r') as f:
                # 跳过第一行（可选头部）
                f.readline()
                # 读取相机数量和点数量
                ncams = int(f.readline().split()[0])
                
                # 读取相机参数
                focals = []
                Rt = []
                for i in range(ncams):
                    focals.append(float(f.readline().split()[0]))
                    R = np.array([list(map(float, f.readline().split())) for x in range(3)])
                    t = np.array(list(map(float, f.readline().split())))
                    Rt.append((R, t))

                # 读取3D点数据
                while True:
                    line = f.readline().strip()
                    if not line:  # 空行表示文件结束
                        break
                    
                    # 点: X Y Z
                    point = list(map(float, line.split()))
                    if len(point) != 3:
                        continue  # 跳过无效的点数据
                    points.append(point)

                    # 颜色: R G B (预期为0-255，为Open3D标准化到0-1)
                    color_line = f.readline().strip()
                    if not color_line:  # 如果没有颜色数据，使用默认颜色
                        colors.append([0.5, 0.5, 0.5])  # 默认灰色
                        continue
                        
                    try:
                        color = list(map(float, color_line.split()))
                        if len(color) != 3:
                            colors.append([0.5, 0.5, 0.5])  # 无效颜色数据时使用默认灰色
                        else:
                            # 确保颜色值在0-1范围内
                            color = [max(0.0, min(1.0, c/255.0)) for c in color]
                            colors.append(color)
                    except ValueError:
                        colors.append([0.5, 0.5, 0.5])  # 颜色解析失败时使用默认灰色
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"文件未找到: {filepath}")
        except ValueError as e:
            raise ValueError(f"bundle.out文件中的数据无法解析为数字: {e}")
        except Exception as e:
            raise Exception(f"读取bundle.out时发生意外错误: {e}")

        if not points:
            raise ValueError("未能从bundle.out文件中读取任何点。请检查文件格式。")

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
        if colors:
            point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
        else:
            print("bundle.out文件中没有颜色信息，将使用默认颜色 (灰色)。")
            point_cloud.colors = o3d.utility.Vector3dVector(np.full((len(points), 3), 0.5, dtype=np.float64))

        return point_cloud


    def closeEvent(self, event):
        """
        重写closeEvent，确保在GUI关闭时停止Open3D线程。
        """
        print("GUI: 应用程序正在关闭，停止Open3D线程...")
        if self.viewer_thread and self.viewer_thread.isRunning():
            self.viewer_thread.stop() # 发送停止请求
            self.viewer_thread.wait(5000) # 等待线程终止，最多5秒
            if self.viewer_thread.isRunning():
                print("GUI: Open3D线程未能优雅退出，强制终止。")
                self.viewer_thread.terminate() # 如果wait超时，强制终止
        event.accept()


# --- 主程序执行 ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    viewer_app = PointCloudViewerApp()
    viewer_app.show()
    
    sys.exit(app.exec_())
