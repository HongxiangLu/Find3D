# 项目介绍

本项目克隆自克隆自[`Find3D`](https://ziqi-ma.github.io/find3dsite/)，主要使用[Data Engine](dataengine)的部分构建新的点云-文本多模态数据集。

# 修改内容

## 新增点云预处理工具

我们所用的点云为没有网格的PLY文件。[`convert_pcd_to_mesh_color.py`](convert_pcd_to_mesh_color.py)将原始的 PLY 点云文件批量转换为带有三角面片的 PLY 网格文件（Triangular Mesh），以适配后续的 PyTorch3D 渲染流程。

## 渲染脚本适配（[`render_2d.py`](dataengine/rendering/render_2d.py)）

1. **PLY 文件加载**: 新增 `ply_to_py3d` 函数，使用 `trimesh` 加载 PLY 文件，并处理顶点颜色（Vertex Colors）为 `TexturesVertex` 格式，以兼容 PyTorch3D 渲染器。

2. **渲染参数调整**：在[`meshutils.py`](dataengine/utils/meshutils.py)中修改缩放比例（`mesh.scale_verts_`），以减少渲染图像中的空白区域，使物体在画面中占比更合理。

3. **输入兼容性**: 修改了 `render_k_views` 主循环，自动根据文件后缀（`.ply` 或 `.glb`）选择对应的加载函数。

4. **工具库修正**：修正了[`meshutils.py`](dataengine/utils/meshutils.py)的模块导入错误路径（`from rendering.utils`）。