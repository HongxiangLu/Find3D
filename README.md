# 项目介绍

本项目克隆自克隆自[`Find3D`](https://ziqi-ma.github.io/find3dsite/)，主要使用[Data Engine](dataengine)的部分构建新的点云-文本多模态数据集。

# 修改内容

由于路径问题，运行脚本前需要先设置路径：`export PYTHONPATH=$PYTHONPATH:.`

## 新增点云预处理工具

我们所用的点云为没有网格的PLY文件。[`convert_pcd_to_mesh_color.py`](convert_pcd_to_mesh_color.py)将原始的 PLY 点云文件批量转换为带有三角面片的 PLY 网格文件（Triangular Mesh），以适配后续的 PyTorch3D 渲染流程。

## 渲染脚本适配（[`render_2d.py`](dataengine/rendering/render_2d.py)）

1. **PLY 文件加载**: 原先脚本只支持处理 GLB 文件。为此，我们新增了 `ply_to_py3d` 函数。脚本自动根据文件后缀（`.ply` 或 `.glb`）选择对应的加载函数。

2. **渲染参数调整**：在[`meshutils.py`](dataengine/utils/meshutils.py)中修改缩放系数（`mesh.scale_verts_`），这与物体在图片中的占比正相关。另外，[`render_2d.py`](dataengine/rendering/render_2d.py)中`get_rasterizer`函数的第一个参数可以确定生成图片的尺寸。

3. **工具库修正**：修正了[`meshutils.py`](dataengine/utils/meshutils.py)的模块导入错误路径（`from rendering.utils`）。