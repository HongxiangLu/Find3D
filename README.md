# 项目介绍

本项目克隆自克隆自[`Find3D`](https://ziqi-ma.github.io/find3dsite/)，主要使用 [Data Engine](dataengine) 的部分构建新的点云-文本多模态数据集。

# 运行前的准备

1. 根据[`dataengine/requirements.txt`](dataengine/requirements.txt)的内容创建conda环境

2. 创建`DATA_ROOT`文件夹，将路径写入[`dataengine/configs.py`](dataengine/configs.py)

3. 准备如下所示的文件结构：

```
DATA_ROOT/
├── labeled/
│   ├── chunk_ids/
│   │   └── chunk0.csv              # [必需] 待处理的 UID 列表
│   └── glbs/                       # [必需] 存放模型文件和路径映射
│       ├── object-paths.json.gz    # [必需] UID 到文件路径的映射
│       └── <你的模型文件>            # [必需] 实际的 .glb 或 .ply 文件
└── obj1lvis/
    └── metadata.csv                # [必需] UID 到类别的映射
```

其中`obj1lvis/metadata.csv`和`labeled/chunk_ids/chunk0.csv`都提供了每个对象的类别（class）信息，如下所示：

```
uid,class
object_uid_1,chair
object_uid_2,table
```

而`labeled/glbs/object-paths.json.gz`是一个压缩的 JSON 文件，记录每个 `uid` 对应的模型文件相对路径（相对于`labeled/glbs/`），如下所示：

```
{
    "object_uid_1": "chair/model_1.glb",
    "object_uid_2": "table/model_2.ply"
}
```

# 修改内容

由于路径问题，运行脚本前需要先设置路径：`export PYTHONPATH=$PYTHONPATH:.`

### 新增点云预处理工具

我们所用的点云为没有网格的PLY文件。[`convert_pcd_to_mesh_color.py`](convert_pcd_to_mesh_color.py)将原始的 PLY 点云文件批量转换为带有三角面片的 PLY 网格文件（Triangular Mesh），以适配后续的 PyTorch3D 渲染流程。

### 渲染脚本适配（[`dataengine/rendering/render_2d.py`](dataengine/rendering/render_2d.py)）

1. **PLY 文件加载**: 原先脚本只支持处理 GLB 文件。为此，我们新增了 `ply_to_py3d` 函数。脚本自动根据文件后缀（`.ply` 或 `.glb`）选择对应的加载函数。

2. **渲染参数调整**：在[`dataengine/utils/meshutils.py`](dataengine/utils/meshutils.py)中修改缩放系数（`mesh.scale_verts_`），这与物体在图片中的占比正相关。另外，[`dataengine/rendering/render_2d.py`](dataengine/rendering/render_2d.py)中`get_rasterizer`函数的第一个参数可以确定生成图片的尺寸。

3. **工具库修正**：修正了[`dataengine/utils/meshutils.py`](dataengine/utils/meshutils.py)的模块导入错误路径（`from rendering.utils`）。

4. **运行结果**：在`DATAROOT/labeled`目录下创建`rendered`文件夹，存储每个点云在不同方向上渲染的图片。

### 物体朝向判断脚本（[`dataengine/llm/query_orientation.py`](dataengine/llm/query_orientation.py)）

1. 模型从Gemini迁移至通义千问系列，调用OpenAI兼容接口（Python openai包）。使用前需要通过环境变量配置API Key：`export DASHSCOPE_API_KEY="sk-你的key"`

2. 移除了原有的轮询逻辑，引入命令行参数支持并行处理。单进程时直接运行脚本，多进程时运行脚本如下所示：

```
# 终端 1
python dataengine/llm/query_orientation.py 0 --total 4
# 终端 2
python dataengine/llm/query_orientation.py 1 --total 4
# ... 以此类推到 3
```

3. **运行结果**：从同一点云的不同方向渲染图中挑出合适的，统一放在`DATA_ROOT/labeled/rendered/点云名称/oriented`文件夹下。

### 掩码生成脚本（[`dataengine/seg2d/get_sam_masks.py`](dataengine/seg2d/get_sam_masks.py)）

1. 修正了掩码的保存逻辑，移除了 Matplotlib 相关代码，改用 OpenCV ，确保生成的掩码可视化图与原始渲染图分辨率完全一致。

2. 修改了原脚本的数据类型兼容性问题。

3. **运行结果**：在`DATA_ROOT/labeled/rendered/点云名称/oriented`目录下建立`masks`文件夹，为`imgs`目录下的每张图片生成掩码。