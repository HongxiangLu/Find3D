# 项目介绍

本项目克隆自克隆自[`Find3D`](https://ziqi-ma.github.io/find3dsite/)，主要使用 [Data Engine](dataengine) 的部分构建新的点云-文本多模态数据集。

# 运行前的准备

1. 从[这里](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)下载 SAM 模型的 Checkpoints，并将其存放路径写入 [`dataengine/configs.py`](dataengine/configs.py) 的 `SAM_CHECKPOINT_PATH`。

2. 根据[`dataengine/requirements.txt`](dataengine/requirements.txt)的内容创建conda环境

3. 创建`DATA_ROOT`文件夹，将路径写入 [`dataengine/configs.py`](dataengine/configs.py) 

4. 准备如下所示的文件结构：

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

### 部件命名脚本（[`name_single_part.py`](dataengine/llm/name_single_part.py)）

1. 模型从Gemini迁移至通义千问系列，调用OpenAI兼容接口（Python openai包）。使用前需要通过环境变量配置API Key：`export DASHSCOPE_API_KEY="sk-你的key"`

2. 移除了原有的轮询逻辑，仅保留单线程运行版本。

3. **运行结果**：使用多模态大模型（Visual LLM）识别上一部 SAM 分割出的部件名称，在 `DATA_ROOT/labeled/rendered/点云名称/oriented/masks` 路径下保存每份点云所有的掩码文件路径及其对应名称，命名为 `partnames.json`。

### 整理掩码脚本（[`dataengine/seg2d/merge_masks.py`](dataengine/seg2d/merge_masks.py)）

1. 修正了一些代码错误。

2. **运行结果**：根据大模型识别出的部件名称（如 "rail", "ground"），将同一视角下具有相同名称的掩码合并，并将所有视角、所有部件的掩码统一打包成结构化的 PyTorch 张量文件。

3. **文件结构**：如下所示。其中：

```
oriented/
├── masks/
│   ├── partnames.json
│   └── merged/
│       ├── allmasks.pt
│       ├── mask_labels.txt
│       └── mask2view.pt
├── imgs/
│   ├── 00.jpg
│   └── ...
└── pix2face.pt
```

`pix2face.pt` 记录了 2D 渲染图上的每一个像素对应 3D 网格模型上的哪一个面片，形状为 (10, H, W)，对应 10 个视角。这份数据是之前生成的，后续步骤会利用它把 2D 掩码「反向投影」回 3D 空间。

`allmasks.pt` 是所有掩码的集合，包含该物体所有视角下所有部件的二值掩码。形状为 (N_MASKS, H, W)，其中第 $i$ 个切片对应第 $i$ 个部件的 2D 形状。

`mask2view.pt` 是掩码所属视角索引，形状为 (N_MASKS,)。如果第 $i$ 个元素是 3，说明 allmasks.pt 中的第 $i$ 个掩码属于第 3 个视角（view03）。

`mask_labels.txt` 是掩码对应的部件名称。，为 N_MASKS 行文本。其中第 $i$ 行文本（如 "rail"）是 allmasks.pt 中第 $i$ 个掩码的标签。它与 mask2view.pt 和 allmasks.pt 是一一对应的。

### 3D表面采样脚本（[dataengine/py3d_customization/sample_points.py](dataengine/py3d_customization/sample_points.py)）

1. 这是个新增脚本，调用了原作者在[dataengine/README.md](dataengine/README.md)中生成修改后的 `sample_points_from_meshes.py` 脚本。

2. **核心功能**：读取 3D 模型（PLY 或 GLB），在模型表面均匀随机地采样指定数量（默认 10000 个）的 3D 点。脚本不仅获取这些点的空间坐标，还记录了每个点具体落在了模型的哪一个三角面片（Face）上。

3. **运行结果**：对于每个处理的物体（UID），脚本会在 DATA_ROOT/labeled/points/{classname}_{uid}/ 目录下生成两个 PyTorch Tensor 文件：

其一是 `points.pt`，记录了采样点的 3D 坐标 (x, y, z)，形状为 (N_SAMPLES, 3)，例如 (10000, 3)。这是点云本身的数据，后续用于训练 Point Cloud 模型或进行对比学习。

其二是 `point2face.pt`，记录了每个采样点所属的网格面索引（Face Index），形状为 (N_SAMPLES,)，例如 (10000,)。这是一个关键的索引映射。它告诉后续程序：“第 5 个采样点位于模型的第 1024 号面片上”。

4. 另外，原作者修改过的 [sample_points_from_meshes.py](dataengine/py3d_customization/sample_points_from_meshes.py) 没有导入必要的库。这里修补了这一点。

5. 这个脚本是为 `label_mask2pt.py` 做准备，将 2D 图片上的分割掩码反向投影到 3D 空间中。

这个脚本产生的 `point2face.pt` 指出「某个 3D 点位于 Face X」，建立了点到面的映射；而之前 `merge_masks.py` 产生的 `pix2face.pt` 指出「某个 2D 像素对应 Face X」，建立了像素到面的映射。

如果 2D 像素和 3D 点都指向同一个 Face X，那么这个 3D 点就应该被标记为该 2D 掩码的类别。

### 掩码点云对应关系脚本（[dataengine/label3d/label_mask2pt.py](dataengine/label3d/label_mask2pt.py)）

1. **主要功能**：建立 2D 掩码（Mask）与 3D 点云（Points）之间的对应关系。在之前的步骤中，我们已经有了：

2D 语义：图像上的像素被标记为某个部件，存储在 `{DATA_ROOT}/labeled/rendered/{点云名称}/oriented/masks/merged/allmasks.pt` 中。

3D 几何：物体表面的 3D 采样点，存储在 `{DATA_ROOT}/labeled/points/{点云名称}/points.pt` 中。

几何映射：像素到网格面的映射 (`pix2face.pt`) 和 点到网格面的映射 (`point2face.pt`)。

这一脚本利用网格面（Face）作为中间桥梁，将 2D 像素 和 3D 点 关联起来。如果一个 2D 掩码覆盖了某个像素，且该像素对应的网格面上采样了某个 3D 点，那么这个 3D 点就被打上该掩码的标签。

2. **产出内容**：在 `DATA_ROOT/labeled/rendered/{点云名称}/oriented/masks/merged/` 目录下生成 `mask2points.pt`，形状为 (N_MASKS, N_POINTS)。这个张量的行索引 i 对应第 i 个掩码（即 allmasks.pt 中的第 $i$ 个掩码，以及 mask_labels.txt 中的第 $i$ 行标签）；列索引 j 对应第 j 个 3D 采样点（即 points.pt 中的第 $j$ 个点）。值 1 表示第 j 个点属于第 i 个掩码（即该点属于这个部件）。值 0 表示不属于。

3. 修改了脚本中的错误，主要是 `label_mask2pt()` 和 `visualize_mask_pts()` 的第二个参数，应为主函数中的 `nameuid`而非 `nameuid`。