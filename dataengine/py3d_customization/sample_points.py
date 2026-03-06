import os
import torch
import gzip
import json
import pandas as pd
from tqdm import tqdm
from pytorch3d.structures import Meshes
# 导入自定义的采样函数
from dataengine.py3d_customization.sample_points_from_meshes import sample_points_from_meshes
# 导入模型加载函数
from dataengine.utils.meshutils import glb_to_py3d, normalize_mesh
# 导入配置
from dataengine.configs import DATA_ROOT

# 如果处理的是 PLY，还需要这个加载函数
import trimesh
from pytorch3d.renderer import TexturesVertex

def _compute_norm_params_from_verts(verts: torch.Tensor):
    """
    与 dataengine.utils.meshutils.normalize_mesh 完全一致的参数：
    x' = (x - center) / scale
    """
    center = verts.mean(0)
    scale = (verts - center).abs().max(0)[0].max()
    scale = float(scale)
    if scale == 0:
        scale = 1.0
    return center, scale

def ply_to_py3d(path):
    mesh_t = trimesh.load(path, force='mesh')
    if isinstance(mesh_t, trimesh.Scene):
        if len(mesh_t.geometry) == 0:
            raise ValueError(f"Empty scene in {path}")
        mesh_t = trimesh.util.concatenate(tuple(mesh_t.geometry.values()))
    verts = torch.from_numpy(mesh_t.vertices).float()
    faces = torch.from_numpy(mesh_t.faces).long()

    # 记录 normalize 参数（与 normalize_mesh 保持一致）
    center, scale = _compute_norm_params_from_verts(verts)

    colors = torch.ones_like(verts) * 0.8
    textures = TexturesVertex(verts_features=[colors])
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    normalize_mesh(mesh)
    return mesh, center, scale

def process_mesh_sampling(chunk_idx=0, num_samples=10000):
    # 路径设置
    chunk_file = f"{DATA_ROOT}/labeled/chunk_ids/chunk{chunk_idx}.csv"
    glbs_root = f"{DATA_ROOT}/labeled/glbs"
    paths_file = f"{glbs_root}/object-paths.json.gz"
    out_root = f"{DATA_ROOT}/labeled/points"

    if not os.path.exists(chunk_file):
        print(f"Chunk file not found: {chunk_file}")
        return

    # 读取 UID 和路径
    cur_df = pd.read_csv(chunk_file)
    uids = cur_df["uid"].tolist()
    classes = cur_df["class"].tolist()

    with gzip.open(paths_file, "rb") as f:
        path_dict = json.loads(f.read())

    print(f"Sampling points for {len(uids)} objects...")
    for uid, classname in tqdm(zip(uids, classes)):
        try:
            save_dir = f"{out_root}/{classname}_{uid}"
            norm_meta_path = f"{save_dir}/norm_transform.json"

            if (
                os.path.exists(f"{save_dir}/points.pt")
                and os.path.exists(f"{save_dir}/point2face.pt")
                and os.path.exists(norm_meta_path)
            ):
                continue
                
            os.makedirs(save_dir, exist_ok=True)
            
            # 2. 加载 Mesh
            if uid not in path_dict:
                print(f"Path not found for {uid}")
                continue
                
            rel_path = path_dict[uid]
            full_path = os.path.join(glbs_root, rel_path)
            
            if not os.path.exists(full_path):
                print(f"File not found: {full_path}")
                continue

            # 加载并放到 GPU
            if full_path.lower().endswith('.ply'):
                mesh, norm_center, norm_scale = ply_to_py3d(full_path)
                mesh = mesh.cuda()
            else:
                mesh = glb_to_py3d(full_path).cuda()
                # glb 暂时不给迁移脚本使用；若你全是 ply，这里不会走到
                norm_center = torch.zeros(3)
                norm_scale = 1.0

            # [新增] 动态计算采样数
            # 获取网格顶点数
            mesh_verts_count = mesh.verts_packed().shape[0]
            # 策略：采样数 = max(10000, 网格顶点数)
            # 这样既保证了简单物体至少有 10k 点，又保证了复杂物体有足够多的点
            dynamic_num_samples = max(num_samples, mesh_verts_count) 
            # 或者更激进一点：采样数 = 网格面数 (通常是顶点的2倍)
            # dynamic_num_samples = mesh.faces_packed().shape[0]
            print(f"UID {uid}: Mesh has {mesh_verts_count} verts. Sampling {dynamic_num_samples} points.")
            
            # 3. 采样点
            # sample_points_from_meshes 返回 (samples, normals, textures, mappers)
            # 我们只需要 samples 和 mappers (即 point2face)
            samples, normals, textures, point2face = sample_points_from_meshes(
                mesh, 
                num_samples=dynamic_num_samples,    # 使用动态数量 
                return_normals=True, 
                return_textures=True
            )
            
            # 4. 保存结果
            # samples 形状 (1, N, 3) -> squeeze -> (N, 3)
            # point2face 形状 (1, N) -> squeeze -> (N,)
            torch.save(samples.squeeze().cpu(), f"{save_dir}/points.pt")
            torch.save(point2face.squeeze().cpu(), f"{save_dir}/point2face.pt")
            norm_meta = {
                "center": [float(x) for x in norm_center.cpu().tolist()],
                "scale": float(norm_scale),
                "formula": "x_norm = (x - center) / scale"
            }
            with open(norm_meta_path, "w", encoding="utf-8") as f:
                json.dump(norm_meta, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Failed on {uid}: {e}")

if __name__ == "__main__":
    # 需要先安装 pytorch3d
    # 确保 dataengine 在 PYTHONPATH 中
    process_mesh_sampling(0)