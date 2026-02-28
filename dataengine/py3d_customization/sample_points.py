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

def ply_to_py3d(path):
    mesh_t = trimesh.load(path, force='mesh')
    if isinstance(mesh_t, trimesh.Scene):
        if len(mesh_t.geometry) == 0:
            raise ValueError(f"Empty scene in {path}")
        mesh_t = trimesh.util.concatenate(tuple(mesh_t.geometry.values()))
    verts = torch.from_numpy(mesh_t.vertices).float()
    faces = torch.from_numpy(mesh_t.faces).long()
    colors = torch.ones_like(verts) * 0.8 # 颜色在这里不重要
    textures = TexturesVertex(verts_features=[colors])
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    normalize_mesh(mesh)
    return mesh

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
            # 1. 检查是否已处理
            save_dir = f"{out_root}/{classname}_{uid}"
            if os.path.exists(f"{save_dir}/points.pt") and os.path.exists(f"{save_dir}/point2face.pt"):
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
                mesh = ply_to_py3d(full_path).cuda()
            else:
                mesh = glb_to_py3d(full_path).cuda()
            
            # 3. 采样点
            # sample_points_from_meshes 返回 (samples, normals, textures, mappers)
            # 我们只需要 samples 和 mappers (即 point2face)
            samples, normals, textures, point2face = sample_points_from_meshes(
                mesh, 
                num_samples=num_samples, 
                return_normals=True, 
                return_textures=True
            )
            
            # 4. 保存结果
            # samples 形状 (1, N, 3) -> squeeze -> (N, 3)
            # point2face 形状 (1, N) -> squeeze -> (N,)
            torch.save(samples.squeeze().cpu(), f"{save_dir}/points.pt")
            torch.save(point2face.squeeze().cpu(), f"{save_dir}/point2face.pt")
            
        except Exception as e:
            print(f"Failed on {uid}: {e}")

if __name__ == "__main__":
    # 需要先安装 pytorch3d
    # 确保 dataengine 在 PYTHONPATH 中
    process_mesh_sampling()