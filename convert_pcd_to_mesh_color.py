import os
import json
import gzip
import pandas as pd
import open3d as o3d
import numpy as np
from tqdm import tqdm
import copy

# 配置路径 (根据你的目录结构调整)
DATA_ROOT = "dataroot"  # 假设在 Find3D-new 下运行
CHUNK_ID = 0
CHUNK_FILE = f"{DATA_ROOT}/labeled/chunk_ids/chunk{CHUNK_ID}.csv"
PATHS_FILE = f"{DATA_ROOT}/labeled/glbs/object-paths.json.gz"
GLB_ROOT = f"{DATA_ROOT}/labeled/glbs"

def transfer_colors_nearest_neighbor(original_pcd, mesh):
    """
    将原始点云的颜色迁移到生成的网格顶点上。
    使用 KDTree 查找最近邻。
    """
    if not original_pcd.has_colors():
        print("Original PCD has no colors, skipping color transfer.")
        return mesh

    # 构建 KDTree 用于快速查找
    pcd_tree = o3d.geometry.KDTreeFlann(original_pcd)
    
    # 获取原始颜色数组
    original_colors = np.asarray(original_pcd.colors)
    
    # 获取网格顶点
    mesh_vertices = np.asarray(mesh.vertices)
    new_colors = []
    
    # 对每个网格顶点，找到最近的原始点
    for vertex in mesh_vertices:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
        # idx[0] 是最近邻的索引
        new_colors.append(original_colors[idx[0]])
    
    # 将新颜色赋给网格
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(new_colors))
    
    return mesh

def reconstruct_mesh_from_pcd(pcd_path):
    try:
        # 1. 读取点云
        pcd = o3d.io.read_point_cloud(pcd_path)
        if len(pcd.points) == 0:
            print(f"Skipping {pcd_path}: No points found.")
            return False

        # --- 新增预处理 ---
        
        # 1.1 去除无效点 (NaN/Inf)
        pcd.remove_non_finite_points()
        
        # 1.2 去除重复点 (关键修复)
        pcd = pcd.remove_duplicated_points()
        
        # 1.3 统计滤波去除离群噪点 (可选，有助于提高质量)
        # nb_neighbors: 考虑多少个邻居, std_ratio: 标准差倍数
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

        # 1.4 坐标归一化 (关键修复 Qhull Error)
        # 记录原始中心和缩放，以便重建后恢复
        original_center = pcd.get_center()
        pcd.translate(-original_center) # 移到原点
        
        max_bound = pcd.get_max_bound()
        min_bound = pcd.get_min_bound()
        max_extent = (max_bound - min_bound).max()
        scale_factor = 1.0 / max_extent if max_extent > 0 else 1.0
        
        pcd.scale(scale_factor, center=(0,0,0)) # 缩放到 [-1, 1]
        
        # ------------------
        
        # 保留一份原始点云的深拷贝用于颜色查找
        original_pcd = copy.deepcopy(pcd)

        # 2. 估计法线
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)

        # 3. 泊松重建
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

        # 4. 清理低密度顶点
        densities = np.asarray(densities)
        if len(densities) > 0:
            density_threshold = np.percentile(densities, 5)
            mesh.remove_vertices_by_mask(densities < density_threshold)

        # 5. 颜色迁移
        if original_pcd.has_colors():
            mesh = transfer_colors_nearest_neighbor(original_pcd, mesh)
            
        # --- 恢复坐标 ---
        # 先缩放回去
        mesh.scale(1.0 / scale_factor, center=(0,0,0))
        # 再平移回去
        mesh.translate(original_center)
        # ----------------
        
        # 6. 保存为网格 PLY
        o3d.io.write_triangle_mesh(pcd_path, mesh, write_ascii=False, compressed=True, print_progress=False)
        
        return True

    except Exception as e:
        print(f"Failed to process {pcd_path}: {e}")
        return False

def main():
    print("Loading paths and uids...")
    
    # 检查文件存在性
    if not os.path.exists(CHUNK_FILE):
        print(f"Error: Chunk file not found at {CHUNK_FILE}")
        return
    if not os.path.exists(PATHS_FILE):
        print(f"Error: Paths file not found at {PATHS_FILE}")
        return

    # 读取 UID 列表
    try:
        uids_df = pd.read_csv(CHUNK_FILE)
        target_uids = set(uids_df["uid"].tolist())
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 读取路径映射
    try:
        with gzip.open(PATHS_FILE, "rb") as f:
            all_paths = json.loads(f.read().decode('utf-8'))
    except Exception as e:
        print(f"Error reading JSON.GZ: {e}")
        return

    print(f"Found {len(target_uids)} UIDs to process.")
    
    success_count = 0
    fail_count = 0
    skipped_count = 0

    # 进度条循环
    pbar = tqdm(target_uids, desc="Processing PCDs")
    for uid in pbar:
        if uid not in all_paths:
            skipped_count += 1
            continue
        
        rel_path = all_paths[uid]
        full_path = os.path.join(GLB_ROOT, rel_path)
        
        # 只处理 PLY 文件
        if not full_path.lower().endswith('.ply'):
            skipped_count += 1
            continue
            
        if not os.path.exists(full_path):
            pbar.write(f"File not found: {full_path}") # 使用 pbar.write 防止进度条错乱
            fail_count += 1
            continue

        if reconstruct_mesh_from_pcd(full_path):
            success_count += 1
        else:
            fail_count += 1
            
        # 更新进度条后缀信息
        pbar.set_postfix({"Success": success_count, "Fail": fail_count})

    print(f"\nProcessing Complete.")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped (not PLY or not in paths): {skipped_count}")

if __name__ == "__main__":
    main()