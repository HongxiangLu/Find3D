# render to 2d and keep track of face/pixel correspondence
import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
import trimesh
import os
import json
import gzip
from PIL import Image
import numpy as np
import pandas as pd
from pytorch3d.structures import Meshes
from tqdm import tqdm
from dataengine.utils.meshutils import glb_to_py3d, normalize_mesh
from common.utils import rotate_pts
from dataengine.configs import DATA_ROOT


def ply_to_py3d(path):
    # 使用 trimesh 加载
    mesh_t = trimesh.load(path, force='mesh')
    
    # trimesh.load 可能返回 Scene，如果是 Scene 则合并几何体
    if isinstance(mesh_t, trimesh.Scene):
        if len(mesh_t.geometry) == 0:
            raise ValueError(f"Empty scene in {path}")
        mesh_t = trimesh.util.concatenate(tuple(mesh_t.geometry.values()))
        
    verts = torch.from_numpy(mesh_t.vertices).float()
    faces = torch.from_numpy(mesh_t.faces).long()
    
    # 处理顶点颜色 (Vertex Colors)
    # PyTorch3D 的 SoftPhongShader 支持 TexturesVertex
    if hasattr(mesh_t.visual, 'vertex_colors') and mesh_t.visual.vertex_colors is not None:
        # vertex_colors 通常是 (N, 4) 的 uint8 RGBA，我们取前3个通道并归一化到 [0, 1]
        colors = torch.from_numpy(mesh_t.visual.vertex_colors[:, :3]).float() / 255.0
    else:
        # 如果没有颜色，给一个默认的白色或灰色
        colors = torch.ones_like(verts) * 0.8
        
    # 创建 TexturesVertex
    textures = TexturesVertex(verts_features=[colors])
    
    # 创建 PyTorch3D Mesh
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    
    # 关键步骤：标准化网格（居中并缩放），以确保相机能看到物体
    normalize_mesh(mesh)
    
    return mesh

def get_cameras(num_views, dist, device = None):
    # up and down alternating
    elev = torch.tile(torch.tensor([30,-20]), (num_views //2,))
    azim = torch.tile(torch.tensor(np.linspace(-180, 180, num=num_views//1, endpoint=False)).float(), (1,))
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)
    return cameras

def get_rasterizer(image_size, blur_radius, faces_per_pixel, cameras, device = None):
    if device is None:
        device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel= faces_per_pixel,
        bin_size = 0,
        perspective_correct=False, # this is important, otherwise gradients will explode!!
    )
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    return rasterizer

def get_phong_shader(cameras, lights, device = None):
    if device is None:
        device = torch.device("cpu")
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
    return shader

def get_face_pixel_correspondence(fragments, faces):
    pix2frontface = fragments.pix_to_face[:,:,:,0]
    # note the index of the faces increments with view
    n_faces = faces.shape[0]
    pix2frontface = pix2frontface*(pix2frontface>=0) % n_faces + -1.0*(pix2frontface<0)
    # the -1 masks remain and others become in the range of [0,n_faces-1]
    return pix2frontface # this is (n_views, h,w)


# assume all uids are in this specific partition
def render_k_views(uid_list, num_views, data_root):
    # 注意：如果你的 PLY 文件不在这个路径，请修改 root_dir
    root_dir = f"{data_root}/labeled/glbs" # 假设你把 ply 放在这里

    fp_correspondence_path = f"{root_dir}/object-paths.json.gz"
    out_dir = f"{data_root}/labeled/rendered"

    with gzip.open(fp_correspondence_path, "rb") as f:
        corr_dict = json.loads(f.read())

    class_corr = pd.read_csv(f"{data_root}/obj1lvis/metadata.csv")
    uid_classes = class_corr[class_corr["uid"].isin(uid_list)]
    del uid_list # to avoid accidentally using this later
    
    uids_neworder = uid_classes["uid"].tolist()
    classes = uid_classes["class"].tolist()
    fps = [corr_dict[uid] for uid in uids_neworder]
    
    cameras = get_cameras(num_views, 3, device = 'cuda')
    lights = PointLights(device='cuda', location=[[0.0, 0.0, -3.0]])
    rasterizer = get_rasterizer(500, 0.00001, 5, cameras, device='cuda')
    shader = get_phong_shader(cameras, lights, device="cuda")
    
    file = open("render_exceptions.txt", "a")  # append mode
    # 在循环内部修改加载逻辑
    for (uid, fp, classname) in tqdm(zip(uids_neworder, fps, classes), total=len(fps)):
        try:
            full_path = os.path.join(root_dir, fp)
            # 根据后缀选择加载方式
            if fp.lower().endswith('.ply'):
                mesh = ply_to_py3d(full_path).cuda()
            else:
                mesh = glb_to_py3d(full_path).cuda()
            # ... (后续的旋转和渲染逻辑保持不变) ...
            # most meshes need to be rotated 180 degrees by z axis
            # after this rotation, most objects are front-facing, some top are facing front
            # they need to be rotated around x axis by 90 degrees
            # since we don't know ahead of time, we render out both
            verts_rotated_v1 = rotate_pts(mesh.verts_packed(), torch.tensor([0,3.14,0]).cuda(), device="cuda")
            verts_rotated_v2 = rotate_pts(verts_rotated_v1, torch.tensor([1.57,0,0]).cuda(), device="cuda")
            verts_rotated_v3 = rotate_pts(verts_rotated_v1, torch.tensor([3.14,0,0]).cuda(), device="cuda")
            mesh_v1 = Meshes(verts=[verts_rotated_v1], faces = [mesh.faces_packed()], textures = mesh.textures)
            mesh_v2 = Meshes(verts=[verts_rotated_v2], faces = [mesh.faces_packed()], textures = mesh.textures)
            mesh_v3 = Meshes(verts=[verts_rotated_v3], faces = [mesh.faces_packed()], textures = mesh.textures)
            del mesh
            mesh_dict = {"norotate": mesh_v1, "front2top": mesh_v2, "flip": mesh_v3}
            for rotate in mesh_dict:
                mesh_new = mesh_dict[rotate]
                fragments = rasterizer(mesh_new.extend(num_views), cameras = cameras)
                pix2face = get_face_pixel_correspondence(fragments, mesh_new.faces_list()[0])
                images = shader(fragments, mesh_new.extend(num_views), cameras=cameras, lights=lights)
                cur_out_dir = f"{out_dir}/{classname}_{uid}/{rotate}"
                os.makedirs(cur_out_dir, exist_ok=True)
                os.makedirs(f"{cur_out_dir}/imgs", exist_ok=True)
                torch.save(pix2face, f"{cur_out_dir}/pix2face.pt")
                for i in range(num_views):
                    rgb = images[i,:,:,:3].cpu().numpy()*255
                    im = Image.fromarray(rgb.astype(np.uint8))
                    im.save(f"{cur_out_dir}/imgs/{i:02d}.jpeg")
                print(f"saved {cur_out_dir}")
                del mesh_new
                del fragments
                del pix2face
                del images
        except Exception as e:
            file.write(f"{classname}_{uid}, {e}\n")
    file.close()
    return


if __name__ == "__main__":
    chunk_idx = 0
    uids = pd.read_csv(f"{DATA_ROOT}/labeled/chunk_ids/chunk{chunk_idx}.csv")["uid"].tolist()
    render_k_views(uids, 10, DATA_ROOT)