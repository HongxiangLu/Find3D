# for usage see
# training: model.training.loss DistillLossContrastive
# evaluate: model.evaluation.core compute_overall_iou_objwise (this is iou per mask)

# this creates under
# e.g. [name]_uid/norotate/masks/merged
# - mask2points.pt (this is n_masks*n_pts binary)
import os
import torch
import os.path as osp
import time
from tqdm import tqdm
from common.utils import visualize_pts
import matplotlib.pyplot as plt
import pandas as pd
from dataengine.configs import DATA_ROOT

def label_mask2pt(obj_dir, nameuid):
    # root_dir is e.g. [name]_uid/norotate
    if not os.path.exists(f"{obj_dir}/masks/merged"): # if no mask, skip
        return
    if len(os.listdir(f"{obj_dir}/masks/merged")) == 0:
        return
    if os.path.exists(f"{obj_dir}/masks/merged/mask2points.pt"): # if already labeled, skip
        pass
    # get per mask points
    pix2frontface = torch.load(osp.join(obj_dir, "pix2face.pt")).cuda() # pix2frontface is n_view, h, w and the value is face index
    point2face = torch.load(f"{DATA_ROOT}/labeled/points/{nameuid}/point2face.pt").cuda() # point2face is of size 5000, each a face index for the point
    all_masks = torch.load(f"{obj_dir}/masks/merged/allmasks.pt").cuda()
    all_masks = all_masks.view(all_masks.shape[0],-1).type(torch.cuda.FloatTensor)
    mask2view = torch.load(f"{obj_dir}/masks/merged/mask2view.pt").cuda()
    n_views = 10

    mask2pt_list = []
    # for each view, for now we don't parallelize since io needs to be sequential anyway
    # all the labels are in order from view0 to view10
    # so by concatenating by filtered view idx in order we preserve the original order
    
    # [优化] 设置分批大小，例如每次处理 1000 个点
    BATCH_SIZE = 1000 
    n_points = point2face.shape[0]

    for i in range(n_views):
        pix2face = pix2frontface[i,:,:].view(-1)  # (H*W,)
        masks_this_view = all_masks[mask2view==i,:] # (K, H*W)
        
        # 如果这个视角没有 mask，直接给全 0
        if masks_this_view.shape[0] == 0:
            mask2pt_list.append(torch.zeros(0, n_points).cuda())
            continue

        # 结果容器：(K, N_PTS)
        view_mask2pt = torch.zeros(masks_this_view.shape[0], n_points, device='cuda')

        # === 分批处理点 (Points Batching) ===
        for start_idx in range(0, n_points, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, n_points)
            
            # 取出一批点对应的面索引: (Batch,)
            batch_point2face = point2face[start_idx:end_idx]
            
            # 计算这一批点的 point2pix: (Batch, H*W)
            # 广播操作只在这个小 Batch 上进行，显存占用降为原来的 1/10 (如果Batch=1000)
            batch_point2pix = ((pix2face - batch_point2face.unsqueeze(1)) == 0).float()
            
            # 矩阵乘法: (K, H*W) @ (H*W, Batch) -> (K, Batch)
            # 注意：这里 batch_point2pix 是 (Batch, H*W)，需要转置
            batch_masks_pts = masks_this_view @ batch_point2pix.T
            
            # 存入结果
            view_mask2pt[:, start_idx:end_idx] = (batch_masks_pts > 0).float()
            
            # 及时释放中间显存
            del batch_point2pix
            del batch_masks_pts
        
        mask2pt_list.append(view_mask2pt)
        
        # 清理视角级显存
        del pix2face
        del masks_this_view
    
    mask2pt = torch.cat(mask2pt_list, dim=0) # this should be n_masks, n_pts
    torch.save(mask2pt, f"{obj_dir}/masks/merged/mask2points.pt")

# for debugging
def visualize_mask_pts(obj_dir, nameuid):
    mask2points = torch.load(f"{obj_dir}/masks/merged/mask2points.pt")
    allmasks = torch.load(f"{obj_dir}/masks/merged/allmasks.pt")
    f = open(f"{obj_dir}/masks/merged/mask_labels.txt", "r")
    labels = f.read().splitlines()
    f.close()
    pt_xyz = torch.load(f"{DATA_ROOT}/labeled/points/{nameuid}/points.pt")

    rand_indices = [10,13,36]

    for idx in rand_indices:
        print(labels[idx])
        mask = allmasks[idx,:,:]
        plt.imshow(mask)
        plt.savefig(f"viz/{labels[idx]}")
        mask2points_idx = mask2points[idx,:] # binary of (n_pts,)
        
        # mark these points purple
        rgb_r = 1-(mask2points_idx * 0.5).view(-1,1)
        rgb_g = 1-mask2points_idx.view(-1,1)
        rgb_b = 1-(mask2points_idx * 0.5).view(-1,1)
        rgb_all = torch.cat([rgb_r,rgb_g, rgb_b], dim=1)
        visualize_pts(pt_xyz, rgb_all, save_path=f"viz/pc{labels[idx]}")



if __name__ == "__main__":
    chunk_id = 1 # change this to process all chunks

    parent_folder = f"{DATA_ROOT}/labeled/rendered"
    cur_df = pd.read_csv(f"{DATA_ROOT}/labeled/chunk_ids/chunk{chunk_id}.csv")
    cur_df["path"] = cur_df["class"] + "_" + cur_df["uid"]
    child_dirs = cur_df["path"].tolist()

    start = time.time()
    for nameuid in tqdm(child_dirs):
        full_dir = parent_folder+"/"+nameuid
        # uid = nameuid.split("_")[-1]
        label_mask2pt(f"{full_dir}/oriented", nameuid)
        # visualize for debugging
        # visualize_mask_pts(f"{full_dir}/oriented", uid)
    end = time.time()