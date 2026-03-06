import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm

from dataengine.configs import DATA_ROOT


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def build_sampled_point_scores(mask2points, mask_labels):
    """
    mask2points: (n_masks, n_sampled_pts), 0/1
    mask_labels: length n_masks (text)
    返回:
      unique_labels: list[str]
      sampled_scores: (n_sampled_pts, n_unique_labels), float32
    """
    unique_labels = sorted(list(set(mask_labels)))
    label2id = {lb: i for i, lb in enumerate(unique_labels)}

    n_masks, n_pts = mask2points.shape
    n_labels = len(unique_labels)

    sampled_scores = np.zeros((n_pts, n_labels), dtype=np.float32)

    # 累加每个 mask 对应标签的命中
    for m in range(n_masks):
        lb = mask_labels[m]
        lid = label2id[lb]
        sampled_scores[:, lid] += mask2points[m].astype(np.float32)

    # 归一化成“置信度”形式（按点归一）
    row_sum = sampled_scores.sum(axis=1, keepdims=True)
    valid = row_sum[:, 0] > 0
    sampled_scores[valid] /= row_sum[valid]

    return unique_labels, sampled_scores


def resolve_raw_pcd_path(cls, uid, raw_root, raw_index_dict):
    if raw_index_dict is not None:
        if uid not in raw_index_dict:
            return None
        return os.path.join(raw_root, raw_index_dict[uid])
    # 默认: raw_root/{uid}.ply
    return os.path.join(raw_root, cls, f"{uid}.ply")


def process_one_object(uid, cls, raw_root, raw_index_dict, out_root, k=1, distance_upper_bound=np.inf):
    # 1) 已有中间结果路径
    nameuid = f"{cls}_{uid}"
    oriented_dir = os.path.join(DATA_ROOT, "labeled", "rendered", nameuid, "oriented")
    merged_dir = os.path.join(oriented_dir, "masks", "merged")
    points_dir = os.path.join(DATA_ROOT, "labeled", "points", f"{cls}_{uid}")

    need_files = {
        "mask2points": os.path.join(merged_dir, "mask2points.pt"),
        "mask_labels": os.path.join(merged_dir, "mask_labels.txt"),
        "sampled_points": os.path.join(points_dir, "points.pt"),
    }

    norm_meta_path = os.path.join(points_dir, "norm_transform.json")

    for kf, fp in need_files.items():
        if not os.path.exists(fp):
            return False, f"missing {kf}: {fp}"

    # 2) 原始点云路径
    raw_pcd_path = resolve_raw_pcd_path(cls, uid, raw_root, raw_index_dict)
    if raw_pcd_path is None or (not os.path.exists(raw_pcd_path)):
        return False, f"missing raw pcd: {raw_pcd_path}"

    # 3) 读取中间结果
    mask2points = torch.load(need_files["mask2points"], map_location="cpu").numpy()  # (n_masks, n_sampled)
    mask_labels = load_lines(need_files["mask_labels"])
    sampled_points = torch.load(need_files["sampled_points"], map_location="cpu").numpy()  # (n_sampled, 3)

    if mask2points.shape[0] != len(mask_labels):
        return False, f"mask count mismatch: mask2points={mask2points.shape[0]}, labels={len(mask_labels)}"

    # 4) 构建采样点标签分数
    unique_labels, sampled_scores = build_sampled_point_scores(mask2points, mask_labels)

    # 5) 读取原始点云
    raw_pcd = o3d.io.read_point_cloud(raw_pcd_path)
    raw_points = np.asarray(raw_pcd.points, dtype=np.float32)
    if raw_points.shape[0] == 0:
        return False, f"empty raw pcd: {raw_pcd_path}"

    # 默认直接查询
    raw_points_query = raw_points

    # 若存在 normalize 参数，则把 raw_points 映射到与 sampled_points 相同坐标系
    if os.path.exists(norm_meta_path):
        with open(norm_meta_path, "r", encoding="utf-8") as f:
            norm_meta = json.load(f)
        center = np.asarray(norm_meta["center"], dtype=np.float32)
        scale = float(norm_meta["scale"])
        if scale == 0:
            scale = 1.0
        raw_points_query = (raw_points - center[None, :]) / scale

    # 6) 最近邻迁移
    tree = cKDTree(sampled_points)

    if k <= 1:
        dists, idx = tree.query(raw_points_query, k=1, distance_upper_bound=distance_upper_bound)
        # idx 可能等于 sampled_points.shape[0]（表示没找到）
        invalid = idx >= sampled_points.shape[0]
        raw_scores = np.zeros((raw_points.shape[0], sampled_scores.shape[1]), dtype=np.float32)
        valid_idx = ~invalid
        raw_scores[valid_idx] = sampled_scores[idx[valid_idx]]
    else:
        dists, idx = tree.query(raw_points_query, k=k, distance_upper_bound=distance_upper_bound)
        # idx: (n_raw, k)
        n_sampled = sampled_points.shape[0]
        raw_scores = np.zeros((raw_points.shape[0], sampled_scores.shape[1]), dtype=np.float32)

        for i in range(raw_points.shape[0]):
            cur_idx = idx[i]
            cur_dist = dists[i]
            valid = cur_idx < n_sampled
            if not np.any(valid):
                continue
            cur_idx = cur_idx[valid]
            cur_dist = cur_dist[valid]
            # 距离加权（越近权重越大）
            w = 1.0 / (cur_dist + 1e-8)
            w = w / w.sum()
            raw_scores[i] = (sampled_scores[cur_idx] * w[:, None]).sum(axis=0)

    # 7) top1 标签
    top1_id = raw_scores.argmax(axis=1)
    top1_conf = raw_scores.max(axis=1)
    top1_text = [unique_labels[i] for i in top1_id]

    # 8) 保存输出
    obj_out = os.path.join(out_root, cls, uid)
    os.makedirs(obj_out, exist_ok=True)

    np.save(os.path.join(obj_out, "raw_points.npy"), raw_points)
    np.save(os.path.join(obj_out, "raw_point_top1_label_id.npy"), top1_id.astype(np.int32))
    np.save(os.path.join(obj_out, "raw_point_top1_conf.npy"), top1_conf.astype(np.float32))
    np.save(os.path.join(obj_out, "raw_point_label_scores.npy"), raw_scores.astype(np.float32))

    with open(os.path.join(obj_out, "label_vocab.txt"), "w", encoding="utf-8") as f:
        for lb in unique_labels:
            f.write(lb + "\n")

    with open(os.path.join(obj_out, "raw_point_top1_label.txt"), "w", encoding="utf-8") as f:
        for lb in top1_text:
            f.write(lb + "\n")

    meta = {
        "uid": uid,
        "class": cls,
        "raw_pcd_path": raw_pcd_path,
        "n_raw_points": int(raw_points.shape[0]),
        "n_sampled_points": int(sampled_points.shape[0]),
        "n_masks": int(mask2points.shape[0]),
        "n_unique_labels": int(len(unique_labels)),
        "k_neighbors": int(k),
    }
    with open(os.path.join(obj_out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--raw_root", type=str, default=os.path.join(DATA_ROOT, "raw", "raw_pcd"))
    parser.add_argument("--raw_index_json", type=str, default="", help="可选: uid->relative_path 的json")
    parser.add_argument("--out_root", type=str, default=os.path.join(DATA_ROOT, "raw", "raw_point_labels"))
    parser.add_argument("--k", type=int, default=1, help="最近邻数量，1表示最近邻复制标签")
    args = parser.parse_args()

    chunk_file = os.path.join(DATA_ROOT, "labeled", "chunk_ids", f"chunk{args.chunk_idx}.csv")
    if not os.path.exists(chunk_file):
        raise FileNotFoundError(chunk_file)

    df = pd.read_csv(chunk_file)
    if "uid" not in df.columns or "class" not in df.columns:
        raise ValueError("chunk csv 需要包含 uid 和 class 两列")

    raw_index_dict = None
    if args.raw_index_json:
        with open(args.raw_index_json, "r", encoding="utf-8") as f:
            raw_index_dict = json.load(f)

    os.makedirs(args.out_root, exist_ok=True)

    ok_cnt = 0
    fail_cnt = 0
    fail_log = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Transfer labels to raw pcd"):
        uid = str(row["uid"])
        cls = str(row["class"])
        ok, msg = process_one_object(
            uid=uid,
            cls=cls,
            raw_root=args.raw_root,
            raw_index_dict=raw_index_dict,
            out_root=args.out_root,
            k=args.k,
        )
        if ok:
            ok_cnt += 1
        else:
            fail_cnt += 1
            fail_log.append(f"{cls}_{uid}: {msg}")

    with open(f"transfer_fail_chunk{args.chunk_idx}.txt", "w", encoding="utf-8") as f:
        for line in fail_log:
            f.write(line + "\n")

    print(f"[DONE] success={ok_cnt}, fail={fail_cnt}, out_root={args.out_root}")


if __name__ == "__main__":
    main()