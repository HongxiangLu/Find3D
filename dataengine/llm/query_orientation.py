import os
import base64
import argparse
import shutil
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dataengine.configs import DATA_ROOT

# 配置
# 默认分片数为1，意味着一个进程处理所有数据。如果你想并行运行，可以增加这个数并运行多个脚本实例。
N_TOTAL_ENDPOINTS = 1 

def encode_image(image_path):
    """将图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_qwen(prompt, image_paths):
    """查询通义千问 VL 模型"""
    # 从环境变量获取 API Key，如果没有则尝试硬编码（请替换）
    api_key = os.getenv("DASHSCOPE_API_KEY") or "sk-你的key"
    
    if not api_key or api_key.startswith("sk-你的"):
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或在脚本中填入正确的 API Key")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 构建多模态消息内容
    # Qwen-VL 支持在一条消息中混合文本和多张图片
    content = [{"type": "text", "text": prompt}]
    
    for img_path in image_paths:
        if os.path.exists(img_path):
            base64_image = encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        else:
            print(f"Warning: Image not found at {img_path}")

    try:
        completion = client.chat.completions.create(
            model="qwen-vl-max",  # 使用通义千问 VL Max 模型
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            temperature=0.01, # 降低随机性
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Qwen API call failed: {e}")
        return ""

def construct_prompt():
    prompt = """
    For each image, is the object in an orientation that is usually seen? Please answer yes or no for each image.
    """
    return prompt

def parse_response(response):
    if not response:
        return 0
    n_yes = response.lower().count("yes")
    n_no = response.lower().count("no")
    if n_yes + n_no == 0:
        return 0
    return n_yes / (n_yes + n_no)

def query_uid(root_dir):
    # skip if already annotated
    if not os.path.exists(root_dir): 
        return
    # 如果已经有结果文件，跳过
    if "orientation.txt" in os.listdir(root_dir):
        return
        
    prompt = construct_prompt()
    orientations = ["norotate", "front2top", "flip"]
    correct_orientation = "none"
    max_yes_ratio = -1
    
    # 遍历三个方向
    for orientation in orientations:
        # 获取该方向下的 10 张图片
        img_paths = [f"{root_dir}/{orientation}/imgs/{i:02d}.jpeg" for i in range(10)]
        
        # 检查图片是否存在，至少要有一张
        valid_img_paths = [p for p in img_paths if os.path.exists(p)]
        if not valid_img_paths:
            continue
            
        try:
            # 调用 Qwen
            response_text = query_qwen(prompt, valid_img_paths)
            yes_ratio = parse_response(response_text)
            
            if yes_ratio > max_yes_ratio:
                max_yes_ratio = yes_ratio
                correct_orientation = orientation
                
        except Exception as e:
            print(f"Error processing {root_dir}/{orientation}: {e}")
            pass

    # 如果所有尝试都失败或没有结果，默认使用 norotate
    if correct_orientation == "none":
        correct_orientation = "norotate"
        
    # 保存结果
    with open(f'{root_dir}/orientation.txt', 'w') as f:
        f.write(correct_orientation)
        
    # 删除其他方向的文件夹，只保留正确的，并重命名为 oriented
    for orientation in orientations:
        dir_to_remove = f"{root_dir}/{orientation}"
        if orientation != correct_orientation:
            if os.path.exists(dir_to_remove):
                shutil.rmtree(dir_to_remove)
    
    # 重命名选中的文件夹
    src_dir = f"{root_dir}/{correct_orientation}"
    dst_dir = f"{root_dir}/oriented"
    if os.path.exists(src_dir) and not os.path.exists(dst_dir):
        os.rename(src_dir, dst_dir)
    elif os.path.exists(src_dir) and os.path.exists(dst_dir):
        # 如果目标已存在（可能是异常中断导致），先移除目标再重命名
        shutil.rmtree(dst_dir)
        os.rename(src_dir, dst_dir)
        
    return

def process_endpoint(endpoint_idx, total_endpoints):
    parent_folder = f"{DATA_ROOT}/labeled/rendered"
    chunk_idx = 0 # 对应 render_2d.py 中的 chunk_idx
    
    chunk_file = f"{DATA_ROOT}/labeled/chunk_ids/chunk{chunk_idx}.csv"
    if not os.path.exists(chunk_file):
        print(f"Chunk file not found: {chunk_file}")
        return

    cur_df = pd.read_csv(chunk_file)
    cur_df["path"] = cur_df["class"] + "_" + cur_df["uid"]
    child_dirs = cur_df["path"].tolist()
    
    full_dirs = [os.path.join(parent_folder, child_dir) for child_dir in child_dirs]
    
    # 数据分片逻辑
    subchunk_size = len(full_dirs) // total_endpoints
    if len(full_dirs) % total_endpoints != 0:
        subchunk_size += 1
        
    start_idx = endpoint_idx * subchunk_size
    end_idx = min((endpoint_idx + 1) * subchunk_size, len(full_dirs))
    
    cur_dirs = full_dirs[start_idx:end_idx]
    
    print(f"Processing {len(cur_dirs)} objects (Endpoint {endpoint_idx}/{total_endpoints})")

    file_e = open(f"orientation_exceptions-{endpoint_idx}.txt", "a")
    for dir_path in tqdm(cur_dirs):
        try:
            query_uid(dir_path)
        except Exception as e:
            file_e.write(f"{dir_path}, {e}\n")
            print(f"Failed on {dir_path}: {e}")
    file_e.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Orientation using Qwen-VL")
    parser.add_argument('endpoint', type=int, nargs='?', default=0, help='Endpoint index (0-based)')
    parser.add_argument('--total', type=int, default=N_TOTAL_ENDPOINTS, help='Total number of endpoints/processes')
    
    args = parser.parse_args()
    
    process_endpoint(args.endpoint, args.total)