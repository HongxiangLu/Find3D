import os
import json
import time
import base64
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dataengine.configs import DATA_ROOT

# 配置
N_TOTAL_ENDPOINTS = 1

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_qwen(prompt, image_path):
    api_key = os.getenv("DASHSCOPE_API_KEY") or "sk-你的key"
    if not api_key or api_key.startswith("sk-你的"):
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image(image_path)

    try:
        completion = client.chat.completions.create(
            model="qwen3-vl-plus-2025-12-19",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ],
                }
            ],
            temperature=0.01,
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise e

def part_name_prompt():
    prompt = """
    This is an image rendered from a point cloud of a railway scene.
    What is the name of the object that is masked out as purple?
    If you cannot find the object or are unsure, say unknown.
    Please output ONLY one of the following names:
     - Ground
     - Vegetation
     - Rail
     - Poles
     - Wires
     - Signaling
     - Fences
     - Installation
     - Building
    Do not output any other text or explanation.
    """
    return prompt

def parse_response(response):
    if not response: return "unknown"
    return response.lower().strip().replace("\n", "").replace("`", "").replace(":", "").replace("the answer is", "").replace("the purple part is", "").replace("the part marked out in purple is", "").replace("purple", "").strip()

def query_part_dir(prompt, root_dir, exception_file):
    overall_dict_savepath = os.path.join(root_dir, "masks", "partnames.json")
    
    if not os.path.exists(root_dir): 
        return
    if os.path.exists(overall_dict_savepath):
        return

    masks_dir = os.path.join(root_dir, "masks")
    if not os.path.exists(masks_dir):
        return

    all_image_paths = []
    viewfolders = [f for f in os.listdir(masks_dir) if os.path.isdir(os.path.join(masks_dir, f))]

    for viewfolder in viewfolders:
        view_dir = os.path.join(masks_dir, viewfolder)
        image_names = [f for f in os.listdir(view_dir) if f.endswith(".png")]
        # 保存相对路径: masks/view00/mask0.png
        image_rel_paths = [os.path.join(viewfolder, img_name) for img_name in image_names]
        all_image_paths += image_rel_paths
    
    if not all_image_paths:
        return

    part_dict = {}
    
    # 逐个查询
    for image_rel_path in tqdm(all_image_paths, desc="Querying masks", leave=False):
        full_img_path = os.path.join(masks_dir, image_rel_path)
        
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            try:
                res = query_qwen(prompt, full_img_path)
                res_name = parse_response(res)
                part_dict[image_rel_path] = res_name
                success = True
                break
            except Exception as e:
                time.sleep(2 * (attempt + 1))
        
        if not success:
            part_dict[image_rel_path] = "unknown"
            exception_file.write(f"{root_dir}: {image_rel_path} - query failed\n")

    with open(overall_dict_savepath, "w") as outfile:
        json.dump(part_dict, outfile, indent=4)
    return

def query_object(obj_dir, exception_file):
    prompt = part_name_prompt()
    query_part_dir(prompt, f"{obj_dir}/oriented", exception_file)
    return

def run_endpoint_load(endpoint_idx, total_endpoints):
    chunk_idx = 0
    parent_folder = f"{DATA_ROOT}/labeled/rendered"
    
    chunk_file = f"{DATA_ROOT}/labeled/chunk_ids/chunk{chunk_idx}.csv"
    if not os.path.exists(chunk_file):
        print(f"Chunk file not found: {chunk_file}")
        return

    cur_df = pd.read_csv(chunk_file)
    cur_df["path"] = cur_df["class"] + "_" + cur_df["uid"]
    child_dirs = cur_df["path"].tolist()
    
    full_dirs = [os.path.join(parent_folder, child_dir) for child_dir in child_dirs]
    
    subchunk_size = len(full_dirs) // total_endpoints
    if len(full_dirs) % total_endpoints != 0:
        subchunk_size += 1
    
    start_idx = endpoint_idx * subchunk_size
    end_idx = min((endpoint_idx + 1) * subchunk_size, len(full_dirs))
    
    cur_endpoint_chunk = full_dirs[start_idx:end_idx]
    
    print(f"Processing {len(cur_endpoint_chunk)} directories (Endpoint {endpoint_idx})")

    file_e = open(f"name_part_exceptions_chunk{endpoint_idx}.txt", "a")
    
    for obj_dir in tqdm(cur_endpoint_chunk, desc="Total Progress"):
        try:
            query_object(obj_dir, file_e)
        except Exception as e:
            file_e.write(f"{obj_dir} - process failed - {e}\n")
            
    file_e.close()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Name Parts using Qwen-VL")
    parser.add_argument('endpoint', type=int, nargs='?', default=0, help='Endpoint index (0-based)')
    parser.add_argument('--total', type=int, default=N_TOTAL_ENDPOINTS, help='Total number of endpoints/processes')
    args = parser.parse_args()
    
    run_endpoint_load(args.endpoint, args.total)