import os
from PIL import Image
import datasets
from tqdm import tqdm
import sys
sys.path.append("/mnt/shared-storage-user/tanxin/lilinfeng/new_track4/GeoTextRetrieve/EVA-CLIP/EVA-CLIP/rei")
from eva_clip import create_model_and_transforms, get_tokenizer

# --- 1. 配置你的路径和模型 ---
DATA_PATH = '/mnt/shared-storage-user/tanxin/lilinfeng/new_track4/track4/data/new_split_merged_train.json'
IMAGE_PATH = '/mnt/shared-storage-user/tanxin/lilinfeng/robosense_track4/track4-cross-modal-drone-navigation/images'
EVA_MODEL_NAME = "EVA02-CLIP-L-14" # 确保和你的训练脚本一致
EVA_PRETRAINED_PATH = "/mnt/shared-storage-user/tanxin/lilinfeng/robosense_track4/BAAI/EVA02_CLIP_L_psz14_s4B.pt"

# --- 2. 加载数据集和图像处理器 ---
print("Loading dataset...")
full_dataset = datasets.load_dataset('json', data_files=DATA_PATH, split='train')

print("Creating image processor...")
_, _, image_processor = create_model_and_transforms(
    EVA_MODEL_NAME, 
    EVA_PRETRAINED_PATH,
    force_custom_clip=True
)

# --- 3. 循环遍历，找到问题样本 ---
print(f"Checking {len(full_dataset)} samples...")
for i, item in enumerate(tqdm(full_dataset)):
    try:
        # 模拟 __getitem__ 的核心逻辑
        positive_img_path = item.get("positive_value")
        
        if not positive_img_path:
            print(f"ERROR at index {i}: 'positive_value' is missing or empty.")
            continue

        complete_img_path = os.path.join(IMAGE_PATH, positive_img_path)

        # 步骤 A: 检查文件是否存在
        if not os.path.exists(complete_img_path):
            print(f"ERROR at index {i}: File not found at '{complete_img_path}'")
            continue

        # 步骤 B: 尝试打开图片
        with Image.open(complete_img_path) as img:
            # 步骤 C: 检查图片模式并尝试转换
            if img.mode != 'RGB':
                # print(f"INFO at index {i}: Image '{positive_img_path}' has mode '{img.mode}'. Converting to RGB.")
                img = img.convert('RGB')
            
            # 步骤 D: 尝试用 image_processor 处理
            processed_img = image_processor(img)

    except Exception as e:
        print(f"\n--- FATAL ERROR at index {i} ---")
        print(f"Data item: {item}")
        print(f"Attempted to open path: {complete_img_path}")
        print(f"Caught Exception: {e}")
        # 找到第一个错误后就可以停止了
        break

print("Debug script finished.")