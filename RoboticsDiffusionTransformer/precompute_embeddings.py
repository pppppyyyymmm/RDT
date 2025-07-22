import os
import json
import torch
import yaml
from tqdm import tqdm
from models.multimodal_encoder.t5_encoder import T5Embedder

# --- 1. 请在这里配置你的路径 ---
GPU = 0
MODEL_PATH = "google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"

# 你的数据集的 meta.jsonl 文件路径
META_FILE_PATH = "data/datasets/so101_2cam_banana_240x320_opencv/meta/episodes.jsonl"

# 【硬编码】你的固定指令
# 既然所有指令都一样，直接在这里写死，避免读取文件
INSTRUCTION = "Pick up the banana and place it on the plate."

# VRAM不足24G时，强烈建议开启offload
# 首先在终端创建这个文件夹: mkdir offload_dir
OFFLOAD_DIR = "offload_dir"


# --- 配置结束 ---

def main():
    # 确定保存 embedding 的目录 (即 meta.jsonl 所在的目录)
    save_dir = os.path.dirname(META_FILE_PATH)

    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)

    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH,
        model_max_length=config["dataset"]["tokenizer_max_length"],
        device=device,
        use_offload_folder=OFFLOAD_DIR
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    # --- 1. 只编码一次指令 ---
    print(f"Encoding the single instruction: '{INSTRUCTION}'")
    tokenized_res = tokenizer(INSTRUCTION, return_tensors="pt", padding="longest", truncation=True)
    tokens = tokenized_res["input_ids"].to(device)
    attn_mask = tokenized_res["attention_mask"].to(device)

    with torch.no_grad():
        text_embeds = text_encoder(input_ids=tokens, attention_mask=attn_mask)["last_hidden_state"].detach().cpu()

    attn_mask = attn_mask.cpu().bool()
    # 得到我们唯一的 embedding
    the_one_embedding = text_embeds[0][attn_mask[0]]
    print("Encoding complete.")

    # --- 2. 读取 meta 文件，只为了获取 episode_index ---
    episodes_indices = []
    with open(META_FILE_PATH, 'r') as f:
        for line in f:
            episodes_indices.append(json.loads(line)['episode_index'])

    # --- 3. 循环保存同一份 embedding ---
    for episode_index in tqdm(episodes_indices, desc="Saving embedding copies"):
        save_path = os.path.join(save_dir, f"lang_embed_{episode_index}.pt")
        torch.save(the_one_embedding, save_path)

    print(f"完成！所有 embedding 已保存至: {save_dir}")


if __name__ == "__main__":
    main()