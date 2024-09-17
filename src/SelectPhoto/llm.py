import os
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity
import optuna
from optuna.samplers import NSGAIIISampler
from optuna.samplers.nsgaii import BaseCrossover
import itertools
from collections import Counter
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

frame_list = [
    {
        "frame_name": "Zoom-in landscape",
        "frame_description": "Gradually zooming into the landscape to emphasize the beauty of nature.",
        "photo_count": 1  # 1枚の写真を使用
    },
    {
        "frame_name": "Split person portrait",
        "frame_description": "Splitting the portrait into two halves, merging them to show different expressions of the same person.",
        "photo_count": 1  # 1枚の写真を使用
    },
    {
        "frame_name": "Panoramic scroll",
        "frame_description": "Scrolling from left to right across a wide panoramic shot to capture the expansive scene.",
        "photo_count": 1  # 1枚の写真を使用
    },
    {
        "frame_name": "Slow fade portrait",
        "frame_description": "Slowly fading in and out of a close-up portrait to create a dramatic, intimate feel.",
        "photo_count": 1  # 1枚の写真を使用
    },
    {
        "frame_name": "Slide-over transition",
        "frame_description": "Sliding one image over another to transition smoothly between two related images.",
        "photo_count": 2  # 2枚の写真を使用
    },
    {
        "frame_name": "Black-and-white highlight",
        "frame_description": "Converting the image to black and white with a single color highlighted for dramatic effect.",
        "photo_count": 1  # 1枚の写真を使用
    },
    {
        "frame_name": "Zoom-out reveal",
        "frame_description": "Starting from a close-up and gradually zooming out to reveal the full context of the scene.",
        "photo_count": 1  # 1枚の写真を使用
    },
    {
        "frame_name": "Cross dissolve",
        "frame_description": "Gently blending two images together, creating a smooth transition from one scene to the next.",
        "photo_count": 2  # 2枚の写真を使用
    },
    {
        "frame_name": "Circular focus",
        "frame_description": "Using a circular zoom effect to focus on the center of the image, highlighting key elements.",
        "photo_count": 1  # 1枚の写真を使用
    },
    {
        "frame_name": "Tilt-shift effect",
        "frame_description": "Blurring the edges of the image to create a miniature-like appearance, emphasizing the central focus.",
        "photo_count": 1  # 1枚の写真を使用
    },
    {
        "frame_name": "Frame within a frame",
        "frame_description": "Showing a photo within a photo effect to give a layered, reflective sense of the moment.",
        "photo_count": 1  # 1枚の写真を使用
    },
    {
        "frame_name": "Motion blur",
        "frame_description": "Adding a slight motion blur effect to simulate movement and create dynamic energy in the photo.",
        "photo_count": 1  # 1枚の写真を使用
    },
    {
        "frame_name": "Mirror reflection",
        "frame_description": "Creating a mirror reflection effect by flipping the image horizontally, emphasizing symmetry.",
        "photo_count": 1  # 1枚の写真を使用
    },
    {
        "frame_name": "Border highlight",
        "frame_description": "Using a colored border around the image to create emphasis and separate it from the background.",
        "photo_count": 1  # 1枚の写真を使用
    },
    {
        "frame_name": "Zoom and rotate",
        "frame_description": "Simultaneously zooming and slightly rotating the image to add a dynamic, energetic feel.",
        "photo_count": 1  # 1枚の写真を使用
    }
]

# モデルID
model_id = "microsoft/Phi-3.5-vision-instruct"

# 8bit量子化の設定
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# モデルを8bit量子化でロード
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='flash_attention_2',
    quantization_config=quantization_config
)

# プロセッサをロード
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    num_crops=1
)

# 画像のExifデータから撮影日時を取得する関数
def get_exif_data(image):
    exif_data = image._getexif()
    if exif_data is not None:
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "DateTimeOriginal":
                return value
    return "No Date"

# LLMにプロンプトを渡して説明を生成する関数
def generate_description(file_name, date_time, image):
    images = [image]
    prompt = f"ファイル名: {file_name}, 撮影日時: {date_time}。Please write a detailed description of this photo as a memory."

    messages = [
        {"role": "user", "content": f"<|image_1|>\n{prompt}"}
    ]

    inputs = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(inputs, images=images, return_tensors="pt").to("cuda")

    generation_args = {
        "max_new_tokens": 200,
        "temperature": 0.0,
        "do_sample": False
    }

    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    description = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return description

# メインの処理
def process_images():
    image_descriptions = []
    image_folder = "images"
    
    for file_name in os.listdir(image_folder):
        if file_name.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_folder, file_name)
            image = Image.open(image_path)

            # Exifデータから撮影日時を取得
            date_time = get_exif_data(image)

            # LLMにプロンプトを渡して説明を生成
            description = generate_description(file_name, date_time, image)
            image_descriptions.append({
                "file_name": file_name,
                "date_time": date_time,
                "description": description
            })
            #メモリ解放
            torch.cuda.empty_cache()

    return image_descriptions

# 画像情報をJSON形式で保存する関数
def save_image_info_to_json(image_description_list, json_file_name="image_info.json"):
    with open(json_file_name, 'w', encoding='utf-8') as f:
        json.dump(image_description_list, f, ensure_ascii=False, indent=4)
    print(f"画像情報を {json_file_name} に保存しました。")

# 画像説明の出力と保存
image_description_list = process_images()

# 画像情報をJSONファイルに保存
save_image_info_to_json(image_description_list)
