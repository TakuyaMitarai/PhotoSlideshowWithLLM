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

# 画像説明の出力
image_description_list = process_images()
for image_info in image_description_list:
    print(f"ファイル名: {image_info['file_name']}")
    print(f"撮影日時: {image_info['date_time']}")
    print(f"説明: {image_info['description']}")
    print("=" * 40)

# SentenceTransformerモデルのロード
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 日時の平均を計算する関数
def average_datetime(datetime_list):
    # datetimeリストをUNIXタイムスタンプに変換して平均を取る
    timestamps = [dt.timestamp() for dt in datetime_list]
    avg_timestamp = sum(timestamps) / len(timestamps)
    return datetime.fromtimestamp(avg_timestamp)

descriptions = [image_info['description'] for image_info in image_description_list]
image_embeddings = model.encode(descriptions)

# 2. フレームの文章ベクトル化
frame_descriptions = [frame['frame_description'] for frame in frame_list]
frame_embeddings = model.encode(frame_descriptions)

# 3. すべてのimage_description間のコサイン類似度を計算
similarity_matrix = cosine_similarity(image_embeddings)

# 4. コサイン類似度行列の上三角行列部分（対角成分を除く）から類似度の上位3つのペアを選ぶ
num_images = len(image_description_list)
similar_pairs = []

# 上三角行列の成分だけを取り出し
for i in range(num_images):
    for j in range(i + 1, num_images):
        similar_pairs.append((similarity_matrix[i, j], i, j))

# 類似度の上位3つのペアを選択
top_3_pairs = sorted(similar_pairs, key=lambda x: x[0], reverse=True)[:6]

# 選ばれたペアの情報を保存するリスト
selected_pairs_info = []

for _, i, j in top_3_pairs:
    # 各ペアのベクトルの平均を計算
    avg_vector = np.mean([image_embeddings[i], image_embeddings[j]], axis=0)
    
    # ファイル名と撮影日時も保存
    file_names = [image_description_list[i]['file_name'], image_description_list[j]['file_name']]
    
    # 撮影日時を取り出して保存
    date_i = datetime.strptime(image_description_list[i]['date_time'], '%Y:%m:%d %H:%M:%S')
    date_j = datetime.strptime(image_description_list[j]['date_time'], '%Y:%m:%d %H:%M:%S')
    avg_date = average_datetime([date_i, date_j])  # 日時の平均を計算
    
    # ペアの情報をリストに追加
    selected_pairs_info.append({
        "image_files": file_names,
        "average_embedding": avg_vector,
        "average_date_time": avg_date.strftime('%Y:%m:%d %H:%M:%S')
    })

# 5. フォトカウントが2のフレームに対して類似度を計算し、最も高いフレームを選択
photo_count_2_frames = [frame for frame in frame_list if frame['photo_count'] == 2]
photo_count_2_frame_embeddings = [frame_embeddings[frame_list.index(frame)] for frame in photo_count_2_frames]

best_frames_for_pairs = []
for pair_info in selected_pairs_info:
    similarities_with_avg = cosine_similarity([pair_info['average_embedding']], photo_count_2_frame_embeddings)
    best_frame_index = np.argmax(similarities_with_avg)
    best_frame = photo_count_2_frames[best_frame_index]['frame_name']
    pair_info['used_frame'] = best_frame  # フレーム名を追加
    best_frames_for_pairs.append(pair_info)

# 6. 残りのimage_descriptionとframe_descriptionの類似度を計算し、最も類似度が高いフレームを選択
used_images = {file for pair in best_frames_for_pairs for file in pair['image_files']}  # 使用済みの画像ファイル名をセットに保存
remaining_images = [i for i in range(num_images) if image_description_list[i]['file_name'] not in used_images]
image_to_frame_map = {}

for img_idx in remaining_images:
    similarities = cosine_similarity([image_embeddings[img_idx]], frame_embeddings)
    best_frame_idx = np.argmax(similarities)
    best_frame = frame_list[best_frame_idx]['frame_name']
    
    # 撮影日時を取得
    image_date = datetime.strptime(image_description_list[img_idx]['date_time'], '%Y:%m:%d %H:%M:%S')
    
    # ファイル名、フレーム、日時を保存
    image_to_frame_map[image_description_list[img_idx]['file_name']] = {
        "used_frame": best_frame,
        "embedding": image_embeddings[img_idx],
        "date_time": image_description_list[img_idx]['date_time']
    }

# 7. ディクショナリ配列の作成と出力
output_dict = {
    "best_frame_for_pairs": best_frames_for_pairs,  # ペアごとの情報を保存
    "other_images": image_to_frame_map  # その他の画像のマッピング
}

# 出力結果を表示
print("ペアごとの最適なフレーム:")
for pair_info in output_dict['best_frame_for_pairs']:
    # ベクトルを除外して出力する
    pair_info_without_vector = {key: value for key, value in pair_info.items() if key != 'average_embedding'}
    print(pair_info_without_vector)


print("\nその他の画像とフレームのマッピング:")
for img, details in output_dict['other_images'].items():
    print(f"画像: {img} - 使用フレーム: {details['used_frame']} - 撮影日時: {details['date_time']}")

# ベクトルとファイル名をoutput_dictから取得してリストにまとめる関数
def extract_embeddings_and_filenames(output_dict):
    embeddings = []
    file_names = []
    
    # ペアごとのベクトルとファイル名
    for pair_info in output_dict['best_frame_for_pairs']:
        embeddings.append(pair_info['average_embedding'])
        file_names.append(f"{pair_info['image_files'][0]} & {pair_info['image_files'][1]}")  # ペアのファイル名を結合して一つの文字列に
    
    # その他の画像のベクトルとファイル名
    for file_name, info in output_dict['other_images'].items():
        embeddings.append(info['embedding'])
        file_names.append(file_name)  # 単独のファイル名
    
    return np.array(embeddings), file_names

# ベクトル類似度を求め、結果をPNGで保存する関数
def visualize_cluster_similarities(clusters, embeddings, file_names, k=3):
    for cluster_id, cluster_file_names in clusters.items():
        # クラスタ内のベクトルを取得
        cluster_embeddings = [embeddings[file_names.index(name)] for name in cluster_file_names]
        
        # コサイン類似度を計算
        similarity_matrix = cosine_similarity(cluster_embeddings)
        
        # 類似度行列をプロット
        fig, ax = plt.subplots()
        cax = ax.matshow(similarity_matrix, cmap='viridis')
        plt.title(f'Cluster {cluster_id + 1} Similarity Matrix', pad=20)
        fig.colorbar(cax)
        
        # x軸、y軸にファイル名をラベルとして表示
        ax.set_xticks(np.arange(len(cluster_file_names)))
        ax.set_yticks(np.arange(len(cluster_file_names)))
        ax.set_xticklabels(cluster_file_names, rotation=90)
        ax.set_yticklabels(cluster_file_names)
        
        plt.tight_layout()

        # PNGとして保存
        output_dir = "cluster_similarity_plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/cluster_{cluster_id + 1}_similarity.png")
        plt.close()

# クラスタリングと可視化
def cluster_descriptions_and_visualize(output_dict, k=3):
    # ベクトルとファイル名を抽出
    embeddings, file_names = extract_embeddings_and_filenames(output_dict)

    # KMeansモデルの作成とフィッティング
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)

    # クラスタリング結果の取得
    cluster_labels = kmeans.labels_

    # クラスタごとにファイル名を保存する辞書を作成
    clusters = {i: [] for i in range(k)}
    for i, label in enumerate(cluster_labels):
        clusters[label].append(file_names[i])

    # 各クラスタのファイル名を出力
    for cluster_id, cluster_file_names in clusters.items():
        print(f"クラスタ {cluster_id + 1}:")
        for file_name in cluster_file_names:
            print(f" - {file_name}")
        print("=" * 40)

    # クラスタごとの類似度を計算しPNGで保存
    visualize_cluster_similarities(clusters, embeddings, file_names, k)

# クラスタリングと類似度可視化の実行
cluster_descriptions_and_visualize(output_dict, k=7)


# エリート個体を選択する戦略（上位50%をそのまま選択）
def elite_population_selection_strategy(study: optuna.Study, trials: list[optuna.trial.FrozenTrial]) -> list[optuna.trial.FrozenTrial]:
    # トライアルを目的関数の値（合計値）でソートし、上位50%をエリートとして選択
    trials_sorted = sorted(trials, key=lambda t: sum(t.values))
    num_elite = len(trials) // 2  # 上位50%を選択
    return trials_sorted[:num_elite]

# カスタムクロスオーバークラスを定義（BaseCrossoverを継承）
class UniformCrossover(BaseCrossover):
    def __init__(self, crossover_prob=0.9):
        self.crossover_prob = crossover_prob

    # 親個体数
    @property
    def n_parents(self):
        return 2

    # クロスオーバーの実装
    def crossover(self, parents_params, rng, study, search_space_bounds):
        # 親個体からのパラメータを一様交叉で子供に渡す
        num_parameters = parents_params.shape[1]
        child_params = np.zeros(num_parameters)

        for i in range(num_parameters):
            # crossover_probの確率で親1と親2のパラメータを交差
            if rng.rand() < self.crossover_prob:
                child_params[i] = parents_params[0, i]  # 親1から引き継ぐ
            else:
                child_params[i] = parents_params[1, i]  # 親2から引き継ぐ

        return child_params

# 反転数をバブルソートで計算する関数
def count_inversions(dates):
    inversions = 0
    n = len(dates)
    for i in range(n):
        for j in range(i + 1, n):
            if dates[i] > dates[j]:
                inversions += 1
    return inversions

# 隣接類似性を計算する関数
def adjacent_similarity(order, embeddings):
    similarity_sum = 0
    for i in range(len(order) - 1):
        sim = sim = cosine_similarity([embeddings[order[i]]], [embeddings[order[i+1]]])[0][0] ** 5
        similarity_sum += sim
    return similarity_sum

# 公平性を計算する関数（各クラスタ内の使用回数は最大で1に制限）
def fairness(order, clusters):
    cluster_counts = {i: 0 for i in range(len(clusters))}  # クラスタごとのカウント

    # すでにカウントされたインデックスを追跡するセット
    counted_indices = set()

    # 各画像がどのクラスタに属しているかを確認
    for idx in order:
        if idx not in counted_indices:  # まだカウントされていない場合
            for cluster_id, cluster in clusters.items():
                if idx in cluster:
                    cluster_counts[cluster_id] += 1  # クラスタ内の使用回数をカウント
                    counted_indices.add(idx)  # このインデックスはもうカウントしない
                    break  # 一つのクラスタでカウントされたら次のインデックスに進む
    
    # 公平性を計算
    fairness_sum = 0
    for cluster_id, cluster in clusters.items():
        if len(cluster) > 0:  # クラスタ内に要素がある場合
            fairness_sum += (cluster_counts[cluster_id] / len(cluster)) ** 2  # 使用比率の二乗を加算
    
    return fairness_sum

# 重複のペナルティを計算する関数
def calculate_penalty(order):
    counter = Counter(order)
    duplicates = sum([count - 1 for count in counter.values() if count > 1])  # 重複数の総和
    return duplicates  # ペナルティの重みとして10倍

# Optunaの目的関数
def objective(trial):
    # 遺伝子の数を30に設定
    num_genes = 30
    
    # 選択肢の数
    num_elements = len(output_dict['best_frame_for_pairs']) + len(output_dict['other_images'])
    
    # 30個の設計変数（遺伝子）を指定された範囲から選択
    order = [trial.suggest_int(f"order_{i}", 0, num_elements - 1) for i in range(num_genes)]

    # 日付リストの作成
    dates = []
    embeddings = []
    clusters = {i: [] for i in range(len(output_dict['best_frame_for_pairs']) + len(output_dict['other_images']))}
    
    # ペアのベクトルと日時をリストに追加
    for i, pair_info in enumerate(output_dict['best_frame_for_pairs']):
        avg_date = datetime.strptime(pair_info['average_date_time'], '%Y:%m:%d %H:%M:%S')
        dates.append(avg_date)
        embeddings.append(pair_info['average_embedding'])
        clusters[i].append(i)

    # 残りの画像のベクトルと日時をリストに追加
    for i, (file_name, info) in enumerate(output_dict['other_images'].items()):
        date = datetime.strptime(info['date_time'], '%Y:%m:%d %H:%M:%S')
        dates.append(date)
        embeddings.append(info['embedding'])
        clusters[len(output_dict['best_frame_for_pairs']) + i].append(len(output_dict['best_frame_for_pairs']) + i)

    # 目的関数
    obj1 = count_inversions([dates[i] for i in order])
    obj2 = -adjacent_similarity(order, embeddings)
    obj3 = fairness(order, clusters)
    penalty = calculate_penalty(order)
    
    obj1 += 10 * penalty
    obj2 -= 100 * penalty
    obj3 += 100 * penalty

    trial.set_user_attr("generation", trial.number // 400)  # 世代ごとにトライアルを分割

    return obj1, obj2, obj3

# 最終世代の上位50個体の解を取得して表示する関数
def get_final_generation_best_trials(study, top_n=50):
    final_generation = max(trial.user_attrs.get("generation", 0) for trial in study.trials)
    final_generation_trials = [trial for trial in study.trials if trial.user_attrs.get("generation", 0) == final_generation]
    sorted_trials = sorted(final_generation_trials, key=lambda t: t.values)
    return sorted_trials[:top_n]

# 画像を読み込んでリサイズする関数
def load_and_resize_image(file_name, width, height):
    image_path = os.path.join(image_folder, file_name)
    img = Image.open(image_path)
    img = ImageOps.fit(img, (width, height), method=Image.LANCZOS)
    return img

# トライアルごとの出力画像を生成する関数
def generate_images_for_trials(best_trials, output_dict):
    for i, trial in enumerate(best_trials):
        print(f"Trial {trial.number}: {trial.values}")
        print("Order:", trial.params)

        # 順番通りに要素の情報を取得してリストに格納
        selected_order = [trial.params[f"order_{j}"] for j in range(len(trial.params))]
        images = []

        print("\n選ばれた要素の情報（順番通り）:")
        for idx in selected_order:
            if idx < len(output_dict['best_frame_for_pairs']):
                # ペアの場合
                pair_info = output_dict['best_frame_for_pairs'][idx]
                print(f"ペア: {pair_info['image_files']} - 撮影日時（平均）: {pair_info['average_date_time']} - 使用フレーム: {pair_info['used_frame']}")
                
                # ペアの画像を順番にリサイズして追加
                img1 = load_and_resize_image(pair_info['image_files'][0], target_width, target_height)
                img2 = load_and_resize_image(pair_info['image_files'][1], target_width, target_height)
                images.append(img1)
                images.append(img2)

            else:
                # 残りの画像の場合
                remaining_idx = idx - len(output_dict['best_frame_for_pairs'])
                file_name = list(output_dict['other_images'].keys())[remaining_idx]
                image_info = output_dict['other_images'][file_name]
                print(f"画像: {file_name} - 撮影日時: {image_info['date_time']} - 使用フレーム: {image_info['used_frame']}")
                
                # 画像をリサイズして追加
                img = load_and_resize_image(file_name, target_width, target_height)
                images.append(img)

        # 横に並べるので合計幅を計算
        total_width = len(images) * target_width
        combined_image = Image.new('RGB', (total_width, target_height))

        # 各画像を順番に右に並べる
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += target_width

        # 結果の画像を保存
        combined_image_path = os.path.join(output_folder, f"combined_trial_{trial.number}.jpg")
        combined_image.save(combined_image_path)
        print(f"Trial {trial.number} の画像を保存しました: {combined_image_path}")

# 最終世代の上位50個体の情報をbest.jsonとして書き出す関数
def save_best_to_json(best_trials, output_dict, file_name="best.json"):
    results = []  # 出力結果を格納するリスト

    for trial in best_trials:
        selected_order = [trial.params[f"order_{i}"] for i in range(len(trial.params))]

        for idx in selected_order:
            if idx < len(output_dict['best_frame_for_pairs']):
                # ペアの場合
                pair_info = output_dict['best_frame_for_pairs'][idx]
                result = {
                    "photo_names": pair_info['image_files'],  # ペアの画像
                    "effect": pair_info['used_frame']  # 使用フレーム
                }
            else:
                # 残りの画像の場合
                remaining_idx = idx - len(output_dict['best_frame_for_pairs'])
                file_name_single = list(output_dict['other_images'].keys())[remaining_idx]
                image_info = output_dict['other_images'][file_name_single]
                result = {
                    "photo_names": [file_name_single],  # 単一の画像
                    "effect": image_info['used_frame']  # 使用フレーム
                }
            results.append(result)

    # JSONファイルに書き出す
    with open(file_name, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"上位個体の情報を {file_name} に書き出しました。")

# パラメータ設定と最適化
k = 7
sampler = NSGAIIISampler(population_size=400, mutation_prob=0.01, crossover=UniformCrossover(crossover_prob=0.5))
study = optuna.create_study(sampler=sampler, directions=["minimize", "maximize", "minimize"])
study.optimize(objective, n_trials=20000)

# フォルダのパス
image_folder = "images"
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# 画像のサイズ
target_width = 1000
target_height = 667

# 最終世代の上位50個体の解を取得して画像生成および出力
best_trials = get_final_generation_best_trials(study, top_n=50)
generate_images_for_trials(best_trials, output_dict)

# 上位50個体の情報をbest.jsonとして書き出し
save_best_to_json(best_trials, output_dict)

# 上位50体の3次元目的関数値をプロットして保存する関数
def plot_3d_objective_values(best_trials, file_name="objectives_plot.png"):
    # 3つの目的関数の値をそれぞれリストに格納
    obj1_values = [trial.values[0] for trial in best_trials]
    obj2_values = [trial.values[1] for trial in best_trials]
    obj3_values = [trial.values[2] for trial in best_trials]

    # 3Dプロットの作成
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 散布図としてプロット
    ax.scatter(obj1_values, obj2_values, obj3_values, c='r', marker='o')

    # 軸ラベルの設定
    ax.set_xlabel('Objective 1 (obj1)')
    ax.set_ylabel('Objective 2 (obj2)')
    ax.set_zlabel('Objective 3 (obj3)')
    ax.set_title('Top 50 Trials: 3D Objective Values')

    # PNGとして保存
    plt.savefig(file_name)
    plt.close()

    print(f"3次元目的関数のプロットを {file_name} に保存しました。")

# 3次元目的関数のプロットを保存
plot_3d_objective_values(best_trials)