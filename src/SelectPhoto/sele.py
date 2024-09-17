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

# JSONファイルから画像情報を読み込む関数
def load_image_info_from_json(json_file_name="image_info.json"):
    with open(json_file_name, 'r', encoding='utf-8') as f:
        image_description_list = json.load(f)
    print(f"{json_file_name} から画像情報を読み込みました。")
    return image_description_list

# JSONファイルから画像情報を読み込み
image_description_list = load_image_info_from_json()


frame_list = [
    {
        "frame_name": "nothing",
        "frame_description": "A simple effect that displays the image without any special animations or transitions.",
        "photo_count": 1
    },
    {
        "frame_name": "Fade_effect",
        "frame_description": "An effect where the image gradually fades in, creating a smooth and gentle introduction. Suitable for revealing landscapes or scenes that appear suddenly.",
        "photo_count": 1
    },
    {
        "frame_name": "Zoomin_effect",
        "frame_description": "This effect gradually zooms in on a photo over time, drawing the eye to specific details within the photo. For example, you can zoom in on a distant lighthouse or a flower lit by the setting sun in a wide landscape photo to highlight the beauty and detail of the moment. This effect is ideal for emphasizing important subjects or fine details.",
        "photo_count": 1
    },
    {
        "frame_name": "Yoko_effect",
        "frame_description": "An effect where the image slides in horizontally from the side, reminiscent of a moving ship. Suitable for images of vehicles or scenes with horizontal motion.",
        "photo_count": 1
    },
    {
        "frame_name": "Rise_effect",
        "frame_description": "This effect creates the illusion that the photo is slowly rising from the bottom of the screen and taking off into the sky. For example, it is perfect for photos with a theme of ascent or flight, such as a hot air balloon floating against a blue sky or the silhouette of a bird taking off at dusk. The effect is expected to add a sense of movement in the background and the expanse of the sky, adding dynamism to the still image.",
        "photo_count": 1
    },
    {
        "frame_name": "nothing2",
        "frame_description": "A simple effect that displays two images side by side without any special animations. Ideal for comparing two related photos or showcasing complementary scenes together.",
        "photo_count": 2
    },
    {
        "frame_name": "Pan_effect",
        "frame_description": "This effect pans across a wide landscape photo, simulating the movement of the eye across a vast scene. Ideal for panoramic shots or cityscapes.",
        "photo_count": 1
    },
    {
        "frame_name": "Rotate_effect",
        "frame_description": "An image rotates into view, adding a dynamic twist. Suitable for action shots or photos with a sense of motion.",
        "photo_count": 1
    },
    {
        "frame_name": "Crossfade_effect",
        "frame_description": "An effect where one image gradually fades out while the next image fades in, creating a smooth transition. Ideal for related scenes or storytelling sequences.",
        "photo_count": 2
    },
    {
        "frame_name": "SlideSwap_effect",
        "frame_description": "One image slides out while the next image slides in from the opposite side. Suitable for showing progression or changes between two photos.",
        "photo_count": 2
    }
]

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

def select_top_pairs(similar_pairs, top_n=10):
    # 重複を避けるために選ばれた要素を追跡するセット
    selected_elements = set()
    # 最終的に選ばれるペアを格納するリスト
    top_pairs = []
    
    # ペアを類似度順にソート（降順）
    sorted_pairs = sorted(similar_pairs, key=lambda x: x[0], reverse=True)
    
    # ペアを順番に確認し、どちらか一方が既に選ばれた要素と一致しない場合のみ追加
    for pair in sorted_pairs:
        _, elem1, elem2 = pair
        if elem1 not in selected_elements and elem2 not in selected_elements:
            top_pairs.append(pair)
            selected_elements.update([elem1, elem2])  # 両方の要素をセットに追加
        if len(top_pairs) == top_n:  # 指定された数だけペアが選ばれたら終了
            break

    return top_pairs

# 類似度の上位3つのペアを選択
top_3_pairs = select_top_pairs(similar_pairs, top_n=10)

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

# 5. フォトカウントが2のフレームに対して類似度を計算し、ランキング選択で最も高いフレームを選択
photo_count_2_frames = [frame for frame in frame_list if frame['photo_count'] == 2]
photo_count_2_frame_embeddings = [frame_embeddings[frame_list.index(frame)] for frame in photo_count_2_frames]

best_frames_for_pairs = []
for pair_info in selected_pairs_info:
    similarities_with_avg = cosine_similarity([pair_info['average_embedding']], photo_count_2_frame_embeddings)
    
    # ランキング計算（降順に類似度をソート）
    ranked_indices = np.argsort(similarities_with_avg[0])[::-1]  # 高い順に並べ替え
    
    # 選択確率を計算 (ランクが高いほど選ばれやすいように確率分布を作る)
    rank_probabilities = np.exp(-np.arange(len(ranked_indices)))  # ランクに基づく指数関数的な重み
    rank_probabilities /= rank_probabilities.sum()  # 正規化して確率に変換
    
    # ランダムにフレームを選択
    best_frame_index = np.random.choice(ranked_indices, p=rank_probabilities)
    best_frame = photo_count_2_frames[best_frame_index]['frame_name']
    
    pair_info['used_frame'] = best_frame  # フレーム名を追加
    best_frames_for_pairs.append(pair_info)

# 6. 残りのimage_descriptionとphoto_countが1のframe_descriptionの類似度を計算し、ランキング選択で最も類似度が高いフレームを選択
used_images = {file for pair in best_frames_for_pairs for file in pair['image_files']}  # 使用済みの画像ファイル名をセットに保存
remaining_images = [i for i in range(num_images) if image_description_list[i]['file_name'] not in used_images]
image_to_frame_map = {}

# photo_countが1のフレームを抽出
photo_count_1_frames = [frame for frame in frame_list if frame['photo_count'] == 1]
photo_count_1_frame_embeddings = [frame_embeddings[frame_list.index(frame)] for frame in photo_count_1_frames]

for img_idx in remaining_images:
    # 類似度を計算（画像とphoto_countが1のフレームの間で）
    similarities = cosine_similarity([image_embeddings[img_idx]], photo_count_1_frame_embeddings)
    
    # ランキング計算（降順に類似度をソート）
    ranked_indices = np.argsort(similarities[0])[::-1]  # 高い順に並べ替え
    
    # 選択確率を計算 (ランクに基づく指数関数的な重み)
    rank_probabilities = np.exp(-np.arange(len(ranked_indices)))
    rank_probabilities /= rank_probabilities.sum()  # 正規化して確率に変換
    
    # ランダムにフレームを選択
    best_frame_idx_in_photo_count_1 = np.random.choice(ranked_indices, p=rank_probabilities)
    best_frame = photo_count_1_frames[best_frame_idx_in_photo_count_1]['frame_name']
    
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

# クラスタリングとベクトル・ファイル名の抽出
def extract_embeddings_and_filenames(output_dict):
    embeddings = []
    file_names = []
    
    # ペアの画像データ
    for pair_info in output_dict['best_frame_for_pairs']:
        embeddings.append(pair_info['average_embedding'])
        file_names.append(pair_info['image_files'])  # ペアのファイル名も格納
    
    # その他の画像データ
    for file_name, info in output_dict['other_images'].items():
        embeddings.append(info['embedding'])
        file_names.append([file_name])  # 単一のファイル名をリストにして格納
    
    return embeddings, file_names

# クラスタリングの実行
def cluster_descriptions(output_dict, k=8):
    embeddings, file_names = extract_embeddings_and_filenames(output_dict)

    # KMeansでクラスタリングを実施
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)

    # クラスタごとのファイル名を管理
    clusters = {i: [] for i in range(k)}
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)  # インデックスを保存して後で使用
    
    return clusters


# エリート個体を選択する戦略（上位50%を選択）
def elite_population_selection_strategy(study: optuna.Study, trials: list[optuna.trial.FrozenTrial]) -> list[optuna.trial.FrozenTrial]:
    # トライアルを目的関数の値でソートし、上位50%を選択
    num_elite = len(trials) // 4  # 上位50%を選択
    elite_trials = sorted(trials, key=lambda t: t.values)[:num_elite]
    return elite_trials

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
def merge_and_count(arr, temp_arr, left, right):
    if left == right:
        return 0

    mid = (left + right) // 2
    inv_count = 0
    
    inv_count += merge_and_count(arr, temp_arr, left, mid)
    inv_count += merge_and_count(arr, temp_arr, mid + 1, right)
    inv_count += merge(arr, temp_arr, left, mid, right)

    return inv_count

def merge(arr, temp_arr, left, mid, right):
    i = left    # 左側部分のインデックス
    j = mid + 1 # 右側部分のインデックス
    k = left    # 結合された配列のインデックス
    inv_count = 0

    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            i += 1
        else:
            temp_arr[k] = arr[j]
            inv_count += (mid - i + 1)
            j += 1
        k += 1

    while i <= mid:
        temp_arr[k] = arr[i]
        i += 1
        k += 1

    while j <= right:
        temp_arr[k] = arr[j]
        j += 1
        k += 1

    for i in range(left, right + 1):
        arr[i] = temp_arr[i]

    return inv_count

def count_inversions(arr):
    n = len(arr)
    temp_arr = [0] * n
    return merge_and_count(arr, temp_arr, 0, n - 1)

# 類似度のキャッシュを作成
def calculate_similarity_cache(embeddings):
    num_embeddings = len(embeddings)
    similarity_cache = np.zeros((num_embeddings, num_embeddings))

    # 全てのペアのコサイン類似度を計算しキャッシュに保存
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            similarity_cache[i][j] = similarity
            similarity_cache[j][i] = similarity  # 対称行列なので両方に値を入れる

    return similarity_cache

# 隣接類似性を計算する関数（事前計算したコサイン類似度を使用）
def adjacent_similarity(order, similarity_cache):
    similarity_sum = 0
    for i in range(0, len(order) - 1, 2):  # インデックスが飛び飛びになるように計算
        sim = similarity_cache[order[i]][order[i + 1]] ** 5
        similarity_sum += sim
    return similarity_sum

# ベクトル埋め込みとファイル名の取得関数
def get_embeddings_and_dates(output_dict):
    dates = []
    embeddings = []

    # ペアのベクトルと日時をリストに追加
    for pair_info in output_dict['best_frame_for_pairs']:
        avg_date = datetime.strptime(pair_info['average_date_time'], '%Y:%m:%d %H:%M:%S')
        dates.append(avg_date)
        embeddings.append(pair_info['average_embedding'])

    # 残りの画像のベクトルと日時をリストに追加
    for file_name, info in output_dict['other_images'].items():
        date = datetime.strptime(info['date_time'], '%Y:%m:%d %H:%M:%S')
        dates.append(date)
        embeddings.append(info['embedding'])

    return embeddings, dates

# 公平性を計算する関数（クラスタごとの使用回数は最大1に制限）
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
    # 遺伝子の数を50に設定
    num_genes = 50

    # 選択肢の数
    num_elements = len(output_dict['best_frame_for_pairs']) + len(output_dict['other_images'])

    # 50個の設計変数（遺伝子）を指定された範囲から選択
    order = [trial.suggest_int(f"order_{i}", 0, num_elements - 1) for i in range(num_genes)]

    # 目的関数
    obj1 = count_inversions([dates[i] for i in order])  # 日付の反転数を計算
    obj2 = -adjacent_similarity(order, similarity_cache)  # キャッシュされた隣接類似度を参照して計算
    obj3 = fairness(order, clusters)  # 公平性を計算

    penalty = calculate_penalty(order)  # 重複ペナルティの計算

    # ペナルティの適用
    obj1 += 100 * penalty
    obj2 += 100 * penalty
    obj3 += 100 * penalty

    # トライアルの世代情報を保存
    trial.set_user_attr("generation", trial.number // 400)

    return obj1, obj2, obj3

# クラスタリングを事前に実行
clusters = cluster_descriptions(output_dict, k=8)

def get_top_best_trials(study, top_n=15):
    # Optunaが保持している非優越ソートされた最良の個体リストを取得
    best_trials = study.best_trials
    
    # 上位top_n個の個体を返す
    return best_trials[:top_n]

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

# 上位個体の情報を各トライアル名で別々にbest.jsonとして書き出す関数
def save_best_to_json(best_trials, output_dict, output_folder="output_json"):
    # フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)

    for trial in best_trials:
        results = []  # 出力結果を格納するリスト
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

        # トライアル名を使用してJSONファイルを作成
        file_name = os.path.join(output_folder, f"trial_{trial.number}.json")
        with open(file_name, "w") as json_file:
            json.dump(results, json_file, indent=4)

        print(f"Trial {trial.number} の情報を {file_name} に書き出しました。")

# パラメータ設定と最適化
# NSGA-IIIを回す前に事前にベクトル埋め込みを計算しておく
embeddings, dates = get_embeddings_and_dates(output_dict)

# 事前にコサイン類似度を計算してキャッシュに保存
similarity_cache = calculate_similarity_cache(embeddings)

k = 8
sampler = NSGAIIISampler(population_size=400, mutation_prob=0.03, crossover=UniformCrossover(crossover_prob=0.9))
study = optuna.create_study(sampler=sampler, directions=["minimize", "minimize", "minimize"])
study.optimize(objective, n_trials=160000)

# フォルダのパス
image_folder = "images"
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# 画像のサイズ
target_width = 1000
target_height = 667

# 最終世代の上位50個体の解を取得して画像生成および出力
best_trials = get_top_best_trials(study, top_n=15)
# best_trials = get_final_generation_best_trials(study, top_n=50)
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
    ax.set_title('Top15 Trials: 3D Objective Values')

    # PNGとして保存
    plt.savefig(file_name)
    plt.close()

    print(f"3次元目的関数のプロットを {file_name} に保存しました。")

# 全ての目的関数のペアで2次元プロットを作成し、指定ディレクトリに保存する関数
def plot_all_2d_objective_pairs(best_trials, output_folder="objective_plots"):
    # フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)

    # 目的関数の数を取得（最初のトライアルから）
    num_objectives = len(best_trials[0].values)
    
    # トライアル番号を取得
    trial_numbers = [trial.number for trial in best_trials]

    # 全ての目的関数のペアをプロット
    for i in range(num_objectives):
        for j in range(i + 1, num_objectives):
            obj_i_values = [trial.values[i] for trial in best_trials]
            obj_j_values = [trial.values[j] for trial in best_trials]

            # 2Dプロットの作成
            fig, ax = plt.subplots()
            ax.scatter(obj_i_values, obj_j_values, c='r', marker='o')

            # 各点にトライアル番号を表示
            for k, trial_num in enumerate(trial_numbers):
                ax.text(obj_i_values[k], obj_j_values[k], f'Trial {trial_num}', fontsize=9)

            # 軸ラベルの設定
            ax.set_xlabel(f'Objective {i + 1} (obj{i + 1})')
            ax.set_ylabel(f'Objective {j + 1} (obj{j + 1})')
            ax.set_title(f'2D Plot of Objective {i + 1} vs Objective {j + 1}')

            # ファイル名を設定して保存
            file_name = f"objective_{i+1}_vs_{j+1}.png"
            file_path = os.path.join(output_folder, file_name)
            plt.savefig(file_path)
            plt.close()

            print(f"目的関数 {i+1} と {j+1} の2次元プロットを {file_path} に保存しました。")

plot_all_2d_objective_pairs(best_trials, output_folder="objective_pairs_plots")

# 3次元目的関数のプロットを保存
plot_3d_objective_values(best_trials)