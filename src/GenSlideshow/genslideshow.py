import json
import os
from moviepy.editor import (
    VideoClip, ImageClip, CompositeVideoClip, concatenate_videoclips, vfx
)
from PIL import Image, ImageFilter
import numpy as np

def resize_and_fit_image(image_path, W, H):
    image = ImageClip(image_path)
    image_ratio = image.w / image.h
    frame_ratio = W / H

    if image_ratio > frame_ratio:
        image = image.resize(width=W)
    else:
        image = image.resize(height=H)

    x = (W - image.w) / 2
    y = (H - image.h) / 2
    return image.set_position((x, y))

def nothing_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080
    image = resize_and_fit_image(os.path.join('photo', photo_names[0]), W , H )  # Match zoomin size
    return image.set_duration(duration)

def zoomin_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080
    image = resize_and_fit_image(os.path.join('photo', photo_names[0]), W , H )
    image = image.set_duration(duration)

    def zoom(t):
        scale = 1 + 0.3 * t / duration
        return scale

    zoomed = image.fx(vfx.resize, zoom)
    return zoomed

def fade_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080
    image = resize_and_fit_image(os.path.join('photo', photo_names[0]), W , H )  # Match zoomin size
    image = image.set_duration(duration).crossfadein(duration / 2)
    return image

def resize_image_to_height(image_path, H):
    """画像の高さを指定し、アスペクト比を維持してリサイズします。"""
    image = ImageClip(image_path)
    image = image.resize(height=H)
    return image

def resize_image_to_width(image_path, W):
    """画像の幅を指定し、アスペクト比を維持してリサイズします。"""
    image = ImageClip(image_path)
    image = image.resize(width=W)
    return image

def yoko_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080
    image = resize_image_to_height(os.path.join('photo', photo_names[0]), H )  # Resize to match zoomin
    image_w, image_h = image.size
    image = image.set_duration(duration)
    
    def position(t):
        x_start = W
        x_end = -image_w
        x = x_start + (x_end - x_start) * (t / duration)
        y = (H - image_h) / 2
        return x, y

    animated_clip = image.set_position(position)
    final_clip = CompositeVideoClip([animated_clip], size=(W, H))
    return final_clip

def rise_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080
    image = resize_image_to_width(os.path.join('photo', photo_names[0]), W )  # Resize to match zoomin
    image_w, image_h = image.size
    image = image.set_duration(duration)

    def position(t):
        y_start = H
        y_end = -image_h
        y = y_start + (y_end - y_start) * (t / duration)
        x = (W - image_w) / 2
        return x, y

    animated_clip = image.set_position(position)
    final_clip = CompositeVideoClip([animated_clip], size=(W, H))
    return final_clip

def gousei_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080

    if len(photo_names) < 2:
        raise ValueError("Gousei_effect requires at least two photos.")

    image1 = resize_and_fit_image(os.path.join('photo', photo_names[0]), W , H ).set_duration(duration)
    image2 = resize_and_fit_image(os.path.join('photo', photo_names[1]), W , H ).set_duration(duration)

    image1 = image1.set_opacity(0.5)
    image2 = image2.set_opacity(0.5)

    final_clip = CompositeVideoClip([image1, image2], size=(W, H))
    return final_clip

def nothing2_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080  # Target frame size

    if len(photo_names) < 2:
        raise ValueError("nothing2_effect requires at least two photos.")

    # Calculate the target height for each image to perfectly fit half of the frame
    target_height = H / 2  # Each image must fit exactly half the height of the frame

    # For 3:2 images, calculate the corresponding width to maintain aspect ratio
    target_width = target_height * (3 / 2)  # Maintain 3:2 aspect ratio

    # If the calculated width is greater than the frame width, adjust the dimensions
    if target_width > W:
        target_width = W  # Cap the width at the frame width
        target_height = W * (2 / 3)  # Adjust height to maintain 3:2 aspect ratio

    # Resize the images to the calculated target size
    image1 = ImageClip(os.path.join('photo', photo_names[0])).resize(height=target_height)
    image2 = ImageClip(os.path.join('photo', photo_names[1])).resize(height=target_height)

    # Position image1 at the top and image2 directly below it
    image1 = image1.set_position(("center", 0)).set_duration(duration)
    image2 = image2.set_position(("center", target_height)).set_duration(duration)

    # Combine the two images into a final video clip that perfectly fits the frame
    final_clip = CompositeVideoClip([image1, image2], size=(W, H))

    return final_clip

def pan_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080

    image = ImageClip(os.path.join('photo', photo_names[0]))
    image = image.resize(height=H )  # Match zoomin size
    image_w, image_h = image.size
    image = image.set_duration(duration)

    def position(t):
        x = -(image_w - W) * (t / duration)
        return x, (H - image_h) / 2

    animated_clip = image.set_position(position)
    final_clip = CompositeVideoClip([animated_clip], size=(W, H))
    return final_clip

def rotate_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080

    image = resize_and_fit_image(os.path.join('photo', photo_names[0]), W , H )  # Match zoomin size
    rotated = image.set_duration(duration).rotate(lambda t: 360 * t / duration, expand=False)
    return rotated

def zoomblur_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080

    image = resize_and_fit_image(os.path.join('photo', photo_names[0]), W , H )  # Match zoomin size
    image = image.set_duration(duration)

    def blur_frame(get_frame, t):
        frame = get_frame(t)
        pil_image = Image.fromarray(frame)
        blur_radius = max(5 - 5 * (t / duration), 0.1)  # ぼかしから元の画像に戻す
        blurred_pil = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return np.array(blurred_pil)

    blurred = image.fl(blur_frame)
    return blurred

def splitscreen_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080

    if len(photo_names) < 2:
        raise ValueError("SplitScreen_effect requires at least two photos.")

    image1 = resize_and_fit_image(os.path.join('photo', photo_names[0]), W  / 2, H )  # Match zoomin size
    image2 = resize_and_fit_image(os.path.join('photo', photo_names[1]), W  / 2, H )  # Match zoomin size

    image1 = image1.set_position((0, 0)).set_duration(duration)
    image2 = image2.set_position((W / 2, 0)).set_duration(duration)

    final_clip = CompositeVideoClip([image1, image2], size=(W, H))
    return final_clip

def colorfade_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080

    image_color = resize_and_fit_image(os.path.join('photo', photo_names[0]), W , H ).set_duration(duration)
    image_bw = image_color.fx(vfx.blackwhite)

    transition = CompositeVideoClip([
        image_bw.crossfadeout(duration / 2),
        image_color.set_start(duration / 2).crossfadein(duration / 2)
    ], size=(W, H))

    return transition.set_duration(duration)

def crossfade_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080

    if len(photo_names) < 2:
        raise ValueError("Crossfade_effect requires at least two photos.")

    # Resize images to fit the screen, no oversizing
    image1 = resize_and_fit_image(os.path.join('photo', photo_names[0]), W, H).set_duration(duration)
    image2 = resize_and_fit_image(os.path.join('photo', photo_names[1]), W, H).set_duration(duration)

    # Apply crossfade effect, images will not exceed the screen size
    crossfaded = CompositeVideoClip([
        image1.crossfadeout(duration / 2),
        image2.set_start(duration / 2).crossfadein(duration / 2)
    ], size=(W, H))

    return crossfaded.set_duration(duration)

def slideswap_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080

    if len(photo_names) < 2:
        raise ValueError("SlideSwap_effect requires at least two photos.")

    # Resize images to fit the screen, no oversizing
    image1 = resize_and_fit_image(os.path.join('photo', photo_names[0]), W, H)
    image2 = resize_and_fit_image(os.path.join('photo', photo_names[1]), W, H)

    # Define the slide positions within the screen bounds
    def position1(t):
        x = -W * t / duration  # Slide from left to right
        return x, 0

    def position2(t):
        x = W - W * t / duration  # Slide from right to left
        return x, 0

    # Set position and duration
    image1 = image1.set_position(position1).set_duration(duration)
    image2 = image2.set_position(position2).set_duration(duration)

    # Combine both clips into one
    final_clip = CompositeVideoClip([image1, image2], size=(W, H))
    return final_clip


def flip_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080

    if len(photo_names) < 2:
        raise ValueError("Flip_effect requires at least two photos.")

    half_duration = duration / 2

    image1 = resize_and_fit_image(os.path.join('photo', photo_names[0]), W , H ).set_duration(half_duration)
    image2 = resize_and_fit_image(os.path.join('photo', photo_names[1]), W , H ).set_duration(half_duration)

    # 横方向に反転させる
    clip1 = image1.fx(vfx.mirror_x)
    clip2 = image2

    # クロスフェードで切り替える
    clip1 = clip1.crossfadeout(half_duration)
    clip2 = clip2.set_start(half_duration).crossfadein(half_duration)

    final_clip = concatenate_videoclips([clip1, clip2], method="compose")
    return final_clip.set_duration(duration)

def pictureinpicture_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080

    if len(photo_names) < 2:
        raise ValueError("PictureInPicture_effect requires at least two photos.")

    background = resize_and_fit_image(os.path.join('photo', photo_names[0]), W , H ).set_duration(duration)
    foreground = resize_and_fit_image(os.path.join('photo', photo_names[1]), W / 3 , H / 3 ).set_duration(duration)

    x = W - foreground.w - 20
    y = H - foreground.h - 20

    # 前景画像にズームイン効果を追加
    foreground = foreground.fx(vfx.resize, lambda t: 1 + 0.1 * t / duration)
    foreground = foreground.set_position((x, y))

    final_clip = CompositeVideoClip([background, foreground], size=(W, H))
    return final_clip

def mosaic_effect(photo_names):
    duration = 4.0
    W, H = 1920, 1080

    images = []
    positions = [(0, 0), (W / 2, 0), (0, H / 2), (W / 2, H / 2)]
    sizes = [(W  / 2, H  / 2)] * 4

    for i in range(4):
        idx = i % len(photo_names)
        image = resize_and_fit_image(os.path.join('photo', photo_names[idx]), sizes[i][0], sizes[i][1])
        image = image.set_position(positions[i]).set_duration(duration)
        images.append(image)

    final_clip = CompositeVideoClip(images, size=(W, H))
    return final_clip

# エフェクト名と関数のマッピング
effect_functions = {
    "nothing": nothing_effect,
    "Fade_effect": fade_effect,
    "Zoomin_effect": zoomin_effect,
    "Yoko_effect": yoko_effect,
    "Rise_effect": rise_effect,
    "Gousei_effect": gousei_effect,
    "nothing2": nothing2_effect,
    "Pan_effect": pan_effect,
    "Rotate_effect": rotate_effect,
    "ZoomBlur_effect": zoomblur_effect,
    "SplitScreen_effect": splitscreen_effect,
    "ColorFade_effect": colorfade_effect,
    "Crossfade_effect": crossfade_effect,
    "SlideSwap_effect": slideswap_effect,
    "Flip_effect": flip_effect,
    "PictureInPicture_effect": pictureinpicture_effect,
    "Mosaic_effect": mosaic_effect,
}

# JSONデータの読み込み
with open('input.json', 'r') as f:
    data = json.load(f)

video_clips = []

# 各項目を処理
for item in data:
    photo_names = item["photo_names"]
    effect_name = item["effect"]

    if effect_name in effect_functions:
        effect_func = effect_functions[effect_name]

        # 必要な写真の数を確認
        required_photos = 2 if effect_name in [
            "Gousei_effect", "nothing2", "SplitScreen_effect",
            "Crossfade_effect", "SlideSwap_effect", "Flip_effect",
            "PictureInPicture_effect", "Mosaic_effect"
        ] else 1

        if len(photo_names) < required_photos:
            print(f"Error: {effect_name} requires at least {required_photos} photos.")
            continue

        try:
            clip = effect_func(photo_names)
            video_clips.append(clip)
        except Exception as e:
            print(f"Error processing {effect_name}: {e}")
    else:
        print(f"Unknown effect: {effect_name}")

if video_clips:
    # 全てのビデオクリップを結合
    final_video = concatenate_videoclips(video_clips, method="compose")
    final_video.write_videofile("slideshow.mp4", fps=24)
else:
    print("No video clips to process.")
