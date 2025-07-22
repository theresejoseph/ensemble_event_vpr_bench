import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import rescale
from tqdm import tqdm
from datetime import datetime

# Height and width of a single image
H = 512
W = 512
TEXT_H = 175
FONTSIZE = 30
SPACE = 50  # Space between two images


def write_labels_to_image(labels=["text1", "text2"]):
    """Creates an image with text"""
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONTSIZE)
    img = Image.new("RGB", ((W * len(labels)) + 50 * (len(labels) - 1), TEXT_H), (1, 1, 1))
    d = ImageDraw.Draw(img)
    for i, text in enumerate(labels):
        _, _, w, h = d.textbbox((0, 0), text, font=font)
        d.text(((W + SPACE) * i + W // 2 - w // 2, 1), text, fill=(0, 0, 0), font=font)
    return np.array(img)[:100]  # Remove some empty space


def draw(img, c=(0, 255, 0), thickness=20):
    """Draw a colored (usually red or green) box around an image."""
    p = np.array([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]])
    for i in range(3):
        cv2.line(img, (p[i, 0], p[i, 1]), (p[i + 1, 0], p[i + 1, 1]), c, thickness=thickness * 2)
    return cv2.line(img, (p[3, 0], p[3, 1]), (p[0, 0], p[0, 1]), c, thickness=thickness * 2)



def build_prediction_image(images_paths, preds_correct, preds_dists):
    assert len(images_paths) == len(preds_correct)
    
    labels = ["Query"]
    for i, (is_correct, dist) in enumerate(zip(preds_correct[1:], preds_dists[1:]), start=1):
        if is_correct == "GT":
            label = f"GT_{dist:.1f}m"
        else:
            label = f"Pred{i}_{dist:.1f}m"
            if is_correct is not None:
                label += f" - {is_correct}"
        labels.append(label)

    num_images = len(images_paths)
    images = [np.array(Image.open(path).convert("RGB")) for path in images_paths]

    for img, correct in zip(images, preds_correct):
        if correct is None:
            continue
        if correct == "GT":
            color = (0, 0, 255)  # Blue for GT
        else:
            color = (0, 255, 0) if correct else (255, 0, 0)
        draw(img, color)

    concat_image = np.ones([H, (num_images * W) + ((num_images - 1) * SPACE), 3])
    rescaleds = [
        rescale(i, [min(H / i.shape[0], W / i.shape[1]), min(H / i.shape[0], W / i.shape[1]), 1])
        for i in images
    ]
    
    for i, image in enumerate(rescaleds):
        pad_width = (W - image.shape[1] + 1) // 2
        pad_height = (H - image.shape[0] + 1) // 2
        image = np.pad(image, [[pad_height, pad_height], [pad_width, pad_width], [0, 0]], constant_values=1)[:H, :W]
        concat_image[:, i * (W + SPACE) : i * (W + SPACE) + W] = image

    try:
        labels_image = write_labels_to_image(labels)
        final_image = np.concatenate([labels_image, concat_image])
    except OSError:  # Handle error in case of missing PIL ImageFont
        final_image = concat_image

    final_image = Image.fromarray((final_image * 255).astype(np.uint8))
    return final_image

def save_file_with_paths(query_path, preds_paths, positives_paths, output_path, use_labels=True):
    file_content = []
    file_content.append("Query path:")
    file_content.append(query_path + "\n")
    file_content.append("Predictions paths:")
    file_content.append("\n".join(preds_paths) + "\n")
    if use_labels:
        file_content.append("Positives paths:")
        file_content.append("\n".join(positives_paths) + "\n")
    with open(output_path, "w") as file:
        _ = file.write("\n".join(file_content))


from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from math import inf

def save_preds(predictions, eval_ds, log_dir, save_only_wrong_preds=None, use_labels=True):
    start_time = datetime.now()
    viz_dir = log_dir / f"preds_{start_time}"
    viz_dir.mkdir()
    
    

    for query_index, preds in enumerate(tqdm(predictions, desc=f"Saving preds in {viz_dir}")):
        query_path = eval_ds.queries_paths[query_index]
        list_of_images_paths = [query_path]  # [0] Query image
        preds_correct = [None]               # [0] Not used for query image
        preds_dists = [0.0]                  # [0] Query has 0 distance to itself

        query_utm = eval_ds.queries_utms[query_index]

        # Find the closest correct GT reference (not from predictions)
        closest_gt_path = None
        min_gt_dist = inf
        if use_labels:
            for gt_index in eval_ds.positives_per_query[query_index]:
                gt_utm = eval_ds.database_utms[gt_index]
                dist = euclidean(query_utm, gt_utm)
                if dist < min_gt_dist:
                    min_gt_dist = dist
                    closest_gt_path = eval_ds.database_paths[gt_index]
            print(f"min gt dist = {min_gt_dist}")
        # Process predicted images
        # Add the true closest correct reference image (from GT) at the end
        if closest_gt_path is not None:
            list_of_images_paths.append(closest_gt_path)
            preds_correct.append("GT")   # Special marker to distinguish it
            preds_dists.append(min_gt_dist)

        for pred in preds:
            pred_path = eval_ds.database_paths[pred]
            pred_utm = eval_ds.database_utms[pred]
            dist = euclidean(query_utm, pred_utm)

            list_of_images_paths.append(pred_path)
            preds_dists.append(dist)

            if use_labels:
                is_correct = pred in eval_ds.positives_per_query[query_index]
            else:
                is_correct = None
            preds_correct.append(is_correct)

        

        # Create the visualization including query, predictions, and GT
        prediction_image = build_prediction_image(list_of_images_paths, preds_correct, preds_dists)

        pred_image_path = viz_dir / f"{query_index:03d}.jpg"
        prediction_image.save(pred_image_path)

        if use_labels:
            positives_paths = [eval_ds.database_paths[idx] for idx in eval_ds.positives_per_query[query_index]]
        else:
            positives_paths = None
        save_file_with_paths(
            query_path=list_of_images_paths[0],
            preds_paths=list_of_images_paths[1:],
            positives_paths=positives_paths,
            output_path=viz_dir / f"{query_index:03d}.txt",
            use_labels=use_labels,
        )
