import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2

def load_categories(categories_file):
    with open(categories_file, 'r') as f:
        categories = json.load(f)
    category_map = {cat['id']: cat['name'] for cat in categories}
    return categories, category_map

def get_image_info(images_dir):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    images = []
    for idx, file_name in enumerate(image_files, 1):
        file_path = os.path.join(images_dir, file_name)
        with Image.open(file_path) as img:
            width, height = img.size
        images.append({
            "id": idx,
            "file_name": file_name,
            "width": width,
            "height": height
        })
    return images

import os
import numpy as np
from tqdm import tqdm
from pycocotools import mask as coco_mask

def get_annotations(masks_dir, images, category_id=0):
    annotations = []
    annotation_id = 1

    for image in tqdm(images, desc="Processing annotations"):
        # Construct file path for the corresponding .npy mask file
        mask_path = os.path.join(masks_dir, os.path.splitext(image['file_name'])[0] + ".npy")

        # Check if the mask file exists
        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found for {image['file_name']}")
            continue

        # Load masks from the .npy file
        masks = np.load(mask_path)
        # print(np.unique(masks))
        # exit()
        # masks[masks==0] = 1
        
        # Ensure the mask is 3D: (number_of_objects, height, width)
        if masks.ndim == 2:
            masks = masks[np.newaxis, :, :]  # Add an object axis if only one mask is present
        

        # Process each binary mask
        for binary_mask in masks:
            # Convert binary mask to uint8 format
            binary_mask_uint8 = binary_mask.astype(np.uint8)

            # Find contours using OpenCV
            contours, _ = cv2.findContours(binary_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If no contours are found, skip this mask
            if not contours:
                continue

            # Encode segmentation in COCO format (list of polygons)
            segmentation = []
            for contour in contours:
                # Flatten the contour array and convert to a list
                segmentation.append(contour.flatten().tolist())

                # Calculate the area of the mask
                area = float(np.sum(binary_mask))

                # Compute bounding box [x, y, width, height]
                y_indices, x_indices = np.where(binary_mask)
                if y_indices.size == 0 or x_indices.size == 0:
                    continue  # Skip if the mask is empty

                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                bbox = [
                    int(x_min),
                    int(y_min),
                    int(x_max - x_min + 1),
                    int(y_max - y_min + 1)
                ]

                # Create the annotation dictionary
                annotation = {
                    'id': annotation_id,
                    'image_id': image['id'],
                    'category_id': category_id,
                    'segmentation': segmentation,
                    'area': area,
                    'bbox': bbox,
                    'iscrowd': 0
                }

                # Append the annotation to the list
                annotations.append(annotation)
                annotation_id += 1

    return annotations

def main():
    
    # 設定
    images_dir = 'dataset/images'        # 画像フォルダのパス
    masks_dir = 'dataset/masks'          # マスクフォルダのパス
    output_file = 'output_coco2.json'     # 出力JSONファイル名
    # A = np.load(os.path.join(masks_dir,"_cars.npy"))
    # B = np.load(os.path.join(masks_dir,"cars2.npy"))
    # C = np.array([A,B])
    # np.save("cars",C)
    # exit()

    # 画像情報の取得
    images = get_image_info(images_dir)

    # アノテーションの生成
    annotations = get_annotations(masks_dir, images)

    # COCOフォーマットの構築
    
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "shrimp", "supercategory": "shrimp"},]
    }

    # JSONファイルとして保存
    with open(output_file, 'w') as f:
        json.dump(coco, f, ensure_ascii=False, indent=4)

    print(f"COCOフォーマットのJSONファイルが {output_file} に保存されました。")

if __name__ == "__main__":
    main()
