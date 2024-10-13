import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from tqdm import tqdm

import matplotlib.pyplot as plt

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

        masks[masks==0] = 1
        
        # Ensure the mask is 3D: (number_of_objects, height, width)
        if masks.ndim == 2:
            masks = masks[np.newaxis, :, :]  # Add an object axis if only one mask is present
        

        # Process each binary mask
        for binary_mask in masks:
            fig,ax = plt.subplots()
            ax.imshow(binary_mask)
            plt.show()
            exit()
            # Convert binary mask to RLE (Run Length Encoding)
            rle = coco_mask.encode(np.asfortranarray(binary_mask))
            rle['counts'] = rle['counts'].decode('utf-8')  # Convert RLE 'counts' from bytes to string

            # Calculate the area and bounding box of the mask
            area = int(coco_mask.area(rle))
            bbox = coco_mask.toBbox(rle).tolist()  # Bounding box as [x, y, width, height]

            # Use RLE as segmentation, no need to use frPyObjects since it's already in the proper format
            segmentation = rle

            # Append the annotation data
            annotations.append({
                "id": annotation_id,
                "image_id": image['id'],
                "category_id": category_id,
                "segmentation": segmentation,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })

            # Increment annotation ID
            annotation_id += 1

    return annotations


def main():
    # 設定
    images_dir = 'dataset/images'        # 画像フォルダのパス
    masks_dir = 'dataset/masks'          # マスクフォルダのパス
    output_file = 'output_coco.json'     # 出力JSONファイル名


    # 画像情報の取得
    images = get_image_info(images_dir)
    print(images)

    # アノテーションの生成
    annotations = get_annotations(masks_dir, images)

    # COCOフォーマットの構築
    
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": {"id": 1, "name": "shrimp", "supercategory": "shrimp"},
    }

    # JSONファイルとして保存
    with open(output_file, 'w') as f:
        json.dump(coco, f, ensure_ascii=False, indent=4)

    print(f"COCOフォーマットのJSONファイルが {output_file} に保存されました。")

if __name__ == "__main__":
    main()
