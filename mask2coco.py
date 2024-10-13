import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from tqdm import tqdm


def load_categories(categories_file):
    with open(categories_file, 'r') as f:
        categories = json.load(f)
    category_map = {cat['id']: cat['name'] for cat in categories}
    return categories, category_map

def get_image_info(images_dir):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    image_files.sort()
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


def save_json(images_dir,masks_dir,output_file):
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



def check(annFile, dataset,img_name):
    # # アノテーションファイルのパスを指定
    # annFile = 'path/to/annotations.json'  # ここを実際のアノファイルのパスに置き換えてください
    # COCOオブジェクトを初期化
    coco = COCO(annFile)

    # COCOのカテゴリとスーパーカテゴリを表示
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('Supercategories: \n{}\n'.format(' '.join(nms)))

    # 'shrimp' カテゴリのカテゴリIDを取得
    catIds = coco.getCatIds(catNms=['shrimp'])
    print(f"Category IDs for 'shrimp': {catIds}")

    # 'shrimp' カテゴリを含む全ての画像IDを取得
    imgIds = coco.getImgIds(catIds=catIds)
    print(f"Total images with 'shrimp' category: {len(imgIds)}")

    if not imgIds:
        print("指定されたカテゴリに対応する画像が見つかりません。")
        exit()

    # ランダムに画像IDを選択（例として最初の画像IDを選択）
    # selected_img_id = 1  # または np.random.choice(imgIds) でランダム選択
    print(imgIds)
    selected_img_id=1

    # 画像情報を取得
    img = coco.loadImgs(selected_img_id)[0]

    # 画像ファイルのパスを構築
    fn = os.path.join("static","dataset", dataset, img['file_name'])
    if not os.path.exists(fn):
        print(f"画像ファイルが見つかりません: {fn}")
        exit()

    # 画像を読み込む（BGR形式からRGB形式に変換）
    I = cv2.imread(fn)
    if I is None:
        print(f"画像の読み込みに失敗しました: {fn}")
        exit()
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    # Matplotlibで画像を表示
    plt.figure(figsize=(12, 8))
    plt.imshow(I)
    plt.axis('off')  # 軸を非表示にする

    # アノテーションIDを取得
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
    anns = coco.loadAnns(annIds)


    # アノテーションを画像上に表示
    coco.showAnns(anns)

    # 保存先ディレクトリを指定（存在しない場合は作成）
    output_dir = "annotated_images"
    os.makedirs(output_dir, exist_ok=True)

    # 保存するファイル名を指定
    output_filename = os.path.join("static","tmp_annotato", dataset, f"{img['file_name']}")

    # 画像を保存
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    return output_filename



def main():
    
    # 設定
    images_dir = 'static/dataset/shrimp'        # 画像フォルダのパス
    masks_dir = 'static/annotation_info/shrimp'          # マスクフォルダのパス
    output_file = 'output_coco2.json'     # 出力JSONファイル名

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
    

if __name__ == "__main__":
    main()
