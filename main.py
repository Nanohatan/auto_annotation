import base64
import io
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from starlette.responses import FileResponse
import glob
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static",html = True), name="static")

templates = Jinja2Templates(directory="templates")

path = os.path.join("static", "dataset")
source_directory = os.path.join("static", "dataset")
pattern = os.path.join(source_directory, "*/") 
directories = glob.glob(pattern)
for dir in directories:
    dir_name = dir.split("/")[2]
    tmp = os.path.join("static", "tmp_annotato",dir_name)
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    tmp = os.path.join("static", "json",dir_name)
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    tmp = os.path.join("static", "annotation_info", dir_name)
    if not os.path.exists(tmp):
        os.makedirs(tmp)

UPLOAD_DIRECTORY = "uploads"
@app.get("/")
async def root():
    return FileResponse('static/index.html')
    # return RedirectResponse(url="/upload")


@app.get("/upload")
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


from pydantic import BaseModel


def show_mask(mask, ax, random_color=False, borders = False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def get_largest_mask(mask ):
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    largest_mask = mask
    if num_labels <= 1:
        largest_mask = mask
    else:
        # 各ラベルのピクセル数を計算
        label_counts = np.bincount(labels.flatten())
        label_counts[0] = 0  # 背景のラベル（0）を除外
        
        # 最も大きな連結成分のラベルを取得
        largest_label = label_counts.argmax()
        
        # 最も大きな連結成分のみを保持
        largest_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return largest_mask

def get_mask_over(mask,img_path):
    mask = mask.astype(np.uint8)
    mask = get_largest_mask(mask)
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    img = cv2.imread(img_path)
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    mask[mask==1] = 255
    return img,mask



class Coordinate(BaseModel):
    fn:str
    dataset:str
    x: float
    y: float
@app.post("/post_coordinates")
async def post_coordinates(coord: Coordinate):
    # 受け取ったデータを処理します
    image_path = os.path.join("static", "dataset", coord.dataset,coord.fn)
    x = coord.x
    y = coord.y
    # ここで必要な処理を実行（例：データベースに保存、ログ出力など）
    print(f"Received image: {image_path}, x: {x}, y: {y}")

    # 画像を開く
    with Image.open(image_path) as image:
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=np.array([[x,y]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        img,mask = get_mask_over(masks[0],image_path)
        annotato_path = os.path.join("static","tmp_annotato",coord.dataset,"tmp.jpg")
        cv2.imwrite(annotato_path, img)
        mask_fn = coord.fn.split(".")[0] + ".mask"
        np.save(os.path.join("static","tmp_annotato",coord.dataset,mask_fn),mask)

    # クライアントにレスポンスを返します
    return {"message": "Coordinates received successfully", "annotato_path":os.path.join(coord.dataset,"tmp.jpg")}

class InputPoint(BaseModel):
    fn:str
    dataset:str
    point:list
    label:list
@app.post("/post_input_point")
async def post_input_point(input_point: InputPoint):
    # 受け取ったデータを処理します
    image_path = os.path.join("static", "dataset", input_point.dataset,input_point.fn)
    # ここで必要な処理を実行（例：データベースに保存、ログ出力など）
    print(f"Received image: {image_path}")

    # 画像を開く
    with Image.open(image_path) as image:
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=np.array(input_point.point),
            point_labels=np.array(input_point.label),
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        img,mask = get_mask_over(masks[0],image_path)
        annotato_path = os.path.join("static","tmp_annotato",input_point.dataset,"tmp.jpg")
        cv2.imwrite(annotato_path, img)
        mask_fn = input_point.fn.split(".")[0] + ".mask"
        np.save(os.path.join("static","tmp_annotato",input_point.dataset,mask_fn),mask)

    # クライアントにレスポンスを返します
    return {"message": "Coordinates received successfully", "annotato_path":os.path.join(input_point.dataset,"tmp.jpg")}

@app.get("/test/{dataset}")
async def post_coordinates(request: Request,dataset:str):
    dataset_dir = os.path.join("static", "dataset", dataset)
    # datasetディレクトリ内の全JPEGファイルをリストアップ
    images = [
        f for f in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, f)) and f.lower().endswith((".jpg", ".jpeg"))
    ]
    return templates.TemplateResponse("test.html", {"request": request, "dataset":dataset, "images":images})


@app.get("/annotation")
async def show(request: Request):
    path = os.path.join("static", "dataset")
    with os.scandir(path) as entries:
            directories = [entry.name for entry in entries if entry.is_dir()]

    return templates.TemplateResponse("annotation.index.html", {"request": request, "dataset_list": directories})

@app.get("/annotation/{dataset}")
async def post_coordinates(request: Request,dataset:str):
    dataset_dir = os.path.join("static", "dataset", dataset)
    # datasetディレクトリ内の全JPEGファイルをリストアップ
    images = [
        f for f in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, f)) and f.lower().endswith((".jpg", ".jpeg"))
    ]
    images.sort()
    return templates.TemplateResponse("annotation.html", {"request": request, "dataset":dataset, "images":images})


class ImageInfo(BaseModel):
    dataset:str
    fn:str
    @property
    def original_img_path(self) -> str:
        return os.path.join("static", "dataset", self.dataset, self.fn)
    @property
    def mask_path(self) -> str:
        return os.path.join("static", "tmp_annotato", self.dataset, self.no_ext+".mask.npy")
    @property 
    def no_ext(self) -> str:
        return self.fn.split(".")[0]


def get_masked_Img():
    pass

@app.post("/save_annotation")
async def save_annotation(image_info: ImageInfo):
    mask = np.load(image_info.mask_path)
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :] 

    fn = os.path.join("static", "annotation_info", image_info.dataset, image_info.no_ext+".npy")

    if os.path.exists(fn):
        mask1 = np.load(fn)
        
        mask = np.concatenate((mask1, mask), axis=0)
    np.save(fn,mask)
    return {"image_path":image_info}

import mask2coco 


@app.post("/annotation_result")
async def get_annotation(image_info: ImageInfo):
    fn = os.path.join("static", "annotation_info", image_info.dataset, image_info.no_ext+".npy")
    masks = np.load(fn)
    json_fn = os.path.join("static", "json", image_info.dataset, image_info.no_ext+".json")
    # if not os.path.exists(json_fn):
    #     mask2coco.save_json(os.path.join("static", "dataset", image_info.dataset),
    #                         os.path.join("static", "annotation_info",image_info.dataset), json_fn)
    #     path = mask2coco.check(json_fn)
    # else:
    mask2coco.save_json(os.path.join("static", "dataset", image_info.dataset),
                        os.path.join("static", "annotation_info",image_info.dataset), json_fn)
    path = mask2coco.check(json_fn,dataset=image_info.dataset,img_name=image_info.fn)
    path = os.path.join("tmp_annotato", image_info.dataset, image_info.fn)
    return {"image_path":path}

