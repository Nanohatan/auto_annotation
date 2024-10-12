from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from starlette.responses import FileResponse 
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image, ImageDraw
import os
import numpy as np
import io
import base64
import os
import shutil
import matplotlib.pyplot as plt
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


@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse(
        request=request, name="item.html", context={"id": id}
    )

UPLOAD_DIRECTORY = "uploads"
@app.get("/")
async def root():
    return FileResponse('static/index.html')
    # return RedirectResponse(url="/upload")


@app.get("/upload")
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/annotation")
async def show(request: Request):
    dataset_dir = os.path.join("static", "dataset")
    # datasetディレクトリ内の全JPEGファイルをリストアップ
    images = [
        f for f in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, f)) and f.lower().endswith((".jpg", ".jpeg"))
    ]
    # 画像のURLリストを作成
    image_urls = [f"/static/dataset/{image}" for image in images]
    return templates.TemplateResponse("annotation.html", {"request": request, "images": image_urls})
from pydantic import BaseModel

class Coordinate(BaseModel):
    image_path: str
    x: float
    y: float

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)
@app.post("/post_coordinates")
async def post_coordinates(coord: Coordinate):
    # 受け取ったデータを処理します
    image_path = coord.image_path
    x = coord.x
    y = coord.y
        # ここで必要な処理を実行（例：データベースに保存、ログ出力など）
    print(f"Received image: {image_path}, x: {x}, y: {y}")

    # 画像を開く
    with Image.open(image_path[1:]) as image:
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
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(masks[0], plt.gca())
        plt.show()
        draw = ImageDraw.Draw(image)
        
        
        # 座標にマーカーをプロット（例: 赤い円）
        radius = 20
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], fill="red", outline="red")
        
        # 画像をバイトデータに変換
        image_format = "PNG" if image.format != "JPEG" else "JPEG"
        fn = image_path.split("/")[-1]
        annotato_path = os.path.join("static","tmp_annotato",fn)
        image.save(annotato_path, format=image_format)


    # クライアントにレスポンスを返します
    return {"message": "Coordinates received successfully", "annotato_path":annotato_path}


@app.get("/test/{dataset}")
async def post_coordinates(request: Request,dataset:str):
    dataset_dir = os.path.join("static", "dataset", dataset)
    # datasetディレクトリ内の全JPEGファイルをリストアップ
    images = [
        f for f in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, f)) and f.lower().endswith((".jpg", ".jpeg"))
    ]
    return templates.TemplateResponse("test.html", {"request": request, "dataset":dataset, "images":images})


