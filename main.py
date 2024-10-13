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


from pydantic import BaseModel

def show_mask(mask, ax, random_color=False, borders = False):
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
import cv2
def get_mask_over(mask,img_path):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # Try to smooth contours

    img = cv2.imread(img_path)
    cv2.drawContours(img, contours, -1, (0,255,0), 3)

    # img[mask==1] = [128, 128, 128] 
    mask[mask==1] = 255

    return img

def save_mask(mask,fn):
    # static/annotaion_info/{dataset}/{image}-01.npy, 01... number of object in a image.

    np.save(fn,mask)



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
        img = get_mask_over(masks[0],image_path)
        annotato_path = os.path.join("static","tmp_annotato",coord.dataset,coord.fn)
        cv2.imwrite(annotato_path, img)


    # クライアントにレスポンスを返します
    return {"message": "Coordinates received successfully", "annotato_path":os.path.join(coord.dataset,coord.fn)}


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
    return templates.TemplateResponse("annotation.html", {"request": request, "dataset":dataset, "images":images})


class ImageInfo(BaseModel):
    dataset:str
    fn:str
    @property
    def original_img_path(self) -> str:
        return os.path.join("static", "dataset", self.dataset, self.fn)


app.get("annotation")
async def get_annotation(image_info: ImageInfo):

    return {"image_path":image_info}