from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from starlette.responses import FileResponse 

import os
import shutil

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
    image: str
    x: float
    y: float
@app.post("/post_coordinates")
async def post_coordinates(coord: Coordinate):
    # 受け取ったデータを処理します
    image = coord.image
    x = coord.x
    y = coord.y
        # ここで必要な処理を実行（例：データベースに保存、ログ出力など）
    print(f"Received image: {image}, x: {x}, y: {y}")
    
    # クライアントにレスポンスを返します
    return {"message": "Coordinates received successfully"}