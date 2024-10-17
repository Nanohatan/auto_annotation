# routers/images.py

from fastapi import APIRouter, HTTPException
import glob
import os
router = APIRouter()



@router.get("/get/{dataset}/{id}")
async def get_image_url(dataset:str ,id: int):    
    files = glob.glob(os.path.join("static", "dataset", dataset, "**"))
    if id >= len(files):
        raise HTTPException(status_code=404, detail="image not found")
    files.sort()
    return {"id": id, "url": files[id]}

@router.get("/get/annotated-image/{dataset}/{id}")
async def get_annotated_image_url(dataset:str ,id: int):    
    files = glob.glob(os.path.join("static", "dataset", dataset, "**"))
    if id >= len(files):
        raise HTTPException(status_code=404, detail="image not found")
    files.sort()
    return {"dataset": dataset, "id": id, "url": files[id]}

@router.get("/get/json/{dataset}")
async def get_json(dataset:str):
    file = os.path.join("static", "dataset", dataset, "out.json")
    return {"dataset": dataset, "url": file}

