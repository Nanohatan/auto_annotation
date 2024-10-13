from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
dataDir='.'
fn='output_coco2'
# fn = "frame_0002"
annFile='{}/annotations/{}.json'.format(dataDir,fn)
coco=COCO(annFile)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['shrimp'])
imgIds = coco.getImgIds(catIds=catIds )
imgIds = coco.getImgIds(imgIds = [1])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

fn = os.path.join("dataset","images",img['file_name'])
I = cv2.imread(fn)
plt.imshow(I); 
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)

anns = coco.loadAnns(annIds)
print(anns)
coco.showAnns(anns)

plt.show()