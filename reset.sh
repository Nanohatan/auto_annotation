rm -rf static/annotation_info/
rm -rf static/json/
rm -rf static/tmp_annotato/
git clone https://github.com/facebookresearch/sam2.git sam_clone
cp sam_clone/sam2 sam2
# fastapi dev main.py     
