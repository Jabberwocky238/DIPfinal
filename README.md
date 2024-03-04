
# 环境需求
python >= 3.9
torch >= 2.0.1
如果你是cuda，一定要先装torchvision!!!timm会把cuda版torchvision顶掉
然后pip install -r requirements.txt

# 下载
https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_b.pth
放到项目根目录

# 课堂展示
运行test_onclass.ipynb

# 如果要使用数据集进行eval
下载https://huggingface.co/lkeab/hq-sam/blob/main/data/sam_vit_b_01ec64.pth
下载https://huggingface.co/lkeab/hq-sam/blob/main/data/sam_vit_b_maskdecoder.pth
放在train/pretrained_checkpoint下

下载https://huggingface.co/lkeab/hq-sam/blob/main/data/thin_object_detection.zip
解压放在train/data下

运行test_eval.ipynb

# 如果要使用yolo
pip install ultralytics
运行test_yolo.ipynb


