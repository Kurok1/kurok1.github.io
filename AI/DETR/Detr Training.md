
# 数据标注
标注可以使用[labelme](https://labelme.io/docs/install-labelme-app)进行标注,标注的格式也是labelme格式

# 数据训练转换
由于训练DETR模型所需的格式要求coco格式，所以我们需要将labelme格式的数据转换成coco格式

coco格式输出文件夹如下：
```
coco_output/
├── JPEGImages/
│   └── *.jpg
└── Visualization/
│   └── *.jpg
├── annotations_train.json
```

`annotations_train.json`即为所需要的训练数据

下面给出python格式的转换脚本
```python
#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
import random

import imgviz
import numpy as np
import labelme

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()

def build_coco_structure(now, class_name_to_id):
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[
            dict(
                url=None,
                id=0,
                name=None,
            )
        ],
        images=[],
        type="instances",
        annotations=[],
        categories=[
            dict(
                supercategory=None,
                id=class_id,
                name=class_name,
            )
            for class_name, class_id in class_name_to_id.items()
        ],
    )
    return data

def main():
    args = parse_args()
    random.seed(args.seed)

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "Visualization"))
    print("Creating dataset:", args.output_dir)

    now = datetime.datetime.now()

    # 读取类别
    class_name_to_id = {}
    class_name_count = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        class_name_count[class_name] = 0

    # 获取所有json文件并划分
    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    random.shuffle(label_files)
    n_total = len(label_files)
    n_train = int(n_total)
    train_files = label_files[:n_train]

    for split_name, split_files in [("train", train_files)]:
        data = build_coco_structure(now, class_name_to_id)
        for image_id, filename in enumerate(split_files):
            print(f"Generating {split_name} dataset from:", filename)
            label_file = labelme.LabelFile(filename=filename)
            base = osp.splitext(osp.basename(filename))[0]
            out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
            img = labelme.utils.img_data_to_arr(label_file.imageData)
            imgviz.io.imsave(out_img_file, img)
            data["images"].append(
                dict(
                    license=0,
                    url=None,
                    file_name=osp.relpath(out_img_file, osp.dirname(args.output_dir)),
                    height=img.shape[0],
                    width=img.shape[1],
                    date_captured=None,
                    id=image_id,
                )
            )
            masks = {}
            segmentations = collections.defaultdict(list)
            for shape in label_file.shapes:
                points = shape["points"]
                label = shape["label"]

                # 如果 label 不在 labels.txt 中，则跳过该 shape（并给出警告）
                if label not in class_name_to_id:
                    print(f"Warning: label '{label}' in file '{filename}' not found in {args.labels}. Skipping this shape.")
                    continue

                # 仅对已知类别进行计数
                class_name_count[label] = class_name_count[label] + 1

                group_id = shape.get("group_id")
                shape_type = shape.get("shape_type", "polygon")
                mask = labelme.utils.shape_to_mask(img.shape[:2], points, shape_type)
                if group_id is None:
                    group_id = uuid.uuid1()
                instance = (label, group_id)
                if instance in masks:
                    masks[instance] = masks[instance] | mask
                else:
                    masks[instance] = mask
                if shape_type == "rectangle":
                    (x1, y1), (x2, y2) = points
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    points = [x1, y1, x2, y1, x2, y2, x1, y2]
                if shape_type == "circle":
                    (x1, y1), (x2, y2) = points
                    r = np.linalg.norm([x2 - x1, y2 - y1])
                    n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                    i = np.arange(n_points_circle)
                    x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                    y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                    points = np.stack((x, y), axis=1).flatten().tolist()
                else:
                    points = np.asarray(points).flatten().tolist()
                segmentations[instance].append(points)
            segmentations = dict(segmentations)
            for instance, mask in masks.items():
                cls_name, group_id = instance
                if cls_name not in class_name_to_id:
                    continue
                cls_id = class_name_to_id[cls_name]
                mask = np.asfortranarray(mask.astype(np.uint8))
                mask = pycocotools.mask.encode(mask)
                area = float(pycocotools.mask.area(mask))
                bbox = pycocotools.mask.toBbox(mask).flatten().tolist()
                data["annotations"].append(
                    dict(
                        id=len(data["annotations"]),
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=segmentations[instance],
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    )
                )
            if not args.noviz:
                viz = img
                if masks:
                    labels, captions, masks_ = zip(
                        *[
                            (class_name_to_id[cnm], cnm, msk)
                            for (cnm, gid), msk in masks.items()
                            if cnm in class_name_to_id
                        ]
                    )
                    viz = imgviz.instances2rgb(
                        image=img,
                        labels=labels,
                        masks=masks_,
                        captions=captions,
                        font_size=15,
                        line_width=2,
                    )
                out_viz_file = osp.join(args.output_dir, "Visualization", base + ".jpg")
                imgviz.io.imsave(out_viz_file, viz)
        out_ann_file = osp.join(args.output_dir, f"annotations_{split_name}.json")
        with open(out_ann_file, "w") as f:
            json.dump(data, f)
        print(class_name_count)

if __name__ == "__main__":
    main()
```

使用参考：
1. 需要提前准备labels.txt文件
```plain
__ignore__
transport_paper
transport_ic
transport_a4
proxy_front
proxy_back
license_front
license_vehicle
license_back
license_check
```
注意首行必须要用`__ignore`开头
2. 执行转换脚本
```sh
python labelme2coco.py input_dir /path/to/labelme output_dir /path/to/labelme --labels /path/to/labels.txt
```

# DETR模型训练
准备好COCO格式的数据后，即可开始进行在线训练。
## 下载基座模型
```sh
export HF_ENDPOINT="https://hf-mirror.com" ##国内加速
hf download facebook/detr-resnet-50
```

## 训练脚本
```python
import json
import os
from datasets import Dataset, DatasetDict
from transformers import DetrForObjectDetection, DetrImageProcessor, TrainingArguments, Trainer
from torchvision import transforms
from PIL import Image
import torch

# 加载模型和处理器
processor = DetrImageProcessor.from_pretrained(local_files_only=True, pretrained_model_name_or_path="./detr-resnet-50")


def convert_coco_to_dataset(coco_json_path, image_dir):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id2filename = {img["id"]: img["file_name"] for img in coco["images"]}
    image_id_to_annotations = {}
    for ann in coco["annotations"]:
        image_id_to_annotations.setdefault(ann["image_id"], []).append({
            "bbox": ann["bbox"],
            "category_id": ann["category_id"],
            "area": ann["area"],
            "segmentation": ann.get("segmentation", []),
            "iscrowd": ann.get("iscrowd", 0)
        })

    data = []
    for img in coco["images"]:
        image_path = os.path.join(image_dir, img["file_name"])
        anns = image_id_to_annotations.get(img["id"], [])
        image = Image.open(image_path).convert("RGB")
        annotations = {
            "image_id": img["id"],
            "file_name": img["file_name"],
            "annotations": anns
        }
        encoding = processor(images=image, annotations=annotations, return_tensors="pt")
        data.append({
            "pixel_values": encoding["pixel_values"].squeeze(),
            "labels": encoding["labels"]
        })

    return Dataset.from_list(data)

def load_coco_datasets(base_path):
    train_dataset = convert_coco_to_dataset(
        os.path.join(base_path, "annotations_train.json"),
        os.path.join("./")
    )
    val_dataset = convert_coco_to_dataset(
        os.path.join(base_path, "annotations_val.json"),
        os.path.join("./")
    )
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

class CocoDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, coco_json_path, image_dir, processor):
        with open(coco_json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.image_dir = image_dir
        self.processor = processor

        # 构建映射关系
        id2filename = {img["id"]: img["file_name"] for img in coco["images"]}
        image_id_to_annotations = {}
        for ann in coco["annotations"]:
            image_id_to_annotations.setdefault(ann["image_id"], []).append({
                "bbox": ann["bbox"],
                "category_id": ann["category_id"],
                "area": ann["area"],
                "segmentation": ann.get("segmentation", []),
                "iscrowd": ann.get("iscrowd", 0)
            })

        # 保存元信息，不加载图像
        self.items = []
        for img in coco["images"]:
            anns = image_id_to_annotations.get(img["id"], [])
            self.items.append({
                "image_id": img["id"],
                "file_name": img["file_name"],
                "annotations": anns
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if isinstance(idx, (list, torch.Tensor)):
            return [self.__getitem__(i) for i in idx]
        item = self.items[idx]
        image_path = os.path.join(self.image_dir, item["file_name"].replace("\\", "/"))
        image = Image.open(image_path).convert("RGB")

        annotations = []
        for ann in item["annotations"]:
            annotations.append({
                "bbox": ann["bbox"],
                "category_id": ann["category_id"],
                "area": ann["area"],
                "segmentation": ann.get("segmentation", []),
                "iscrowd": ann.get("iscrowd", 0)
            })

        encoding = self.processor(
            images=image,
            annotations={
                "image_id": item["image_id"],
                "annotations": annotations
            },
            return_tensors="pt"
        )

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),  # 去掉 batch 维度
            "pixel_mask": encoding["pixel_mask"],
            "labels": encoding["labels"][0]
        }

train_ds = CocoDetectionDataset(
    coco_json_path="coco_vehicle/annotations_train.json",#指向coco数据路径
    image_dir="./",
    processor=processor
)


model = DetrForObjectDetection.from_pretrained(
    local_files_only=True, 
    pretrained_model_name_or_path="./detr-resnet-50", 
    num_labels=9,  #labels数保持一致
    ignore_mismatched_sizes=True  # 若分类头 shape 不一致
)
# 类别映射（你需要与COCO JSON中的 category_id 对应）
model.config.id2label = {
    0: "transport_paper",
    1: "transport_ic",
    2: "transport_a4",
    3: "proxy_front",
    4: "proxy_back",
    5: "license_front",
    6: "license_vehicle",
    7: "license_back",
    8: "license_check"
}
model.config.label2id = {v: k for k, v in model.config.id2label.items()}

# 训练参数
args = TrainingArguments(
        output_dir="./detr-finetune-vehicle-20251125",
        per_device_train_batch_size=2,
        num_train_epochs=100, #训练epochs数，推荐100-300
        learning_rate=1e-5,
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        lr_scheduler_type="cosine",
        warmup_steps=200,
    )
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    pixel_mask = [item["pixel_mask"] for item in batch]  # 如果你启用了 pixel_mask
    labels = [item["labels"] for item in batch]  # 保持原始 List[Dict]

    encoding = processor.pad(
        # {"pixel_values": pixel_values, "pixel_mask": pixel_mask},  # 如果有 pixel_mask，一起 pad
        pixel_values,
        return_tensors="pt"
    )

    encoding["labels"] = labels  # 不做任何变换，保留 List[Dict]

    return encoding

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    tokenizer=processor,
    data_collator=collate_fn
)

# 开始训练
trainer.train()
model.save_pretrained("/path/to/output") #指向你的目标输出路径
processor.save_pretrained("/path/to/output")
```


## 使用Accelerate进行训练
`accelerate`是transformers官方提供的训练执行命令，方便在多gpu环境下进行模型训练工作

### 安装accelerate
```sh
pip install accelerate
```

### 配置accelerate
```sh
accelerate config
```
执行后根据当前环境进行配置，比如GPU数量，要求训练精度等

### 训练
```sh
accelerate launch training.py
##如果希望后台挂载
nohup accelerate launch training.py > training.log 2>&1 &
```
