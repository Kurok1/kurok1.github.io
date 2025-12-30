
# 单个IoU


> 比较检测结果的面积和已经标注的面积交叉部分的占比

IoU定义：
 $$\text{IoU} = \frac{预测框 ∩ 标注框}{预测框 ∪ 标注框}  $$


---

推荐流程如下：

Step 1：一一匹配预测框和 GT 框

- 通常用 **最大 IoU 匹配**
    
- 一个 GT 只能匹配一个预测框
    

Step 2：计算 IoU

```python
iou = inter_area / union_area
```

Step 3：设定阈值

```text
IoU ≥ 0.5 → True Positive
IoU < 0.5 → False Positive
GT 没被匹配 → False Negative
```


---

## 验证流程

- 准备原始图片和coco格式的图片
coco格式参数如下：
```json
{
  "images": [{"id": 1, "file_name": "test.jpg"}],
  "annotations": [
    {
      "image_id": 1,
      "bbox": [x, y, w, h],
      "category_id": 3
    }
  ]
}

```

假设detr模型预测格式如下：
```json
predictions = [
    {
        "bbox": [x1, y1, x2, y2],  # 注意：xyxy
        "score": 0.92,
        "category_id": 3
    }
]

```

- 计算两个bbox的IoU
```python
def compute_iou(box1, box2):
    """
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

```
- 从单个coco格式文件中读取bbox和labels
```python
import json

def load_coco_gt(coco_json_path, image_id):
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    gt_boxes = []
    gt_labels = []

    for ann in coco["annotations"]:
        if ann["image_id"] == image_id:
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x + w, y + h])  # 转 xyxy
            gt_labels.append(ann["category_id"])

    return gt_boxes, gt_labels

```
- IoU 匹配 + TP/FP/FN 统计
```python
"""
TP：True Positive，分类器预测结果为正样本，实际也为正样本，即正样本被正确识别的数量。
FP：False Positive，分类器预测结果为正样本，实际为负样本，即误报的负样本数量。
FN：False Negative，分类器预测结果为负样本，实际为正样本，即漏报的正样本数量。
"""
def evaluate_single_image(gt_boxes, gt_labels, preds, iou_thresh=0.5):
    matched_gt = set()
    results = []

    for pred in preds:
        best_iou = 0
        best_gt_idx = -1

        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            if pred["category_id"] != gt_labels[i]:
                continue

            iou = compute_iou(pred["bbox"], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_thresh:
            matched_gt.add(best_gt_idx)
            results.append(("TP", best_iou))
        else:
            results.append(("FP", best_iou))

    fn = len(gt_boxes) - len(matched_gt)
    return results, fn

```
- 调用参考
```python
# COCO 标注
coco_json = "annotations.json"
image_id = 1

gt_boxes, gt_labels = load_coco_gt(coco_json, image_id)

# 模型预测（示例）
predictions = [
    {
        "bbox": [100, 120, 220, 260],  # xyxy
        "score": 0.9,
        "category_id": 3
    }
]

# 置信度过滤
predictions = [p for p in predictions if p["score"] > 0.5]

results, fn = evaluate_single_image(gt_boxes, gt_labels, predictions)

print("Results:")
for r in results:
    print(r)

print("False Negatives:", fn)

#结果可能如下
("TP", 0.73)   # 正确检测，IoU=0.73
("FP", 0.12)   # 预测错误
False Negatives: 1
```


# mAP评估

## 一、整体评估流程（先对齐概念）

你要做的 **整体 mAP 评估**，本质是：

> 用 **COCO 官方评估逻辑**：
> 在 **整个验证集** 上统计 **多 IoU 阈值 + 多类别** 的 Precision-Recall，得到 mAP

标准流程如下：

```
GT（COCO格式）
        ↓
模型预测所有图片
        ↓
整理成 COCO detection 格式（pred.json）
        ↓
COCOeval
        ↓
mAP / AP50 / AP75 / Recall
```

👉 **强烈建议**：不要自己手写 mAP，直接用 `pycocotools`

---

## 二、需要准备的 2 个文件

Ground Truth（你已经有）

COCO 原始标注文件，例如：

```text
instances_val.json
```

---

Detection Results（你需要生成）

这是一个 **list of dict**，格式**非常严格**：

```json
[
  {
    "image_id": 1,
    "category_id": 3,
    "bbox": [x, y, w, h],
    "score": 0.92
  }
]
```

⚠️ 注意：

- bbox 必须是 **xywh**
    
- image_id / category_id **必须和 GT 一致**
    
- score 是置信度
    

---

## 三、从 DETR 输出生成 COCO Detection 结果

假设 DETR 输出（单张图）：

```python
outputs = {
    "boxes": Tensor[N, 4],   # xyxy
    "scores": Tensor[N],
    "labels": Tensor[N]
}
```

---

 ✅ DETR → COCO bbox 转换（可选实现,本地离线推理使用）

```python
def detr_to_coco(preds, image_id, score_thresh=0.8):
	"""
	在线detr推理直接调用在线服务即可
	"""
    results = []

    for box, score, label in zip(
        preds["boxes"], preds["scores"], preds["labels"]
    ):
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = box.tolist()
        w = x2 - x1
        h = y2 - y1

        results.append({
            "image_id": image_id,
            "category_id": int(label),
            "bbox": [x1, y1, w, h],
            "score": float(score)
        })

    return results
```

---

✅ 遍历整个验证集

```python
all_results = []

for image_id, image in dataloader:
    preds = model(image)

    coco_preds = detr_to_coco(preds, image_id)
    all_results.extend(coco_preds)
```

---

✅ 保存为 json

```python
import json

with open("detr_results.json", "w") as f:
    json.dump(all_results, f)
```

---

## 四、COCO mAP 官方评估代码（核心）
使用uv安装
```sh
uv add pycocotools
```

使用pip安装

```bash
pip install pycocotools
```

---

### ✅ 完整评估代码

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 1. 加载 GT
coco_gt = COCO("instances_val.json")

# 2. 加载预测结果
coco_dt = coco_gt.loadRes("detr_results.json")

# 3. 创建 evaluator
coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

# 4. 运行评估
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
```

---

### 📊 输出解释（非常重要）

```text
Average Precision  (AP) @[ IoU=0.50:0.95 | area=all ] = 0.421
Average Precision  (AP) @[ IoU=0.50      | area=all ] = 0.623
Average Precision  (AP) @[ IoU=0.75      | area=all ] = 0.447
Average Recall     (AR) @[ IoU=0.50:0.95 | area=all ] = 0.511
```

|指标|含义|
|---|---|
|AP@[0.5:0.95]|**主 mAP（论文对比用）**|
|AP@0.5|宽松准确率|
|AP@0.75|严格准确率|
|AR|召回率|

👉 **DETR 论文就是用这个指标**

---

## 五、DETR 评估中的 5 个常见坑（一定要检查）

❌ 1. bbox 坐标格式错

- GT：`xywh`
    
- DETR 输出：`xyxy`
    
- **一定要转**
    

---

❌ 2. category_id 对不上

- COCO category_id **不是连续的**
    
- 如果你是自定义数据集：
    

```python
label → coco_category_id
```

要显式映射

---

❌ 3. image_id 不一致

- image_id **必须是 COCO 标注里的 id**
    
- 不是文件名，不是 index
    

---

❌ 4. score 过滤过高

- DETR 推荐：
    

```python
score_thresh = 0.05
```

mAP 会自动考虑排序

---

❌ 5. 用单张图判断模型好坏

- mAP **必须是数据集级别**
    
- 单张图只能 debug
