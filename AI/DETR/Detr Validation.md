
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


## 四、DETR 评估中的 4 个常见坑（一定要检查）

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

❌ 5. 用单张图判断模型好坏

- mAP **必须是数据集级别**
    
- 单张图只能 debug

# mAP推导过程
DETR 在论文/代码里报告的 **mAP**，本质上就是用 **COCO 官方评测（pycocotools/COCOeval）**算出来的 AP 指标（常见写法 **AP@[0.50:0.95]**）。DETR 只是“怎么产生预测框”的方式不同（set prediction、通常不做 NMS），**评测时从 IoU 到 mAP 的计算流程和其他检测器是一样的**。([GitHub](https://raw.githubusercontent.com/facebookresearch/detr/main/README.md "raw.githubusercontent.com"))

下面按“从 IoU 一步步到 mAP”讲清楚，并配上公式。

---

## 1) IoU：从框的重叠开始

对任意一个预测框 (b) 和一个 GT 框 (g)，定义

$$
\mathrm{IoU}(b,g)=\frac{|b\cap g|}{|b\cup g|}.  
$$

IoU 是后面判定“这个预测算不算命中（TP）”的核心依据。

---

## 2) 选定 IoU 阈值：COCO 不是只用 0.5

COCO 的主指标会在 **10 个 IoU 阈值**上都算一遍 AP，然后再平均：

$$  
T={0.50,0.55,\dots,0.95}\quad(\Delta=0.05).  
$$

COCOeval 里就写着默认 **iouThrs = [.5:.05:.95]，T=10**。([Hugging Face](https://huggingface.co/spaces/sklum/detection_metrics/blob/3e2a0ca16993a7736a7b61f4281c2151a2eb406a/cocoeval.py "cocoeval.py · sklum/detection_metrics at 3e2a0ca16993a7736a7b61f4281c2151a2eb406a"))

---

## 3) 在每个类别、每个阈值下：用 IoU 做“匹配” → 得到 TP/FP

对每个类别 $(c)$（COCO 是按类算 AP，再平均），在某个 IoU 阈值 $(t\in T)$ 下：

1. 收集该类别的所有预测框 $({(b_i, s_i)})$，其中 $(s_i)$ 是置信度（DETR 的分类概率/score）。
    
2. **按 score 从高到低排序**。
    
3. 逐个预测框 (b_i) 去匹配同一张图、同一类别里“尚未被匹配过”的 GT 框，找 IoU 最大的那个：  
    $$  
    g^*(i)=\arg\max_{g\in \mathcal{G}_{\text{unmatched}}}\mathrm{IoU}(b_i,g)  
    $$
    
    - 若 $(\max \mathrm{IoU}(b_i,g)\ge t)$，则 $(b_i)$ 是 **TP**，并把对应 GT 标记为已匹配（一个 GT 只能匹配一次）。
        
    - 否则 $(b_i)$ 是 **FP**。
        

> COCO 评测还会对每张图最多取前 (100) 个检测（maxDets 默认是 ([1,10,100])），通常大家报告的框 AP 用的是 maxDet=100 那档。([Hugging Face](https://huggingface.co/spaces/sklum/detection_metrics/blob/3e2a0ca16993a7736a7b61f4281c2151a2eb406a/cocoeval.py "cocoeval.py · sklum/detection_metrics at 3e2a0ca16993a7736a7b61f4281c2151a2eb406a"))

---

## 4) 用 TP/FP 序列构造 Precision–Recall 曲线

设该类别在全数据集里 GT 总数为 $(N_{\text{gt}})$ 。对排序后的预测从 1 到 (k) 做累积：

$$  
\mathrm{TP}(k)=\sum_{i=1}^{k}\mathbf{1}[i\text{ is TP}],\qquad  
\mathrm{FP}(k)=\sum_{i=1}^{k}\mathbf{1}[i\text{ is FP}]  
$$

则

$$  
\mathrm{Precision}(k)=\frac{\mathrm{TP}(k)}{\mathrm{TP}(k)+\mathrm{FP}(k)},\qquad  
\mathrm{Recall}(k)=\frac{\mathrm{TP}(k)}{N_{\text{gt}}}.  
$$

当你从高分到低分不断“放宽阈值”（等价于取更长的前缀 (k)），就得到一条 PR 曲线。

---

## 5) 插值（interpolated precision）：把 PR 曲线“抹平”

COCO 使用 **101 个 recall 采样点**：

$$  
R={0,0.01,0.02,\dots,1.00}\quad(|R|=101),  
$$

COCOeval 里写的默认就是 **recThrs = [0:.01:1]，R=101**。([Hugging Face](https://huggingface.co/spaces/sklum/detection_metrics/blob/3e2a0ca16993a7736a7b61f4281c2151a2eb406a/cocoeval.py "cocoeval.py · sklum/detection_metrics at 3e2a0ca16993a7736a7b61f4281c2151a2eb406a"))

对每个采样 recall 值 $(r\in R)$，COCO 用“向右取最大”的插值精度（保证精度随 recall 单调不增）：

$$  
\hat p(r)=\max_{\tilde r\ge r} p(\tilde r),  
$$

其中 $(p(\tilde r))$ 是原始 PR 曲线上对应 recall 处的 precision（实现上是用离散点近似）。

---

## 6) 得到 AP：对 101 个 recall 点的插值 precision 求平均（近似面积）

在类别 (c)、IoU 阈值 (t) 下：

$$  
\mathrm{AP}_{c,t}=\frac{1}{101}\sum_{r\in R}\hat p(r)  
\approx \int_{0}^{1}\hat p(r),dr.  
$$

很多库/说明都会概括为：COCO 的 AP 是把 precision 在 **101 个 recall 点**上取值后平均，并且还会在多个 IoU 阈值上再平均。([Medium](https://medium.com/data-science-at-microsoft/how-to-smoothly-integrate-meanaverageprecision-into-your-training-loop-using-torchmetrics-7d6f2ce0a2b3?utm_source=chatgpt.com "How to smoothly integrate MeanAveragePrecision into ..."))

---

## 7) 得到 mAP（COCO 的主指标 AP@[.50:.95]）

COCO 报告的“AP”（很多论文也叫 mAP）是 **先对 IoU 阈值平均，再对类别平均**：

$$  
\mathrm{mAP}=\frac{1}{|C|}\sum_{c\in C}\Big(\frac{1}{|T|}\sum_{t\in T}\mathrm{AP}_{c,t}\Big),  
\quad T={0.50,\dots,0.95}.  
$$

所以你常见的：

- **AP50**：只取 (t=0.50) 的 AP
    
- **AP75**：只取 (t=0.75) 的 AP
    
- **AP@[.50:.95]**：10 个阈值都算，再平均（主指标）
    

而 DETR README 里写的 “**42 AP on COCO** / AP is computed on COCO 2017 val5k，并使用 pycocotools 评测”就是这个指标体系。([GitHub](https://raw.githubusercontent.com/facebookresearch/detr/main/README.md "raw.githubusercontent.com"))

---

### 一句话把链路串起来

$$  
\boxed{\text{IoU} \xrightarrow[\text{每类、每阈值}]{\text{匹配}} \text{TP/FP序列}  
\xrightarrow{\text{累积}} (P(k),R(k))  
\xrightarrow{\text{101点插值}} \mathrm{AP}_{c,t}  
\xrightarrow{\text{对 }t\text{ 与 }c\text{ 平均}} \mathrm{mAP}}  
$$

# 推理代码改进（不要使用高阈值过滤）
在标准 COCOeval 评测里，一般不会随意调一个“高阈值”去剪掉大量框来作弊；它会基于预测的 score 排序去画 PR 曲线，并且还有每图最多取前 100 个检测（maxDets=100）这类规则。  
但如果你在自己的代码里先用很高的 score 阈值把预测砍掉，再送去评测，确实会改变 PR 曲线，从而改变 AP（通常召回会下降）。

---

因此需要改进推理代码，由原先的**高score阈值**改进为**低score阈值+top-K过滤**
## score0阈值(τ)会不会影响 mAP？

**标准 COCO mAP（AP@[.50:.95]）理论上不需要你手动设 score 阈值。**

原因：COCOeval 会对你提交的所有检测按 score 排序，等价于“从 $τ=+∞$ 慢慢降低到 $−∞$”扫描整个 PR 曲线，最后对 PR 曲线积分得到 AP。

所以：

- 如果你**不人为删预测**（或只做很宽松的过滤，比如保留大量预测），评测会自动利用 score 排序生成整条 PR 曲线。
    
- 如果你在送评测前用了一个较高的 $τ$ 把低分框删掉了，相当于把 PR 曲线的“低阈值部分”截掉了：
    
    - 召回上不去 → AP 往往会 **下降**（尤其是对难例/小目标）。
        
- 但有时如果你的输出里充满极低分的垃圾框，删掉它们对 maxDets=100 的截断也许会有轻微影响；不过**正确做法**通常是让评测工具处理，而不是手工卡死一个固定 $τ$。


### 常用改进策略：

#### A. 直接降低阈值（最简单）

把 τ\tauτ 从 0.9 先降到：

- **0.5**（通常更平衡）
    
- 或 **0.3**（更偏召回，框会多一点）
    

经验上 DETR 的很多正确框分数不一定到 0.9，尤其是小目标/遮挡/远处目标。

#### B. 用 top-k（更稳定）

因为 DETR 固定输出 NNN 个 queries（例如 100 个），你可以：

1. 对每个 query 取“非 no-object 的最大类别概率”当 score
    
2. **按 score 排序取前 kkk**（比如 k=100k=100k=100、505050、202020），再做可视化/下游任务
    

这样不会因为阈值过高而“全无”，也不会因为阈值过低而爆炸式增框。

## 改进后的推理代码
```python
model.eval()

inputs = processor(images=img, return_tensors="pt").to(DEVICE)
with torch.inference_mode():  # 比 no_grad 更适合纯推理
    outputs = model(**inputs)

target_sizes = torch.tensor([img.size[::-1]], device=DEVICE)

# ① 评测/想要高召回：阈值设很低（甚至 0.0），不要用 0.9
results = processor.post_process_object_detection(
    outputs,
    threshold=0.05,          # 建议：评测用 0.0~0.05
    target_sizes=target_sizes
)[0]

# ② 用 top-k 控制输出数量（COCO 常用每图最多 100）
top_k = 100
if results["scores"].numel() > top_k:
    idx = results["scores"].topk(top_k).indices
    results = {k: v[idx] for k, v in results.items()}

```
核心就是：**别再用 0.9**；用 **较低 threshold 保召回**，再用 **top-k 控制数量**。

# 使用Pycocotool进行评价

## 1) COCO mAP 官方评估代码（核心）
使用uv安装
```sh
uv add pycocotools
```

使用pip安装

```bash
pip install pycocotools
```

## 2) 用COCOeval跑评测（bbox mAP）
```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ann_file = "instances_val2017.json"      # GT
pred_file = "detr_results.json"      # 你的预测

cocoGt = COCO(ann_file)
cocoDt = cocoGt.loadRes(pred_file)

cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
# 可选：只评某些图片
# cocoEval.params.imgIds = [397133, 12345, ...]

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

```
这会输出 COCO 标准的：

- AP @[IoU=0.50:0.95 | area=all | maxDets=100]
    
- AP50、AP75、以及 small/medium/large 等。[GitHub](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb?utm_source=chatgpt.com)

如何分析mAP结果，参考[此文章](./mAP结果分析.md)
