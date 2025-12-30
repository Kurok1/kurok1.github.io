
## 一、pycocotools / COCO 的 mAP 输出一般长这样

典型输出（示例）👇：

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.652
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.458
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.215
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium| maxDets=100 ] = 0.447
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.612
```

---

## 二、每个指标代表什么（这是重点）

### 1️⃣ **最重要的指标：**

```
AP@[IoU=0.50:0.95]
```

👉 **这就是大家口中的“COCO mAP”**

- IoU 从 0.50 到 0.95，每隔 0.05 算一次
    
- 对模型**综合能力要求最高**
    
- 论文、模型对比**默认看这个**
    

📌 **判断好坏主要看它**

---

### 2️⃣ AP@0.50（PASCAL mAP）

```
AP@[IoU=0.50]
```

- 只要求 IoU ≥ 0.5
    
- **对框不准的模型也很友好**
    
- 数值通常 **比 mAP 高很多**
    

👉 如果：

- AP@0.50 很高
    
- AP@0.75 / AP@0.50:0.95 很低
    

说明：**能找到目标，但框不准（定位能力差）**

---

### 3️⃣ AP@0.75（定位能力）

```
AP@[IoU=0.75]
```

- 对框的位置要求严格
    
- 能反映 **bounding box 精度**
    

👉 DETR 类模型一般：

- AP@0.75 **不会太低**
    
- 否则说明收敛或训练有问题
    

---

### 4️⃣ small / medium / large

```
AP small / medium / large
```

- small：小目标（< 32²）
    
- medium：中目标
    
- large：大目标
    

👉 常见现象：

- **small 很低是正常的**
    
- large 明显高于 small
    

---

## 三、DETR 的 mAP 大概什么水平算“正常 / 好”？

下面是**经验参考（COCO 类数据集）**👇

|AP@[0.50:0.95]|评价|
|---|---|
|< 0.25|❌ 很差（可能没收敛 / 数据有问题）|
|0.30 ~ 0.35|⚠️ 勉强可用|
|0.38 ~ 0.42|✅ 正常 DETR|
|0.45 ~ 0.48|👍 训练得很好|
|> 0.50|🚀 很强（改进版 DETR）|

> 原始 **DETR-R50** 在 COCO val 上 ≈ **0.42**

---

## 四、如果你的是自定义数据集，还要注意这几点

1. **类别数多少？**
    
    - 类别少 → mAP 通常更高
        
2. **数据量是否足够？**
    
    - DETR 对数据量敏感
        
3. **训练轮数**
    
    - 原始 DETR 通常需要 **300 epochs**
        
4. **AP@0.50 很高但 mAP 很低**
    
    - 99% 是 **box 回归没学好**
        
