
## 原始数据json格式要求
```json
{
    "text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，", 
    "label": {
        "name": {
            "叶老桂": [
                [9, 11],
                [32, 34]
            ]
        }, 
        "company": {
            "浙商银行": [
                [0, 3]
            ]
        }
    }
}
```
要求⚠️：
1. text存放原始文本
2. label记录相关标签
3. 每个标签记录对应的原始文本(subtext)和索引下标起始位置(start,end)，其中满足`text[start:end] == subtext`

## BIO格式转换
> [BIO格式介绍](`https://blog.csdn.net/HappyRocking/article/details/79716212`)

转换脚本，从json到bio格式：
```python
# convert_json_to_conll.py
import json
import argparse
from collections import defaultdict

def detect_span_mode(text, spans, ent_text):
    # 检测 [start,end] 是含右边还是不含右边
    return "inclusive"
    # s,e = spans
    # if 0 <= s <= len(text)-1 and 0 <= e <= len(text)-1:
    #     if text[s:e+1] == ent_text:
    #            # [s,e] inclusive
    # # check exclusive
    # if 0 <= s <= len(text) and 0 <= e <= len(text):
    #     if text[s:e] == ent_text:
    #         return "exclusive"   # [s,e) exclusive
    # return None

def apply_spans(text, span_list, label_type):
    # span_list: list of [s,e] as in input; label_type: e.g. "name"
    applied = []
    mode = None
    for s,e in span_list:
        if mode is None:
            mode = detect_span_mode(text, (s,e), text[s:e+1])
            if mode is None:
                # try exclusive
                mode = detect_span_mode(text, (s,e), text[s:e])
        if mode == "inclusive":
            applied.append((s, e))
        elif mode == "exclusive":
            applied.append((s, e-1))
        else:
            # as fallback, assume inclusive
            applied.append((s, e))
    return applied

def convert_one(item):
    text = item['text']
    chars = list(text)
    labels = ['O'] * len(chars)
    for ent_type, mapping in item.get('label', {}).items():
        # mapping: { "实体文本": [[s,e], ...], ... }
        for ent_text, spans in mapping.items():
            # try detect whether spans are inclusive/exclusive per span
            applied_spans = []
            for span in spans:
                s,e = span
                # detection
                mode = detect_span_mode(text, (s,e), ent_text)
                if mode == "inclusive":
                    applied_spans.append((s, e))
                elif mode == "exclusive":
                    applied_spans.append((s, e-1))
                else:
                    # fallback: try to find occurrences of ent_text in text and match by order
                    # find all indices of ent_text
                    starts = []
                    st = 0
                    while True:
                        idx = text.find(ent_text, st)
                        if idx == -1:
                            break
                        starts.append(idx)
                        st = idx + 1
                    if starts:
                        # pick next unused start
                        # naive: pick first occurrence whose span length matches
                        chosen = None
                        for idx in starts:
                            if idx <= s <= idx+len(ent_text)-1:
                                chosen = (idx, idx+len(ent_text)-1)
                                break
                        if chosen is None:
                            chosen = (starts[0], starts[0]+len(ent_text)-1)
                        applied_spans.append(chosen)
                    else:
                        # last resort: use provided s,e as inclusive
                        applied_spans.append((s, e))
            # apply labels
            for (s2,e2) in applied_spans:
                if s2 < 0 or e2 >= len(chars) or s2 > e2:
                    # skip invalid
                    continue
                labels[s2] = 'B-' + ent_type
                for i in range(s2+1, e2+1):
                    labels[i] = 'I-' + ent_type
    return chars, labels

def main(infile, outfile, label_out):
    raw = None
    with open(infile, 'r', encoding='utf-8') as f:
        txt = f.read().strip()
        if txt.startswith('['):
            raw = json.loads(txt)
        else:
            # assume jsonl
            raw = []
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                raw.append(json.loads(line))
    label_set = set()
    converted = []
    for item in raw:
        tokens, labels = convert_one(item)
        converted.append({'tokens': tokens, 'labels': labels})
        label_set.update(labels)
    # ensure O exists and order labels: O, then B-/I- group by ent and B before I
    label_set.discard('O')
    entity_types = sorted({lab.split('-',1)[1] for lab in label_set})
    label_list = ['O']
    for ent in entity_types:
        label_list.append('B-' + ent)
        label_list.append('I-' + ent)
    with open(outfile, 'w', encoding='utf-8') as f:
        for obj in converted:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    with open(label_out, 'w', encoding='utf-8') as f:
        json.dump(label_list, f, ensure_ascii=False, indent=2)
    print(f'Wrote {len(converted)} samples to {outfile}')
    print(f'Label list saved to {label_out}: {label_list}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='原始 JSON/JSONL 文件')
    parser.add_argument('--out', '-o', default='data_converted.jsonl', help='输出 jsonl，每行{{tokens, labels}}')
    parser.add_argument('--label_out', default='label_list.json', help='输出标签列表')
    args = parser.parse_args()
    main(args.input, args.out, args.label_out)
```

使用这个转换脚本
```sh
python bert-convert-json.py -i /path/to/input.json -o /path/to/output.json --label_out /path/to/labels.json
```

## 模型训练
训练脚本：
```python
# train_token_classify.py
import json
import argparse
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import precision_score, recall_score, f1_score
# --------------------------
# 工具函数
# --------------------------
def load_jsonl_dataset(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))
    return Dataset.from_list(items)
def read_label_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        max_length=512,
        truncation=True
    )
    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # CLS / SEP / padding
                label_ids.append(-100)
            else:
                label = labels[word_idx]
                if word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label])
                else:
                    if label == "O":
                        label_ids.append(label_to_id["O"])
                    else:
                        ent = label.split("-", 1)[1]
                        label_ids.append(label_to_id.get("I-" + ent, label_to_id["O"]))
                previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs
# --------------------------
# metric
# --------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_labels = []
    true_preds = []
    for i in range(labels.shape[0]):
        lab = []
        pred = []
        for j in range(labels.shape[1]):
            if labels[i, j] != -100:
                lab.append(label_list[labels[i, j]])
                pred.append(label_list[preds[i, j]])
        true_labels.append(lab)
        true_preds.append(pred)
    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds)
    }
# --------------------------
# main
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--label_list", required=True)
    parser.add_argument("--model_name", default="./chinese-roberta-wwm-ext-large")
    parser.add_argument("--output_dir", default="./out")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()
    # --------------------------
    # 加载数据
    # --------------------------
    dataset = load_jsonl_dataset(args.data)
    # ✅ 真实数据集 95% / 5% 切分
    dataset = dataset.train_test_split(
        test_size=0.05,
        shuffle=True,
        seed=42
    )
    dataset = DatasetDict({
        "train": dataset["train"],
        "validation": dataset["test"]
    })
    # --------------------------
    # 标签
    # --------------------------
    global label_list
    label_list = read_label_list(args.label_list)
    label_to_id = {x: i for i, x in enumerate(label_list)}
    id_to_label = {i: x for i, x in enumerate(label_list)}
    # --------------------------
    # 模型和 tokenizer
    # --------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        local_files_only=True,
        num_labels=len(label_list),
        label2id=label_to_id,
        id2label=id_to_label
    )
    # --------------------------
    # Tokenize + 对齐标签
    # --------------------------
    def preprocess(examples):
        return tokenize_and_align_labels(examples, tokenizer, label_to_id)
    dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=["tokens", "labels"]
    )
    collator = DataCollatorForTokenClassification(tokenizer)
    # --------------------------
    # arguments
    # --------------------------
    args_train = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        metric_for_best_model="f1",
        greater_is_better=True
    )
    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    # --------------------------
    # 开始训练
    # --------------------------
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ 训练完成:", args.output_dir)
```

使用脚本:
```sh
acclerate config ##如果已经配置过了，可以跳过该步骤
nohup accelerate launch training.py \
--data /path/to/data.jsonl \
--label_list /path/to/labels.json \
--output_dir /path/to/output \
--epochs 10 \ #训练周期
--batch_size 16 \ #训练批处理大小
--learning_rate 3e-5 > output.log 2>&1 &
```

