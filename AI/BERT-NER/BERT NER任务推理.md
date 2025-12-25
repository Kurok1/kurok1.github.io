这里提供在线推理服务脚本：
```python
import torch
import uuid
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForTokenClassification
# -----------------------------------------------------
# 模型注册表（配置你的多个NER模型）
# -----------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict] = {
    "ner-default": {
        "path": "/path/to/bert-ner",
        "device": "cuda:0"
    }
}
# -----------------------------------------------------
# 模型实例池
# -----------------------------------------------------
class NERModel:
    def __init__(self, path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(path)
        self.id2label = self.model.config.id2label
        self.model.to(self.device)
        self.model.eval()
    @torch.no_grad()
    def infer(self, text: str):
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            padding=True,
        )
        tokens = enc["input_ids"].shape[1]
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = enc.to(self.device)
        logits = self.model(**enc).logits
        preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
        entities = self.decode(preds, offsets, text)
        return entities, tokens
    def decode(self, preds, offsets, text):
        ents = []
        cur = None
        for p, (s, e) in zip(preds, offsets):
            if s == e:
                continue
            label = self.id2label[p]
            if label == "O":
                if cur:
                    ents.append(cur)
                    cur = None
                continue
            tag, ent = label.split("-", 1)
            if tag == "B":
                if cur:
                    ents.append(cur)
                cur = {
                    "type": ent,
                    "start": s,
                    "end": e - 1,
                    "text": text[s:e]
                }
            elif tag == "I" and cur:
                cur["end"] = e - 1
                cur["text"] = text[cur["start"]:e]
        if cur:
            ents.append(cur)
        return ents
# -----------------------------------------------------
# 初始化全部NER模型
# -----------------------------------------------------
MODEL_POOL: Dict[str, NERModel] = {}
print("Loading NER models...")
for name, cfg in MODEL_REGISTRY.items():
    MODEL_POOL[name] = NERModel(
        path=cfg["path"],
        device=cfg["device"]
    )
    print(f"✅ Loaded [{name}] → {cfg['path']} on {cfg['device']}")
# -----------------------------------------------------
# OpenAI 协议
# -----------------------------------------------------
class Message(BaseModel):
    role: str
    content: str
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
# -----------------------------------------------------
# API
# -----------------------------------------------------
app = FastAPI(title="Multi-NER OpenAI Server")
@app.get("/health")
def health():
    return {
        "models": list(MODEL_POOL.keys())
    }
@app.post("/v1/chat/completions")
async def chat(req: ChatCompletionRequest, response: Response):
    
    if req.model not in MODEL_POOL:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' not found"
        )
    # 拼接用户输入
    text = ""
    for msg in req.messages:
        if msg.role == "user":
            text += msg.content
    entities, tokens = MODEL_POOL[req.model].infer(text)
    # Header 写 token
    response.headers["X-Prompt-Tokens"] = str(tokens)
    return {
        "id": f"nercmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": str(entities)
                }
            }
        ],
    }
```

## vLLM部署
可以选择[vllm](https://github.com/vllm-project/vllm)进行部署
```
vllm serve /path/to/bert-ner --runner pooling --port 9002 --served-model-name bert --task classify
```

访问示例：
```sh
curl --location 'http://{host}:{port}/pooling' \
--header 'X-Model: bert' \
--header 'Content-Type: application/json' \
--data '{
   "encoding_format": "json",
   "task": "token_classify",
   "messages": [
    {
        "role": "user",
        "content": "据说，刘德华平时生活在香港，但是他喜欢去北京钓鱼"
    }
   ]
 }'
```