
NL2SQL很重要的一环就是Schema信息的维护和检索。Schema信息应该包含以下信息：
1. 表信息，所有使用的表名称，表的描述信息
2. 字段信息，所有表中的字段信息，包含字段名称，类型，所属表，描述信息，规则限制等（譬如数量必须为正数）
3. 关联信息，表字段的关联信息。一般是外键中声明，考虑到实际生产中使用外键情况不多，所以一般不会根据外键自动生成，而是手工维护。

稳定的工程化路线是：
> 把“库的语义知识”外置成可更新的**Schema Catalog（结构/语义目录）**，推理时按需检索，再让模型在受约束的上下文里生成 SQL，并用执行反馈闭环修正。

# Schema的维护
## 1) 先明确：要存的不只是 schema，而是“语义 schema”

仅存 `table/column/type` 往往不够，NL2SQL 真正决定准确率的是这些额外信息：

- **表/字段中文名、业务定义、口径说明、枚举含义**（例如 status=1/2/3 各代表什么）
    
- **主键/外键/关联路径、基数信息**（1对多/多对多）
    
- **常用过滤条件、默认时间粒度**（例如“最近7天”指 event_time 还是 created_at）
    
- **同义词/别名**（“GMV/交易额/成交金额”映射到哪个字段或派生指标）
    
- **敏感字段标识与权限**（列级脱敏、禁止查询、行级策略）
    
- **常用派生指标定义**（例如留存、ARPU、转化率的计算式）
    

因此你需要一个“可检索、可治理、可更新”的 Catalog，而不是静态文本。

## 2) Schema 信息怎么存：Postgres 做“权威源”，Weaviate 做“检索视图”

### 2.1 Postgres 中的 Catalog 表（建议你建 3 张）

**(1) table_catalog**

- table_name
    
- table_cn_name（可选）
    
- description（表用途、粒度）
    
- primary_key
    
- row_count_estimate（可选）
    
- updated_at
    

**(2) column_catalog**

- table_name
    
- column_name
    
- data_type
    
- column_cn_name（可选）
    
- description（业务定义/口径）
    
- synonyms（数组或 JSONB）
    
- is_sensitive（bool）
    
- sample_values（可选，枚举型特别有用）
    
- updated_at
    

**(3) relation_catalog**

- left_table, left_column
    
- right_table, right_column
    
- relation_type（fk / logical）
    
- cardinality（m:1 等，可选）
    
- updated_at
    

这些表可以由脚本定时从 `information_schema` / `pg_catalog` 同步，再由你手工补充中文名、口径、同义词。

### 2.2 Weaviate 存什么

在 Weaviate 建两个 class 就够了：

- `ColumnDoc`：一列一个向量
    
- `MetricDoc`（可选）：指标定义（如果你有指标层）
    

`ColumnDoc` 建议字段：

- id（table.column）
    
- table_name, column_name
    
- text（拼接：中文名 + 英文名 + description + synonyms + sample_values）
    
- metadata（json：data_type、敏感、更新时间等）


# Schema检索
推荐流程：
- **Query 解析**：抽取实体/时间/指标/维度/过滤词（可用 LLM 或规则）
    
- **候选召回**：
    
    - 向量：召回 ColumnDoc
        
    - 关键词：表名、字段名、别名、枚举值（BM25/全文索引）
        
- **候选重排（rerank）**：用更强的模型或交叉编码器重排前 50→前 10
    
- **Join 路径搜索**：在关系图上找最短可行 join（并考虑基数/方向）
    
- **SQL 生成**：只允许使用“白名单候选表/字段”
    
- **SQL 验证**：
    
    - 语法解析（sqlglot 等）
        
    - 逻辑规则：必须有 limit/时间过滤、禁止 select *、禁止笛卡尔积
        
- **Explain/试跑**：先 `EXPLAIN` 或小 limit 执行，必要时自我修正

## 混合检索的主流架构Sparse 召回 + Dense 召回 + 融合
### 1.1 召回侧（Candidate Generation）

建议至少两路召回：

**A. Sparse（关键词/BM25/全文索引）**  
适合命中：

- 表名/字段名（含下划线、缩写）、枚举值、固定业务词（“退款”“下单”“支付成功”）
    
- 用户输入里出现了具体 token（ID、手机号、地区代码、渠道名）
    

实现选项：

- Elasticsearch/OpenSearch：BM25 + 同义词词典 + ngram（用于下划线/拼写变体）
    
- Postgres FTS（tsvector）也能做，但 ES 更强
    

关键配置建议：

- 对字段名做 **snake_case 拆分 + camelCase 拆分 + 中文分词**
    
- 额外加一份 **keyword 字段**（不分词）用于精确匹配
    
- 对枚举值、别名（synonyms）提高权重（boost）
    

**B. Dense（向量检索）**  
适合命中：

- 中文业务语义（“成交额=GMV”“活跃=DAU”）
    
- 口径描述、字段释义（不是字段名本身）
    

实现选项：

- Milvus / Qdrant / Weaviate / pgvector
    
- 向量模型建议选择中文效果好的 embedding（你们若已定供应商就沿用）
    

关键配置建议：

- 向量检索 topK 不要太大：一般 **30–100** 足够（越大噪声越多）
    
- 文档粒度以 **ColumnDoc/MetricDoc** 为主，TableDoc 为辅（表级描述用来解释粒度、分区键等）
    

### 1.2 融合侧（Fusion）

融合方式推荐两种（按复杂度）：

**方案 1：加权 RRF（Reciprocal Rank Fusion）**（最稳、最常用）

- 对两路召回各取 topK（如 sparse 50 + dense 50）
    
- 用 RRF 融合得到一个统一排名
    
- 再送 rerank  
    优点：不需要分数对齐，工程实现简单，稳定性高。
    

**方案 2：加权线性融合（score normalization）**

- 对 BM25 和 cosine 分数做 min-max 或 z-score 归一化
    
- 按业务权重求和  
    优点：可控性强；缺点：需要调参、不同 query 分数分布波动会导致不稳。
    

工程上我更建议 **RRF 起步**，后续再考虑线性融合。

## 2) Rerank：把“相关”变成“正确”（尤其是字段）

混合召回解决“**别漏**”，rerank 解决“**别选错**”。

### 2.1 经典有效的两级 rerank（推荐）

**一级：轻量 rerank（Cross-Encoder 或小 reranker）**

- 输入：用户 query + 候选 ColumnDoc（一般 50–200 条）
    
- 输出：重排 topN（10–30 条）
    

常用开源路线：

- bge-reranker 系列（中文表现普遍不错）
    
- jina-reranker 系列
    
- 也可以用你们已有的 rerank API（若公司有统一组件）
    

**二级：LLM 语义裁决（可选，但对 NL2SQL 很有价值）**

- 只对 top20 左右做“结构化打分”，让模型输出：
    
    - 该候选是否可解释用户意图（0/1）
        
    - 需要的聚合粒度/时间字段/过滤字段建议
        
    - 若是指标，返回依赖字段/表
        
- 这个阶段不需要长文本，只给紧凑的 schema 片段即可
    

这样做的好处是：你把 LLM 从“海量信息里找”变成“在少量候选里做裁判”，稳定性会明显提升。

### 2.2 Rerank 的输入格式建议（很关键）

对每个候选（字段/指标）构造统一文本，示例：

- `type: column`
    
- `table: orders`
    
- `column: pay_amount`
    
- `cn_name: 支付金额`
    
- `definition: 已支付订单的实付金额（含优惠后）`
    
- `synonyms: 成交额, GMV(若等价), 实收`
    
- `constraints: 仅对 pay_status=paid 有意义`
    
- `privacy: non_sensitive`
    

将这些字段拼成短文本送 rerank，比直接丢 DDL 更有效。

## 3) 关系与 Join 路径：不要只检索表字段，还要检索“关系证据”

很多 NL2SQL 失败并非字段不对，而是 join 错。

建议你把 `RelationDoc` 也纳入检索与 rerank：

- `A.user_id = B.id`
    
- 基数：many-to-one
    
- 推荐方向：fact -> dim
    
- 约束：仅在同一 tenant_id 下有效
    

流程建议：

1. 先确定核心事实表（fact）候选（通过指标/事件词）
    
2. 再确定维度表候选（地区、渠道、用户属性等）
    
3. 在关系图上做最短路径搜索（限制跳数 ≤ 2 或 3）
    
4. 把“路径证据”注入到最终 SQL 生成上下文

## 4) 可落地方案
- 文档库：ColumnDoc + RelationDoc + TableDoc
    
- Sparse：BM25 top 60
    
- Dense：向量 top 60
    
- Fusion：RRF（k=60，权重 sparse:dense = 1:1 起步）
    
- 一级 rerank：对融合后的 top 120 做 cross-encoder rerank
    
- 二级 LLM 裁决：取 rerank top 20，输出 top 8 “允许使用”的字段/指标 + 理由
    
- SQL 生成：强约束只允许使用这 top8 涵盖的表字段 + 关系路径白名单

### 3.1 Postgres 侧（sparse）

用 FTS 或 trigram 都行。规模小我建议 trigram + 简单权重：

- 对 `table_name`、`column_name`、`synonyms` 做 trigram 相似度
    
- 对中文描述做 FTS
    

你可以返回一个 `sparse_score`，例如：

- 精确命中 column_name：+5
    
- trigram 相似度：+ (0~1)
    
- 命中 synonyms：+2
    
- 命中 table_name：+1
### 3.2 Weaviate 侧（dense）

对用户 query 做 embedding，向量检索 `nearText` / `nearVector`，返回 `dense_score`（距离转相似度即可）。

### 3.3 融合（RRF）

对两路各取 topK（你这里 K=10 足够），用 RRF 合并：

- `rrf(doc) = Σ 1 / (k + rank_i(doc))`
    
- k 取 20 或 60 都行；规模小差异不大
    

输出融合后的 topN（比如 top 8 列 + 涉及表）。

### 3.4 Rerank的替代方案：用 LLM 结构化裁决，比 cross-encoder 更划算
**输入：**

- 用户问题
    
- topN 候选 ColumnDoc（每个候选用“字段卡片”，几十字）
    
- relation_catalog（仅包含候选表之间的关系）
    

**输出（结构化 JSON）：**

- selected_tables
    
- selected_columns（按用途：select/group/filter/join）
    
- join_path（明确 join 条件）
    
- assumptions（如果口径不明确）
    

这一步本质就是 “rerank + schema linking + join planning” 三合一。


# 执行流程演示（MVP）
```sql
-- 1. 创建表 transport_bill
CREATE TABLE transport_bill (
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY,
    bill_no varchar(50) NOT NULL,
    vehicle_no varchar(20) NOT NULL,
    start_time timestamp NULL,
    end_time timestamp NULL,
    goods_id bigint NOT NULL,
    goods_name varchar(50) NOT NULL,
    goods_weight decimal(15,3) NULL,
    created_by_user bigint NOT NULL,
    created_by_user_name varchar(50) NULL,
    CONSTRAINT transport_bill_pkey PRIMARY KEY (id)
);

COMMENT ON TABLE transport_bill IS '运输运单表';
COMMENT ON COLUMN transport_bill.id IS '主键id';
COMMENT ON COLUMN transport_bill.bill_no IS '运单号，使用JKD开头';
COMMENT ON COLUMN transport_bill.vehicle_no IS '运输车牌号';
COMMENT ON COLUMN transport_bill.start_time IS '运输开始时间';
COMMENT ON COLUMN transport_bill.end_time IS '运输结束时间';
COMMENT ON COLUMN transport_bill.goods_id IS '运输货品id';
COMMENT ON COLUMN transport_bill.goods_name IS '运输货品名称';
COMMENT ON COLUMN transport_bill.goods_weight IS '运输货品数量，单位为吨';
COMMENT ON COLUMN transport_bill.created_by_user IS '创建用户id';
COMMENT ON COLUMN transport_bill.created_by_user_name IS '创建用户真实名称';


-- 2. 创建表 goods
CREATE TABLE goods (
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY,
    goods_code varchar(50) NOT NULL,
    goods_name varchar(20) NOT NULL,
    price decimal(10,3) NULL,
    created_time timestamp NOT NULL,
    updated_time timestamp NOT NULL,
    CONSTRAINT goods_pkey PRIMARY KEY (id)
);

COMMENT ON TABLE goods IS '货品表';
COMMENT ON COLUMN goods.id IS '主键id';
COMMENT ON COLUMN goods.goods_code IS '货品编码';
COMMENT ON COLUMN goods.goods_name IS '货品名称';
COMMENT ON COLUMN goods.price IS '货品单价，按吨来计算';
COMMENT ON COLUMN goods.created_time IS '创建时间';
COMMENT ON COLUMN goods.updated_time IS '更新时间';


-- 3. 创建表 user (注意：user是保留字，建议加双引号或改名为 sys_user)
CREATE TABLE "user" (
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY,
    user_code varchar(50) NOT NULL,
    user_name varchar(20) NOT NULL,
    user_phone varchar(20) NULL,
    user_sex smallint NULL,
    created_time timestamp NOT NULL,
    updated_time timestamp NOT NULL,
    CONSTRAINT user_pkey PRIMARY KEY (id)
);

COMMENT ON TABLE "user" IS '用户表';
COMMENT ON COLUMN "user".id IS '主键id';
COMMENT ON COLUMN "user".user_code IS '用户编码，登录账号';
COMMENT ON COLUMN "user".user_name IS '用户真实名称';
COMMENT ON COLUMN "user".user_phone IS '用户手机号';
COMMENT ON COLUMN "user".user_sex IS '用户性别 0女1男';
COMMENT ON COLUMN "user".created_time IS '创建时间';
COMMENT ON COLUMN "user".updated_time IS '更新时间';
```

## 1) 先补齐最关键的一件事：把隐式关系显式化
现在 DDL 里没有 FK 约束，但语义上明显存在：

- `transport_bill.goods_id` → `goods.id`
    
- `transport_bill.created_by_user` → `"user".id`
    

建议加外键（哪怕你生产不想强约束，也建议至少在 Catalog 里维护“逻辑外键”），否则 join 路径推断会更依赖 LLM，稳定性下降。

## 2) 文档（schema/语义）怎么组织：一列一文档 + 关系单独建档
### 2.1 Weaviate 建议两类对象就够

- `ColumnDoc`：每个字段一条（总共约 30 条）
    
- `RelationDoc`：每条关联一条（你这边 2 条）
    

如果你后续要做“运费、总价、GMV”等派生口径，再加 `MetricDoc`。

### 2.2 ColumnDoc 的“文本”应该包含哪些信息

对 NL2SQL 来说，最有效的是把列压成“字段卡片”，用短文本，避免塞 DDL：

**示例（transport_bill.goods_weight）：**

- table: transport_bill（运输运单表）
    
- column: goods_weight
    
- cn: 运输货品数量
    
- unit: 吨
    
- type: decimal(15,3)
    
- hints: 用于按吨计量、可与 goods.price 计算金额
    
- synonyms: 数量, 重量, 吨数, 运量
    

拼成一个 `text` 字段用于向量化，例如：

> “运输运单表.transport_bill.goods_weight：运输货品数量，单位吨，decimal(15,3)。同义词：重量/吨数/运量。常用于统计运输量、按吨计费。”

同样方式为每个字段构造 text。你 DDL 里已写了 COMMENT，这些 comment 是天然的高质量语义源。

## 3) Postgres 侧：Catalog 做权威源，并给 sparse 检索提供结构化打分

即便你用 Weaviate，仍建议把 schema/comment/synonyms/关系维护在 Postgres（便于更新、审计、权限裁剪）。

### 3.1 建三张 Catalog 表（足够支撑检索与执行约束）

- `table_catalog`
    
- `column_catalog`
    
- `relation_catalog`
    

字段设计我建议如下（可直接用）：

```sql
CREATE TABLE table_catalog (
  table_name text PRIMARY KEY,
  table_comment text,
  updated_at timestamptz default now()
);

CREATE TABLE column_catalog (
  table_name text NOT NULL,
  column_name text NOT NULL,
  data_type text NOT NULL,
  column_comment text,
  synonyms jsonb,               -- ["运单号","单号",...]
  is_sensitive boolean default false,
  unit text,
  updated_at timestamptz default now(),
  PRIMARY KEY (table_name, column_name)
);

CREATE TABLE relation_catalog (
  left_table text NOT NULL,
  left_column text NOT NULL,
  right_table text NOT NULL,
  right_column text NOT NULL,
  relation_type text NOT NULL,  -- 'fk' or 'logical'
  updated_at timestamptz default now(),
  PRIMARY KEY (left_table, left_column, right_table, right_column)
);

```

### 3.2 从系统表抓 COMMENT（自动化同步）

你现在的 COMMENT 已经在数据库里了，可以用 `obj_description/col_description` 抽取同步到 Catalog。同步脚本做一次即可，后续定时刷新。

## 4) 混合检索：在你这个规模下，“RRF + topK 很小”就足够稳

### 4.1 Sparse（Postgres）你可以做得非常简单但有效

在 `column_catalog` 上做：

- 精确命中（表名/列名/别名）：高分
    
- trigram 相似度（pg_trgm）：补充模糊匹配
    
- comment 的 FTS：补充中文描述
    

输出 sparse topK（建议 K=10~15）。

### 4.2 Dense（Weaviate）

对 `ColumnDoc.text` 做向量检索，取 dense topK（K=10~15）。

### 4.3 融合用 RRF

两路合并得到 20~30 个候选，再进入 rerank/裁决。你这里字段少，RRF 的收益主要在于“兼顾命名匹配与中文语义”。

---

## 5) Rerank 的最佳实践：你现在用“LLM 结构化裁决”就够了

你这点规模不值得上 cross-encoder reranker（部署和调参成本 > 收益）。建议用 LLM 做一个严格格式的裁决输出，相当于把 rerank + schema linking + join planning 合并：

**输入：**

- 用户问题
    
- RRF topN 的字段卡片（每条几十字）
    
- RelationDoc（候选表之间的 join 规则）
    
- 约束：只能用候选字段；涉及 `"user"` 必须引用 `"user"`；默认 limit；时间字段优先 `start_time/end_time/created_time`
    

**输出 JSON（示意）：**

- selected_tables: ["transport_bill","goods"]
    
- joins: [{"left":"transport_bill.goods_id","op":"=","right":"goods.id"}]
    
- select: ["goods.goods_name","sum(transport_bill.goods_weight) as total_ton"]
    
- filters: [{"col":"transport_bill.start_time","op":">=","value":"..."}]
    
- group_by: ["goods.goods_name"]
    
- confidence / assumptions
    

这一步做得好，你的 SQL 生成会非常稳定，且可解释、可审计。

---

## 6) 基于你这 3 张表，schema linking 的关键规则（建议固化为程序约束）

1. **优先用 ID 做 join**
    
    - `transport_bill.goods_id = goods.id`
        
    - `transport_bill.created_by_user = "user".id`
        
2. **避免用 name 字段做 join**  
    `transport_bill.goods_name` 与 `goods.goods_name` 存在冗余，理论上可能不一致。除非你明确保证一致，否则 join 用 id 更稳。
    
3. **金额类问题的默认计算**  
    你有 `goods.price`（按吨）和 `transport_bill.goods_weight`（吨），很多用户会问“金额/费用/总价”。建议在 MetricDoc 或规则里定义：
    

- amount = goods.price * transport_bill.goods_weight  
    并注明需要 join goods。
    

4. **人员类维度**  
    `transport_bill.created_by_user_name` 也是冗余字段，若要最新用户信息/手机号/性别，应 join `"user"` 表；若只要“当时录入人名称”，可以直接用 `created_by_user_name`（更快且不依赖 user 表完整性）。这一点可以作为 LLM 裁决的策略：
    

- 问“创建人手机号/性别” → 必须 join `"user"`
    
- 仅问“创建人名称” → 优先 `created_by_user_name`（除非要求以用户表为准）
    

---

## 7) 你可以立刻用来验收的一组测试问题（覆盖检索 + join + 计算）

1. “统计最近7天每种货品运输总吨数”  
    应选：transport_bill.goods_weight + start_time + goods_id → join goods → group by goods.goods_name
    
2. “查询运单号以JKD开头、车牌号为xxx的运单明细”  
    应选：transport_bill.bill_no（前缀）+ vehicle_no
    
3. “某个用户创建的运单数量，按天统计”  
    应选：created_by_user + start_time（或 created_time 若你有）→ group by date_trunc
    
4. “本月运输金额最高的货品”  
    应选：goods.price + goods_weight → join goods → sum(price * weight) → order desc limit