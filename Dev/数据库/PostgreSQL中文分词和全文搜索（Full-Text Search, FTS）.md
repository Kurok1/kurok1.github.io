# 部署开通中文分词插件的PostgreSQL（docker环境）

## 1) 自定义Dockerfile打包镜像
```Dockerfile
# 1. 选择基础镜像 (建议与生产环境版本一致，这里以 pg 16 为例)
FROM postgres:16

# 2. 设置环境变量，防止交互式提示打断构建
ENV DEBIAN_FRONTEND=noninteractive

# 3. 安装编译所需的依赖
# postgresql-server-dev-16 是开发头文件，必须与 postgres 版本对应
RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 \
    wget \
    make \
    gcc \
    libc6-dev \
    postgresql-server-dev-16 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 4. 下载并安装 SCWS (zhparser 的底层依赖)
WORKDIR /tmp
RUN wget -q -O scws-1.2.3.tar.bz2 http://www.xunsearch.com/scws/down/scws-1.2.3.tar.bz2 \
    && tar xjf scws-1.2.3.tar.bz2 \
    && cd scws-1.2.3 \
    && ./configure --prefix=/usr/local \
    && make \
    && make install

# 5. 下载并安装 zhparser
WORKDIR /tmp
# 使用 GitHub 的最新代码
RUN wget -q -O zhparser.tar.gz https://github.com/amutu/zhparser/archive/master.tar.gz \
    && tar xzf zhparser.tar.gz \
    && cd zhparser-master \
    && make \
    && make install

# 6. 清理编译垃圾 (可选，减小镜像体积)
WORKDIR /
RUN rm -rf /tmp/scws* /tmp/zhparser*
```

使用docker构建镜像
```sh
docker build -t my-postgres-zhparser:pg16 .
```

## 使用docker部署，并启动时加载插件
```yaml
services:
  postgres:
    image: my-postgres-zhparser:pg16
    container_name: postgres
    restart: unless-stopped

    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: appdb
      TZ: Asia/Shanghai

    ports:
      - "5432:5432"

    # 将容器内 PG 数据目录映射到本机目录
    volumes:
      - ./pgdata:/var/lib/postgresql/data
      # 初始化脚本：首次启动（数据目录为空时）自动启用 pgvector 扩展
      - ./initdb:/docker-entrypoint-initdb.d:ro

    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB"]
      interval: 10s
      timeout: 5s
      retries: 5
```
同时新增`initdb/initdb/001-enable-pgvector.sql`文件，作为数据库初始化脚本（加载插件使用）
```sql
-- 1. 启用扩展
CREATE EXTENSION IF NOT EXISTS zhparser;

-- 2. 创建一个名为 'chinese' 的全文搜索配置
CREATE TEXT SEARCH CONFIGURATION chinese (PARSER = zhparser);

-- 3. 配置分词规则 (映射)
-- 将名词(n)、动词(v)、形容词(a)、成语(i)、叹词(e)、习语(l) 映射到 simple 字典
-- simple 字典只是简单地转换成小写，适合中文处理
ALTER TEXT SEARCH CONFIGURATION chinese
    ADD MAPPING FOR n,v,a,i,e,l WITH simple;

-- (可选) 如果你希望忽略标点符号等，就不要把它们 ADD MAPPING 进来
```

完成后，使用docker compose启动即可
```sh
docker compose up -d
```

# 测试中文分词和FTS
## 1) 准备基础表结构和数据
```sql
-- 建表

CREATE TABLE docs (

    id SERIAL PRIMARY KEY,

    content TEXT,

    -- 这里的配置名 'english' 要改成 'chinese'

    tsv tsvector GENERATED ALWAYS AS (to_tsvector('chinese', content)) STORED
);

CREATE INDEX idx_docs_tsv ON docs USING GIN(tsv);

-- 插入数据

INSERT INTO docs (content) VALUES

('PostgreSQL是一个强大的开源关系型数据库系统'),

('zhparser是PostgreSQL的一个中文分词扩展'),

('今天天气真不错');
```

## 2) 测试查询
```sql
-- 查询 "分词"

SELECT * FROM docs WHERE tsv @@ to_tsquery('chinese', '分词');

-- 结果应包含第 2 条

  

-- 查询 "数据库 & 系统"

SELECT * FROM docs WHERE tsv @@ to_tsquery('chinese', '数据库 & 系统');

-- 结果应包含第 1 条

  

-- 查询 "PostgreSQL"

SELECT * FROM docs WHERE tsv @@ to_tsquery('chinese', 'PostgreSQL');

-- 结果应包含第 1 条和第 2 条
```


# 🚀 快速上手案例：博客文章搜索

假设我们有一个存储博客文章的表，包含 `title`（标题）和 `content`（内容），我们需要对这两列进行全文检索。

## 1. 创建表并准备数据

这里我们直接创建一个带有 **自动生成 `tsvector` 列** 的表。这是现代 PostgreSQL（12+）的最佳实践，它会自动维护搜索向量，无需手动触发器。

SQL

```
-- 创建表
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    -- 创建一个生成的 tsvector 列，自动组合标题和内容
    -- 'english' 是分词配置，中文场景下文会单独说明
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', title), 'A') || -- 标题权重设为 A (最高)
        setweight(to_tsvector('english', content), 'B')  -- 内容权重设为 B
    ) STORED
);

-- 插入测试数据
INSERT INTO posts (title, content) VALUES
('PostgreSQL Tutorial', 'PostgreSQL is a powerful, open source object-relational database system.'),
('Full Text Search', 'Search allows you to find documents based on the text content.'),
('Performance Tips', 'Using GIN index can significantly speed up your search queries.'),
('Database Indexing', 'Indexes are crucial for database performance, especially for text search.');
```

## 2. 创建索引 (关键步骤)

如果没有索引，数据库会全表扫描，速度很慢。对于全文搜索，**GIN 索引** 是标准配置。

SQL

```
-- 在生成的向量列上创建 GIN 索引
CREATE INDEX idx_posts_search ON posts USING GIN(search_vector);
```

## 3. 执行查询

### 使用 `@@` 操作符和 `to_tsquery` 函数进行搜索。

**基础搜索：** 查找包含 "database" 的文章。

SQL

```
SELECT id, title
FROM posts
WHERE search_vector @@ to_tsquery('english', 'database');
```

**逻辑组合搜索：** 查找包含 "PostgreSQL" 并且包含 "system" 的文章。

SQL

```
SELECT id, title
FROM posts
WHERE search_vector @@ to_tsquery('english', 'PostgreSQL & system');
```

**模糊/前缀搜索：** 查找以 "perf" 开头的词（如 performance）。

SQL

```
SELECT id, title
FROM posts
WHERE search_vector @@ to_tsquery('english', 'perf:*');
```

## 4. 结果相关性排序 (Ranking)

通常我们需要把“最匹配”的结果排在前面（比如命中标题的权重要高于命中内容的）。可以使用 `ts_rank`。

SQL

```
SELECT id, title, ts_rank(search_vector, query) as rank
FROM posts, to_tsquery('english', 'database | performance') query
WHERE search_vector @@ query
ORDER BY rank DESC;
```

### 总结

1. **核心字段：** `tsvector` (存储处理后的文本) 和 `tsquery` (存储查询条件)。
    
2. **核心操作符：** `@@` (匹配)。
    
3. **性能关键：** 务必使用 `USING GIN` 创建索引。
    
4. **自动化：** 使用 `GENERATED ALWAYS AS ... STORED` 列来自动管理向量数据。

# 词汇更新
如果你有一些专有名词（比如公司名“字节跳动”），默认可能分词为“字节”和“跳动”。 你可以在 Postgres 配置目录下的 `tsearch_data` 文件夹中添加 `dict.utf8.xdb`（zhparser 使用 SCWS 的词典格式）。但在 Docker 环境中，最简单的方法是通过 SQL 动态添加（临时生效）或映射宿主机目录。 _SQL 添加自定义词方式：_

SQL

```
insert into zhparser.zhprs_custom_word values('字节跳动');
SELECT zhprs_sync_dict_xdb(); -- 必须同步才会生效
```

但是需要注意的是：
> 添加自定义词汇（并执行 `zhprs_sync_dict_xdb()`）只会改变**未来**的某些操作（如新插入的数据、新执行的查询）的分词逻辑。
> 
   对于数据库中**已经存储在磁盘上**的 `tsvector` 数据，PostgreSQL **不会**自动回头去重新计算它们。这就导致了一个现象：**旧数据用旧规则，新数据用新规则，导致搜索结果不一致。**
   >
   你需要根据你的表结构，手动触发刷新。

## 字段刷新
### 场景一：你使用了生成列 (GENERATED ... STORED) —— 【最常见】

如果你采用了我之前推荐的方案，即表结构里有一个实际存储的 `tsvector` 列：

SQL

```
-- 你的表结构类似这样
tsv tsvector GENERATED ALWAYS AS (to_tsvector('chinese', content)) STORED
```

因为数据是 `STORED`（物理存储）的，你需要强制触发一次全表的“重算”。

#### 解决方法：

执行一个 Update 操作，告诉 PostgreSQL 重新计算该列的默认生成逻辑。

SQL

```
-- 强制更新 tsv 列，使用 DEFAULT 关键字触发重算
UPDATE docs SET tsv = DEFAULT;
```

_注意：PostgreSQL 会自动更新相关的 GIN 索引，不需要你手动重建索引。_

---

### 场景二：你只创建了索引 (Expression Index)

如果你没有创建单独的 `tsvector` 列，而是直接在索引里写的逻辑：

SQL

```
-- 没有 tsv 列，直接建索引
CREATE INDEX idx_content ON docs USING GIN (to_tsvector('chinese', content));
```

这种情况下，索引文件里的数据依然是旧的。

#### 解决方法：

你需要重建索引。

SQL

```
REINDEX INDEX idx_content;
```

---

### ⚠️ 生产环境的关键警告

如果你的数据量很小（几万条），上面的命令随便跑，几秒钟就完事。 但如果你的数据量有 **百万/千万级**，直接执行 `UPDATE docs SET tsv = DEFAULT` 会产生严重的后果：

1. **锁表：** 可能会锁住全表，导致业务中断。
    
2. **IO 爆炸：** 会产生大量的磁盘写操作和 WAL 日志（因为每一行都被重写了）。
    
3. **表膨胀：** PostgreSQL 的 UPDATE 本质是“标记旧行删除 + 插入新行”，这会导致表体积迅速膨胀，需要后续 VACUUM。