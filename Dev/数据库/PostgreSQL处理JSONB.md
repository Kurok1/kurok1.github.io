# 底层剖析
下面按“**写入 → 存储 → 读取/执行 → 索引**”这条链路，把 PostgreSQL 在底层如何处理 **JSONB** 讲清楚（偏实现细节，不是 SQL 用法）。

## 1) 写入时：把文本 JSON “拆解+规范化”成二进制结构

- `jsonb` 输入时会走 JSON lexer/parser，把原始文本解析成内部结构（因此写入比 `json` 稍慢）。在 `json` 上，很多函数每次执行都要重解析文本；而 `jsonb` 由于已经是“拆解后的二进制格式”，后续处理通常更快。 ([PostgreSQL](https://www.postgresql.org/docs/current/datatype-json.html "PostgreSQL: Documentation: 18: 8.14. JSON Types"))
    
- 规范化行为（这也是很多人第一次踩坑的点）：
    
    - 不保留空白；
        
    - **不保留对象键顺序**；
        
    - **不保留重复键**（只保留最后一个）。 ([PostgreSQL](https://www.postgresql.org/docs/current/datatype-json.html "PostgreSQL: Documentation: 18: 8.14. JSON Types"))
        

## 2) 存储时：JSONB 是 varlena + “树形容器(JsonbContainer)”

JSONB 在磁盘上是一个 **varlena**（变长字段）对象：前 4 字节（或短 varlena 头）记录总长度。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

### 2.1 树结构与节点布局

`jsonb` 是一棵树：每个节点概念上由 **JEntry 头** + **可变长内容**组成。关键实现点是：

- **JEntry 和内容不挨着存**：  
    对于一个“容器节点”（数组/对象），它会先放一段 **JEntry 数组**（描述所有子节点），然后紧跟着放这些子节点的 **变长数据区**。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))
    
- 根节点没有 JEntry（因为没父容器来存它的 JEntry）。根节点开头是一个 `uint32 header`，用 flag 区分 object/array。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))
    
- “裸 scalar”（比如只存一个 `5`）在 jsonb 里会被包装成 **单元素数组**，并打上 `JB_FSCALAR | JB_FARRAY` 标记。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))
    

### 2.2 JEntry：用“长度 or 偏移”的折中换压缩率 + 随机访问

JEntry 的 32 位布局大意是：

- 低 28 位：存 **长度** 或 **end+1 偏移**
    
- 3 位：类型（string / numeric / bool / null / container）
    
- 最高位：标记低 28 位到底是“长度”还是“偏移” ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))
    

为什么要这么绕？因为：

- 如果每个元素都存 offset，JEntry 数组对 TOAST 压缩很不友好；
    
- 如果都只存 length，随机访问会退化成 O(N) 扫描累加；
    
- 所以 PG 采用 **stride**：每隔 `JB_OFFSET_STRIDE` 个 JEntry 存一次 offset，其余存 length。这样既更可压缩，又能做到“最坏看 stride 次”仍是 O(1) 级别的随机定位。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))
    

## 3) 读取/执行时：迭代器 + 对象键二分查找 + 需要时 detoast

### 3.1 对象查 key：二分查找

JSONB 对象内部的键是为了检索友好而组织的，取某个 key 时会对容器做 **binary search**（源码注释/实现里明确写了这一点）。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h.html "PostgreSQL Source Code: src/include/utils/jsonb.h File Reference"))  
这也是为什么 `jsonb` 不保留“键的原始顺序”——它更像一个可检索的 map 结构。 ([PostgreSQL](https://www.postgresql.org/docs/current/datatype-json.html "PostgreSQL: Documentation: 18: 8.14. JSON Types"))

### 3.2 数组随机取第 i 个：靠 JEntry 的 offset/length 规则定位

数组按下标取元素时，会根据 JEntry 的 offset/length 编码计算元素起始位置；实现上可能需要向前回溯到最近一个“存了 offset 的 JEntry”，再累加若干 length。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h.html "PostgreSQL Source Code: src/include/utils/jsonb.h File Reference"))

### 3.3 大 JSONB：TOAST（行外存储/压缩）参与

当一行太宽（典型阈值 ~2KB 级别）或某列值太大时，会触发 TOAST：

- 可能先压缩，再拆分为 ~2000 字节左右的 chunk 存到 TOAST 表；
    
- 主表里只留一个 18 字节的 TOAST 指针（带必要元信息）。 ([PostgreSQL](https://www.postgresql.org/docs/current/storage-toast.html "PostgreSQL: Documentation: 18: 66.2. TOAST"))  
    因此“读 jsonb”很多时候隐含了 **detoast / 解压 / 组装**成本。
    

## 4) 索引时：GIN 是主力；两套 opclass 走不同“抽取键”策略

### 4.1 GIN 支持哪些 jsonb 运算符

内置 GIN operator class 里，`jsonb_ops`（默认）与 `jsonb_path_ops` 都支持 `@>`，以及 `@?` / `@@`（jsonpath）。 ([PostgreSQL](https://www.postgresql.org/docs/current/gin.html "PostgreSQL: Documentation: 18: 65.4. GIN Indexes"))

### 4.2 jsonb_ops vs jsonb_path_ops：索引条目长什么样

- `jsonb_ops`：为**每个 key 和 value**建立独立索引条目；
    
- `jsonb_path_ops`：只为 value 建条目，但条目是“**value + 通往它的 key 路径**”的 hash，因此通常更小、更快（但支持的操作更少，也有“找不到空对象结构”等局限）。 ([PostgreSQL](https://www.postgresql.org/docs/current/datatype-json.html "PostgreSQL: Documentation: 18: 8.14. JSON Types"))
    

另外，`jsonb_ops` 的 GIN 条目是文本编码并带首字节 flag 区分 key/null/bool/num/string；过长的文本会被 hash 以避免超过索引项长度并节省空间，且出现 hash 条目时需要 recheck。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

### 4.3 jsonpath 如何用上索引

对 `@?` / `@@`，GIN 会从 jsonpath 表达式里抽取形如 `accessors_chain == constant` 的子句来做索引检索；不同 opclass 支持的 accessor（如 `.*`、`.**`）范围不同。 ([PostgreSQL](https://www.postgresql.org/docs/current/datatype-json.html "PostgreSQL: Documentation: 18: 8.14. JSON Types"))


# JSONB的结构

## 1) 顶层：`Jsonb`（on-disk datum）= varlena + root container

```c
typedef struct
{
    int32          vl_len_;   // varlena header
    JsonbContainer root;      // 根容器（array 或 object）
} Jsonb;
```

- `vl_len_` 是变长字段头（varlena），真正 JSONB 内容从 `root` 开始。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))
    
- 根容器的 header 里带 count + flags，可用 `JB_ROOT_IS_ARRAY/OBJECT/SCALAR`、`JB_ROOT_COUNT` 等宏直接读。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))
    

---

## 2) 核心：`JsonbContainer`（数组/对象节点）= header + JEntry[] + dataProper

`JsonbContainer` 的定义长这样：

```c
typedef struct JsonbContainer
{
    uint32 header;     // count + flags
    JEntry children[]; // 变长数组（flexible array member）
    // 紧接着：每个 child 的变长数据区（data proper）
} JsonbContainer;
```

([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

### 2.1 container 的“物理布局图”

把它按字节顺序画出来就是：

```text
JsonbContainer (array 或 object)
┌──────────────────────────────┐
│ uint32 header                │  ← count(低 28 bits) + flags(高位)
├──────────────────────────────┤
│ JEntry children[0]           │
│ JEntry children[1]           │  ← 一段连续的 JEntry 数组
│ ...                          │
│ JEntry children[n-1]         │
├──────────────────────────────┤
│ dataProper (varlen payloads) │  ← 紧跟着放所有 child 的变长内容
│   child0 payload             │
│   child1 payload             │
│   ...                        │
│   child(n-1) payload         │
└──────────────────────────────┘
```

关键点：**JEntry 和 payload 不挨着存**，而是“JEntry 数组在前，payload 区在后”。([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

### 2.2 header 的 flags / count

- `JB_CMASK`：count 掩码（低 28 位）
    
- `JB_FSCALAR / JB_FOBJECT / JB_FARRAY`：标识 scalar/对象/数组（高位 flag）  
    并提供 `JsonContainerIsArray/Object/Scalar` 等宏。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))
    

---

## 3) JEntry：32-bit 小头，描述“类型 + 长度/偏移”

`JEntry` 本质就是一个 `uint32`： ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

### 3.1 位布局图

源码注释里把它讲得很直白：低 28 位存 **长度**或 **end+1 偏移**，再上面 3 位存类型，最高位表示“这 28 位到底是 offset 还是 length”。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

```text
JEntry (uint32)
bit31          bit30..28           bit27..0
┌────────────┬───────────────────┬────────────────────────────┐
│ HAS_OFF    │ TYPE (3 bits)     │ OFF/LEN (28 bits)          │
└────────────┴───────────────────┴────────────────────────────┘

HAS_OFF=1: OFF/LEN 字段存 “end+1 offset”
HAS_OFF=0: OFF/LEN 字段存 “length”
```

对应的宏（掩码/类型）在头文件里就是：`JENTRY_OFFLENMASK`、`JENTRY_TYPEMASK`、`JENTRY_HAS_OFF` 以及 `JENTRY_ISSTRING/ISNUMERIC/ISBOOL.../ISCONTAINER`。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

### 3.2 为什么要“有时存 offset、有时存 length”？

因为压缩与随机访问的折中：如果全存 offset，JEntry 数组压缩性很差；全存 length，随机访问会退化。PG 的策略是**每隔 `JB_OFFSET_STRIDE` 个 entry 存一次 offset**，其他存 length；stride 默认 32。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

---

## 4) 数组 vs 对象：children[] 的逻辑排列完全不同

### 4.1 数组（array）

- `children[i]` 对应第 i 个元素（顺序就是数组顺序）。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))
    

```text
array container
children[]: [ elem0 ][ elem1 ][ elem2 ] ...
payload区 :  elem0_data elem1_data elem2_data ...
```

### 4.2 对象（object）

对象更“反直觉”：**先放所有 key，再放所有 value**，而且 key 是**按排序后的 key 顺序**放的；value 的顺序与 key 顺序一一对应。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

```text
object container (nPairs = k)
children[]: [ key0 ][ key1 ]..[ key(k-1) ][ val0 ][ val1 ]..[ val(k-1) ]
payload区 :   key0_data ...           ...         val0_data ...
```

这样做是为了让 key 更紧凑、更 cache-friendly，方便查找某个 key。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

---

## 5) “裸 scalar”在 on-disk 里怎么放？

jsonb 的根节点必须是 array 或 object；如果你存的是一个 scalar（比如 `5` 或 `"x"`），会被包装成“**单元素数组**”，并把 header flags 设成 `JB_FSCALAR | JB_FARRAY`。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

---

## 6) in-memory：`JsonbValue` / `JsonbPair` / `JsonbIterator`（操作时用）

这些结构不直接等同于 on-disk 布局，但看懂它们能帮助你读扩展/内核代码。

### 6.1 `JsonbValue`：内存里的“反序列化 union”

`JsonbValue` 有一个 `type`（`jbvNull/jbvString/jbvNumeric/.../jbvBinary` 等），以及一个 union，string/array/object/binary/datetime 都在里面。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

特别注意：

- `jbvBinary` 分支里存的是 **指向 on-disk 格式 `JsonbContainer*`** 的指针（以及长度）。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))
    
- `array.rawScalar` 用来标记“顶层 raw scalar 的伪数组”。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))
    

### 6.2 `JsonbPair`：构建对象时的临时结构（并负责去重策略）

`JsonbPair` 只在构建 Jsonb 时短暂使用，**不是 on-disk 表示**；它记录了 `order` 用于重复 key 去重，“last observed wins”。 ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

### 6.3 `JsonbIterator`：遍历容器（array/object）时的状态机

迭代器里会保存：

- 当前容器 `container`
    
- `children` 指向 JEntry 数组
    
- `dataProper` 指向 payload 区开头
    
- `curIndex/curDataOffset` 等游标  
    ([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))
    

# 案例参考
以此json作为参考

```json
{"b":2,"a":1,"c":[3,4,5],"d":{"e":6,"f":7}}
```

---

## 1) 先发生的规范化：对象 key 会被重排

- `jsonb` **不保留对象键顺序**（也不保留空白/重复键）。([PostgreSQL](https://www.postgresql.org/docs/current/datatype-json.html "PostgreSQL: Documentation: 18: 8.14. JSON Types"))
    
- 更细一点：对象 key 在内部会按一种“适合二分查找”的顺序排序：**先按 key 长度，再按二进制 memcmp**。([Doxygen](https://doxygen.postgresql.org/jsonb__util_8c_source.html?utm_source=chatgpt.com "src/backend/utils/adt/jsonb_util.c Source File"))
    

你的 key 都是 1 个字符，因此按字节序就是：

**a, b, c, d**

所以 jsonb 输出时通常会变成（顺序可能表现为这样）：

```json
{"a":1,"b":2,"c":[3,4,5],"d":{"e":6,"f":7}}
```

---

## 2) 根节点（root）一定是容器：这里是 object

在 `jsonb.h` 里，root 没有单独的 JEntry，而是直接以一个 `uint32 header` 开头区分 array/object；object/array 的子节点 JEntry 集中放在 children[]，后面跟 dataProper。([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

并且：**object 的 children[] 先放所有 key（按 key sort order），再放所有 value（顺序与 key 对应）**。([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

---

## 3) 根 object 的 JsonbContainer：4 个 pair → 8 个 children

根对象有 4 对键值：a,b,c,d

### 3.1 根容器布局图

```text
ROOT: JsonbContainer (OBJECT, nPairs=4)
┌──────────────────────────────────────────────┐
│ header = (JB_FOBJECT | count=4)              │  ← count 是 pair 数
├──────────────────────────────────────────────┤
│ children[0] = key "a"  (JENTRY_ISSTRING)     │
│ children[1] = key "b"  (JENTRY_ISSTRING)     │
│ children[2] = key "c"  (JENTRY_ISSTRING)     │
│ children[3] = key "d"  (JENTRY_ISSTRING)     │
│ children[4] = val 1    (JENTRY_ISNUMERIC)    │
│ children[5] = val 2    (JENTRY_ISNUMERIC)    │
│ children[6] = val [3,4,5] (JENTRY_ISCONTAINER│
│ children[7] = val {"e":6,"f":7} (CONTAINER)  │
├──────────────────────────────────────────────┤
│ dataProper (变长 payload 串起来放)           │
│   payload(key"a")  payload(key"b") ...       │
│   payload(val1) payload(val2) payload(array) │
│   payload(object)                             │
└──────────────────────────────────────────────┘
```

> 上面“key 全在前、value 全在后”这点是 jsonb 查 key 能做二分查找的重要原因。([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

---

## 4) “c”的 value：一个 array 容器（3 个元素）

jsonb.h 明确说：**array 的 children[] 按数组顺序**存每个元素。([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

### 4.1 c 对应的 array 容器图

```text
VALUE of "c": JsonbContainer (ARRAY, nElems=3)
┌────────────────────────────────────┐
│ header = (JB_FARRAY | count=3)     │  ← count 是元素数
├────────────────────────────────────┤
│ children[0] = 3  (NUMERIC)         │
│ children[1] = 4  (NUMERIC)         │
│ children[2] = 5  (NUMERIC)         │
├────────────────────────────────────┤
│ dataProper: payload(3) payload(4) payload(5)│
└────────────────────────────────────┘
```

---

## 5) “d”的 value：一个 object 容器（2 个 pair）

同样按规则：object 先 key 后 value。([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))  
并且 d 里的 key 是 e、f（长度相同，memcmp 顺序 e<f）。([Doxygen](https://doxygen.postgresql.org/jsonb__util_8c_source.html?utm_source=chatgpt.com "src/backend/utils/adt/jsonb_util.c Source File"))

### 5.1 d 对应 object 容器图

```text
VALUE of "d": JsonbContainer (OBJECT, nPairs=2)
┌──────────────────────────────────────┐
│ header = (JB_FOBJECT | count=2)      │
├──────────────────────────────────────┤
│ children[0] = key "e" (STRING)       │
│ children[1] = key "f" (STRING)       │
│ children[2] = val 6   (NUMERIC)      │
│ children[3] = val 7   (NUMERIC)      │
├──────────────────────────────────────┤
│ dataProper: payload("e") payload("f")│
│            payload(6) payload(7)     │
└──────────────────────────────────────┘
```

---

## 6) JEntry 里到底放 length 还是 offset？这个例子说明“怎么定位”

`jsonb.h` 解释了：JEntry 的低 28 位要么存 **length**，要么存 **end+1 offset**，高位 `HAS_OFF` 标记到底是哪一种；并且为了兼顾压缩和随机访问，构建时会按 `JB_OFFSET_STRIDE=32` 的策略“每隔一段存 offset，其余存 length”。([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))

但**读取时不会假设固定 stride 规律**，而是：

- 看目标 entry 自己有没有 `HAS_OFF`
    
- 若没有，则**向前回溯**到最近一个带 `HAS_OFF` 的 entry，把中间的 length 累加起来得到 offset。这个逻辑在 `getJsonbOffset()` 里写得很清楚：从 `index-1` 往回走，累加 `OFFLENFLD`，直到遇到 `HAS_OFF`。([Doxygen](https://doxygen.postgresql.org/jsonb__util_8c_source.html?utm_source=chatgpt.com "src/backend/utils/adt/jsonb_util.c Source File"))
    

### 用根对象的 children[] 举例（概念演示）

假设你要定位 `children[6]`（也就是 key="c" 对应的 value：那个 array 容器）在 dataProper 里的起始位置：

1. 先看 `children[6]` 自己 `HAS_OFF` 是否为 1（表示里面存的是 end+1 offset）。
    
2. 如果不是，就按 `getJsonbOffset()` 的思路：从 `children[5]`、`children[4]`…往前累加每个 entry 的 length，直到遇到某个 `HAS_OFF=1` 的 entry（如果容器很小，可能一个都没有，那么就等价于从 0 累加）。([Doxygen](https://doxygen.postgresql.org/jsonb__util_8c_source.html?utm_source=chatgpt.com "src/backend/utils/adt/jsonb_util.c Source File"))
    

> 例子根对象只有 8 个 children、嵌套对象/数组更小，通常会出现“绝大多数甚至全部 JEntry 都只存 length”的情况；但不管怎样，定位算法都靠 `HAS_OFF` 位自洽。([Doxygen](https://doxygen.postgresql.org/jsonb_8h_source.html "PostgreSQL Source Code: src/include/utils/jsonb.h Source File"))
