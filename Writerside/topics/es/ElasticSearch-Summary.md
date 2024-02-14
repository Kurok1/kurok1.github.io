# ElasticSearch Summary

## config

- cluster.name 集群名称

- node.name 当前节点名称

- path.data 数据存储路径

- path.logs 日志存储路径

- http.port 对外服务暴露的端口

- network.host 本机网络ip，不要配置成0.0.0.0

- discovery.seed_hosts 服务节点列表

- cluster.initial_master_nodes 新集群master候选者列表




## concept

### 1.集群(Cluster)

多个协同工作的ES实例组合成集群，具备**高可用性**和**可拓展性**

### 2.节点(Node)

单个运行中的ES进程，视为一个节点。节点分类：

* 主节点(Master)。主节点在整个集群中是唯一的，Master节点是从有资格进行选举的节点中选举出来的(`cluster.initial_master_nodes`)。主要负责集群的变更，元数据的变更。
* 数据节点(Data Node)。负责存储数据，同时负责执行数据相关操作，如：CRUD，搜索，聚合等。该类型节点对机器的cpu，内存等相关配置要求较高
* 协调节点(Coordinating Node)。负责接受客户端的请求，分发路由，并将结果反馈。因为需要处理结果集并排序，故对cpu和内存资源要求较高
* 预处理节点(Ingest Node)。预处理节点支持自定义processors和管道，在写入文档时对数据进行转换。节点启动后默认为预处理节点
* 冷暖节点(Hot & Warm Node)。对数据访问频率进行划分，高频数据放在Hot Node，低频数据放在Warm Node。可以显著降低集群部署成本

节点配置（7.8版本以及之前）：

- node.master master节点，默认为true
- node.data 数据节点，默认为true
- node.ingest 预处理节点，默认为true

**如需要指定协调节点，上述三个选项设置为false即可**

在7.8版本以后，节点配置使用**node.roles**来确定类型，如：

```
node.roles: [master,data]
```

可选值如下：

| 参考值                | 选项说明                                                     |
| --------------------- | ------------------------------------------------------------ |
| master                | master候选节点，真正master会从这些节点中选举出来             |
| voting_only           | 参与master选举的节点，只有投票权限，**不会成为master**       |
| data                  | 数据节点,主要保存文档的分片数据                              |
| data_content          | 常规的数据文档存储节点                                       |
| data_hot              | 此节点会根据数据写入 ES 的时间存储时序数据，例如日志数据，data_hot 节点对数据读写要求快速，应当使用 SSD 存储。 |
| data_warm             | 此节点会将`data_hot`中查询相对不频繁的数据存储起来，此时仍然允许对数据进行修改 |
| data_cold             | 当数据不再需要定期按时序搜索时，数据会从`data_warm`转移到`data_cold`，此时数据为只读状态 |
| data_forzen           | 当数据几乎不再被查询时候，数据会从`data_cold`到`data_forzen`状态，视为归档状态 |
| ingest                | 预处理数据节点                                               |
| ml                    | 提供机器学习功能                                             |
| remote_cluster_client | 充当跨集群客户端并连接到其他集群。                           |
| transform             | 允许运行transform相关api                                     |



### 3.分片和副本

分片（Shard）是 ES 底层基本的读写单元，分片是为了分割巨大的索引数据，让读写可以由多台机器来完成，从而提高系统的吞吐量。

为了保证数据可靠性，一般分布式系统都会对数据进行冗余备份，这个备份也就是副本了。**ES将数据副本分为主从两类型：主分片（primary shard）和副分片（replica shard）** 。在写入的过程中，先写主分片，成功后并发写副分片，在数据恢复时以主分片为主。多个副本除了可以保证数据可靠性外，还有一个好处是可以承担系统的读负载。

### 4.集群健康状态

通过集群的健康状态，我们可以了解集群是不是出现问题了。 集群健康状态有以下 3 种。

- **Green**，集群处于健康状态，所有的主分片和副本分片都正常运行。
- **Yellow**，所有的主分片都运行正常，但是有部分副本分片不正常，意味着可能存在单点故障的风险（如果部分主分片没有备份了，一旦这个主分片数据丢失，将导致这些数据永久丢失）。如果集群只有 3 个数据节点，但是分配了 4 个副本（主分片 + 副本分片的总数），这个时候有一个副本无法分配的，因为相同的两份数据不应该被分配到同一个节点上。
- **Red**，有部分主分片没有正常运行。

需要注意的是，每个索引也有这三种状态，**如果索引丢失了一个副本分片，那么这个索引和集群的状态都变为 Yellow 状态，但是其他索引的的状态仍为 Green**。

### 5.文档（Document）

ES中的每一条数据都可以视为一个文档。对应数据库系统中的一行记录

### 6.索引（Index）

一系列相关文档的集合，如用户相关数据统一存放在用户索引中(User Index)。对应数据库系统中的一张表

### 7.字段映射（Mappings）

用于定义一个索引下的文档，有哪些字段，字段是什么类型。有两个重要作用

- 定义了索引中各个字段的名称和对应的类型；
- 定义各个字段、倒排索引的相关设置，如使用什么分词器等。



## search

#### 1.match(匹配查询)

匹配查询可以用于全文本查询，精确字段。示例如下：

```
GET /book/_search
{
  "query": {
    "match": {
      "name": {
        "query": "linux kernel",
        "operator": "and"
      }
    }
  }
}
```

参数说明：

* name 表示查询匹配的字段
* query 查询的值，多个值用空格分割
* operator 查询值的组合关系，默认为or，该例子使用and，表示与关系
* minimum_should_match, 允许指定词项的最少匹配个数，可以填写固定数字，如果无法评估数量，可以设置为百分比，如75%

上面的例子使用了match api进行了一次全文本的搜索，表示要求name字段必须同时包含`linux`和`kernel`这两个词语



### 2.match phrase(短语匹配)

一般来说短语匹配与`match`的查询逻辑保持一致，但是短语匹配要求词语出现的顺序与条件的顺序保持一致

```
GET /book/_search
{
    "query": {
        "match_phrase": {
            "name": {
                "query": "linux kernel"
            }
        }
    }
}
```

查询结果如下：

```json
"hits": {
        "total": {
            "value": 1,
            "relation": "eq"
        },
        "max_score": 0.5753642,
        "hits": [
            {
                "_index": "book",
                "_type": "_doc",
                "_id": "1",
                "_score": 0.5753642,
                "_source": {
                    "book_id": "4ee82462",
                    "name": "Dive into the Linux kernel architecture",
                    "author": "Wolfgang Mauerer",
                    "intro": "The content is comprehensive and in-depth, appreciate the infinite scenery of the Linux kernel.",
                    "price": 20.9,
                    "date": "2010-06-01"
                }
            }
        ]
    }
```

如果我们将`query`对应的词语从`linux kernel`改成`linux architecture`,那么将没有结果反馈



### 3.match phrase prefix(短语前缀匹配)

match phrase prefix 与 match phrase 类似，但最后一个词项会作为前缀，并且匹配这个词项开头的任何词语。可以使用 max_expansions 参数来控制最后一个词项的匹配数量，此参数默认值为 50。

```
GET /book/_search
{
  "query": {
    "match_phrase_prefix": {
      "name": {
        "query": "linux kern",
         "max_expansions": 2
      }
    }
  }
}
```



### 4.multi match

multi match建立在match查询的基础上，允许多个字段执行同一查询

```
GET /book/_search
{
  "query": {
    "multi_match": {
      "query": "linux architecture",
      "fields": ["nam*", "intro^2"],
      "type": "best_fields",
      "tie_breaker": 0.3
    }
  }
}
```

在该示例中，fields参数上一个列表，里面的元素是需要查询的字段。**fields 中的值既可以支持以通配符方式匹配文档的字段，又可以支持提升字段的权重**。如`nam*`就是用通配符的方式，`intro^2`表示对于`intro`的相关性评分*2

type字段表示其执行类型，有如下可选值:

1. **best_fields**: 默认的类型，会执行 match 查询并且将所有与查询匹配的文档作为结果返回，但是只使用评分最高的字段的评分来作为评分结果返回。
2. **most_fields**: 会执行 match 查询并且将所有与查询匹配的文档作为结果返回，并将所有匹配字段的评分加起来作为评分结果。
3. **phrase**: 在 fields 中的每个字段上均执行 match_phrase 查询，并将最佳匹配字段的评分作为结果返回。
4. **phrase_prefix**: 在 fields 中的字段上均执行 match_phrase_prefix 查询，并将最佳匹配字段的评分作为结果返回。
5. **cross_fields**：它将所有字段当成一个大字段，并在每个字段中查找每个词。例如当需要查询英文人名的时候，可以将 first_name 和 last_name 两个字段组合起来当作 full_name 来查询。
6. **bool_prefix**：在每个字段上创建一个`match_bool_prefix`查询，并且合并每个字段的评分作为评分结果。

一般来说文档的相关性算分由得分最高的字段来决定的，但当指定 "tie_breaker" 的时候，算分结果将会由以下算法来决定：

1. 令算分最高的字段的得分为 s1
2. 令其他匹配的字段的算分 * tie_breaker 的和为 s2
3. 最终算分为：s1 + s2

"tie_breaker" 的取值范围为：[0.0, 1.0]。当其为 0.0 的时候，按照上述公式来计算，表示使用最佳匹配字段的得分作为相关性算分。当其为 1.0 的时候，表示所有字段的得分同等重要。当其在 0.0 到 1.0 之间的时候，代表其他字段的得分也需要参与到总得分的计算当中去。通俗来说就是**其他字段可以使用 "tie_breaker" 来进行“维权”** 。



## Term Level Query

**Term Level Query 会将输入的内容会作为一个整体来进行检索，并且使用相关性算分公式对包含整个检索内容的文档进行相关性算分**。

**Term 是文本经过分词处理后得出来的词项，是 ES 中表达语义的最小单位**。ES 中提供很多基于 Term 的查询功能，下面几个 API 是我们今天将会介绍的：

- **Term Query**，返回在指定字段中准确包含了检索内容的文档。
- **Terms Query**，跟 Term Query 类似，不过可以同时检索多个词项的功能。
- **Range Query**，范围查询。
- **Exist Query**，返回在指定字段上有值的文档，一般用于过滤没有值的文档。
- **Prefix Query**，返回在指定字段中包含指定前缀的文档。
- **Wildcard Query**，通配符查询。



### 1.Term Query

精确查询某个字段的值。注意，**如果要对 text 类型的字段进行搜索，应该使用 match API 而不是 Term Query API。**

```
POST book/_search
{
  "query": {
    "term": {
      "book_id": {
        "value": "4ee82463"
      }
    }
  }
}
```

### 2.Terms Query

跟Term Query类似，不过支持多值查询

```
POST book/_search
{
  "query": {
    "terms": {
      "author": [ # 数组，可以指定多个作者的名字
        "Stephen Hawking",
        "Wolfgang Mauerer"
      ]
    }
  }
}
```

### 3.Range Query

范围查询，允许查询字段符合范围的数据

```
POST books/_search
{
  "query": {
    "range": {
      "price": {
        "gte": 10.0,
        "lt": 20.0
      }
    }
  }
}
```

对于大小的比较可以查看以下列表：

- **gt**：表示大于
- **gte**: 表示大于或者等于
- **lt**: 表示小于
- **lte**: 表示小于或者等于

### 4.Exist Query

Exist Query Api可以用于查询指定字段有值的文档，其他判断条件为：

- 字段的 JSON 值为 null 或者 []，如果一个字段压根不存在于文档的 _source 里，也被认为是空的。
- 一个字段在 Mapping 定义的时候设置了 "index" : false。
- 一个字段的值的长度超出了 Mapping 里这个字段设置的 ignore_above 时。
- 当字段的值不合规，并且 Mapping 中这个字段设置了 ignore_malformed 时。

```
# 查询出所有存在 "price" 字段的文档
POST book/_search
{
  "query": {
    "exists": {
      "field": "price"
    }
  }
}

# 查询出所有存在 "press"（出版社） 字段的文档
POST book/_search
{
  "query": {
    "exists": {
      "field": "press"
    }
  }
}
```

### 5.Prefix Query

允许查询指定字段中包含特定前缀的文档

```
POST book/_search
{
    "query": {
        "prefix": {
            "name": {
                "value": "lin"
            }
        }
    }
}
```

**需要注意的是，text 类型的字段会被分词，成为一个个的 term，所以这里的前缀匹配是匹配这些分词后term！**

### 6.Wildcard Query

xWildcard Query允许使用通配符的方式进行匹配，支持两种通配符

* ?, 匹配任意字符
* *, 匹配0或者多个字符

```
POST book/_search
{
  "query": {
    "wildcard": {
      "name": "linu*"
    }
  }
}
```

注意，文档在存储中往往是按照倒排的方式进行存储，因此要尽量避免左通匹配模式，如`*inux`



## Suggest

用于词项输入推荐，Suggesters 会将输入的文本分解为 token（token 就是根据规则切分文本后一个个的词），然后在索引里查找相似的 Term。根据使用场景的不同，ES 提供了以下 4 种 Suggester：

- **Term Suggester**：基于单词的纠错补全。
- **Phrase Suggester**：基于短语的纠错补全。
- **Completion Suggester**：自动补全单词，输入词语的前半部分，自动补全单词。
- **Context Suggester**：基于上下文的补全提示，可以实现上下文感知推荐。



### 1.Term Suggester

提供了基于单词纠错和自定补全功能，是基于编辑距离来运作的，核心思想在于，**一个词需要改变多少个字符就可以和另一个词一致**。所以如果一个

转换成原词所需要的字符数越少，越可能是最佳匹配。

> The `term` suggester suggests terms based on edit distance. The provided suggest text is analyzed before terms are suggested. The suggested terms are provided per analyzed suggest text token. The `term` suggester doesn’t take the query into account that is part of request.
>

通用参数

* text 输入的文本

* field 匹配文档的字段

* analyzer 分词器。默认与字段的分词器一致

* size 每个单词提供的最大建议数量

* sort 建议排序结果，提供两个选项
  * score:优先按照相似性得分排序
  * frequency:优先按文档出现频率排序



示例：

```
GET /book/_search
{ 
  "suggest": {
    "my_suggest": {
      "text": "lnux kerne arch",
      "term": {
        "suggest_mode": "missing",
        "field": "name"
      }
    }
  }
}
```

全局suggest示例

```
GET /book/_search
{ 
  "suggest": {
    "text": "lnux kerne arch",
    "my_suggest": {
      "term": {
        "suggest_mode": "missing",
        "field": "name"
      }
    },
    "another_suggest": {
      "term": {
        "suggest_mode": "always",
        "field": "name"
      }
    }
  }
}
```



### 2.Phrase Suggester

Term Suggester主要场景是面向单个单词的，如果想要针对整个短语或者一句话做推荐，那么Term Suggester就显得无能为力了。

**Phrase Suggester 在 Term Suggester 的基础上增加了一些额外的逻辑，因为是短语形式的建议，所以会考量多个 term 间的关系，比如相邻的程度、词频等**。

```
GET /book/_search
{
  "suggest": {
    "my_suggest": {
      "text": "Brief Hestory Of Tome",
      "phrase": {
        "field": "name",
        "highlight": {
          "pre_tag": "<em>",
          "post_tag": "</em>"
        }
      }
    }
  }
}
```

highlight会将命中的关键词语用标签标注



### 3.Completion Suggester

**Completion Suggester 提供了自动补全的功能，其应用场景是用户每输入一个字符就需要返回匹配的结果给用户**。

**Completion Suggester 在实现的时候会将 analyze（将文本分词，并且去除没用的词语，例如 is、at这样的词语） 后的数据进行编码，构建为 FST 并且和索引存放在一起**

在使用 Completion Suggester 前需要定义 Mapping，对应的字段需要使用 "completion" type。

```
/GET /book/_search
{
  
  "suggest": {
    "my_suggest": {
      "prefix": "a brief hist",
      "completion": {
        "field": "name_completion"
      }
    }
  }
}
```

反馈结果如下：

```json
{
  "suggest" : {
    "my_suggest" : [
      {
        "text" : "a brief hist",
        "offset" : 0,
        "length" : 12,
        "options" : [
          {
            "text" : "A Brief History Of Time",
            "_id" : "2",
            "_source" : {
              "book_id" : "4ee82463",
              "name_completion" : "A Brief History Of Time",
              ......
            }
          }
        ]
      }
    ]
  }
}
```

