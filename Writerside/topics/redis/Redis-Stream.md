# Redis-Stream

## 基本概念

### 1.Stream {id="1-stream_1"}

`Stream`是Redis5.0之后提供的一种新数据类型，数据结构类似于列表+Hash对，但是只可进行追加操作，不可删除。

一个`Stream Key`下对应多个`Stream Entry`，`Entry`即为消息，每个`Entry`基于其插入顺序排序，同时包含一个或者多个`key-value`对。

### 2.消费者/消费者组 {id="2_1"}

消费者是用于接受消息的基本单位，每个消费者都有其唯一对应的消费者组，在消费者组中，通过`name`对每个消费者进行区分。

一条消息(`Stream Entry`)，只会被当前消费者组中的一个`消费者`消费,消息到底指定消费者后，消息变更成`PENDING`状态，不会被同组内其他消费者获取.

一条消费可以被多个消费者组消费



## 消息创建和查询

### 1.创建`Stream`并插入消息

`XADD key [NOMKSTREAM] [ MAXLEN | MINID [ = | ~] threshold [LIMIT count]] * | id field value [ field value ...]`

- key参数代表Stream数据对应的key，如果key不存在，则创建一个新的Stream数据
- [NOMKSTREAM]代表如果key对应Stream对象不存在，则不创建，反馈报错
- *｜id 表示允许指定消息id,id需要保证全局自增，如果使用`*`则表示由redis进行生成id。id默认生成规则如下为：`timestamp-sequence`，timestamp表示提交时间戳，sequence表示提交顺序，由0自增
- field value 表示对应的数据k-v对

### 2.消息范围查询 {id="2_2"}

`XRANGE key start end [COUNT count]`

- key参数代表Stream数据对应的key
- start表示消息id的开始区间。`-`指向消息id最小值
- end表示结束消息id的结束区间。`+`指向消息id最大值
- count表示最大返回结果数

返回结果为[start,end]区间内的消息。如果指定的是[-,+]，则等价于查询所有消息.

由于默认查询结果为闭区间，在**Redis6.2**以后，如果需要查询开区间的消息可以用()指定，如

`XRANGE key (start end)`

如果需要对查询结果反转，可以使用`XREVRANGE`命令

`XREVRANGE key end start [COUNT count]`

### 3.等值查询

范围查询时间复杂度为O(N),如果只需要查询特定id的消息，则可以使用`XREAD`命令

`XREAD [COUNT count] [BLOCK milliseconds] STREAMS key [key ...] id [id ...]`

- count 表示最大查询条数

- milliseconds 表示查询最大阻塞时间的毫秒数

- key id 对应Stream中消息id，允许多条Stream同时查询，但是key-id须一一对应,如

  `XREAD COUNT 2 STREAMS mystream1 mystream2 0-0 0-0`



## 消息消费

### 1.消息订阅

`XREAD [COUNT count] [BLOCK milliseconds] STREAMS key [key ...] id [id ...]`

- count 表示最大查询条数

- milliseconds 表示查询最大阻塞时间的毫秒数,0表示永远阻塞,直到有新消息反馈

- key id 对应Stream中消息id，id=$表示不订阅之前的消息数据。允许多条Stream同时订阅，但是key-id须一一对应,如

  `XREAD COUNT 2 STREAMS mystream1 mystream2 0-0 $`

### 2.消费者组的创建

`XGROUP CREATE key groupname id | $ [MKSTREAM] [ENTRIESREAD entries_read]`

- key 表示对应的Stream对象key
- groupname 消费者组名称，大小写敏感
- id | $ 告知redis这个消费者组从哪条消息开始消费，传递$表示之前的消息不会给当前消费者组消费。类似于Kafka中的offset概念
- [MKSTREAM] 不存在Stream则创建选项

### 3.重置消息消费id

`XGROUP SETID key groupname id | $ [ENTRIESREAD entries_read]`

类似于Kafka中的`seek`.重置消费id,如果需要重头开始消费，id指定为0即可

### 4.消息拉取给消费者

`XREADGROUP GROUP group consumer [COUNT count] [BLOCK milliseconds] [NOACK] STREAMS key [key ...] id [id ...]`

当消费者组创建后，会根据消费者组指定的消息id开始进行消息分配。即从起始消息位置开始，后续所有的消息均会分配给当前消费者组。

上诉命令则是将消费者组中的指定范围的消息推送给消费者

- group 消费者组名称

- consumer 消费者名称，大小写敏感

- [count] 最大条数

- [milliseconds] 最大阻塞时间

- key Stream key

- id 起始id位置，`>`表示当前消费者组所有剩余的消息均分配给当前消费者,参考redis官方文档

  > The ID to specify in the **STREAMS** option when using `XREADGROUP` can be one of the following two:
  >
  > - The special `>` ID, which means that the consumer want to receive only messages that were *never delivered to any other consumer*. It just means, give me new messages.
  > - Any other ID, that is, 0 or any other valid ID or incomplete ID (just the millisecond time part), will have the effect of returning entries that are pending for the consumer sending the command with IDs greater than the one provided. So basically if the ID is not `>`, then the command will just let the client access its pending entries: messages delivered to it, but not yet acknowledged. Note that in this case, both `BLOCK` and `NOACK` are ignored.



当消息分配给某一消费者后，其余消费者无法分配该消息。

改消息状态变更为`PENDING`，等待`ACK`

### 5.消息分配

前面提及，当消息分配给某一消费者后，其余消费者无法获得该消息。

但是Redis却额外提供了`XCLAIM`命令，用于将`PENDING`消息分配给指定的消费者

`XCLAIM key group consumer min-idle-time id [id ...] [IDLE ms] [TIME unix-time-milliseconds] [RETRYCOUNT count] [FORCE] [JUSTID]`

- key Stream key
- group 消费者组
- consumer 指定的消费者
- min-idle-time 消息最小的空闲时间，即该消息在当前消费者中多久没有被使用过
- id 消息id
- [IDLE ms] 重置消息空闲时间
- [TIME unix-time-milliseconds] 重置消息空闲时间,指定UNIX时间
- [RETRYCOUNT count] 设置消息重试次数
- [JUSTID] 返回处理成功的id列表

注意，分配的消息必须要处于`PENDING`状态

### 6.ACK

`XACK key group id [id ...]`

确认一条消息，确认后的消息无法被消费者组中的任何消费者消费



## 总结

`Redis Stream`的前身为`Redis Pub/Sub`模型。前者很好的解决了**消息不支持持久化**，**历史消息无法消费**和**点对点消费**的问题。与传统消息队列模型十分吻合，比较适合一些轻量级**生产者-消费者**业务场景