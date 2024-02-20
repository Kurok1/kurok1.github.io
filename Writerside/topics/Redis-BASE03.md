# Redis笔记day3

## Redis缓存淘汰策略
Redis的数据淘汰策略，我们按照是否会进行数据淘汰把它们分成两类：
1. 不进行数据淘汰的策略，只有 `noeviction` 这一种。
2. 会进行淘汰的 7 种策略，我们可以再进一步根据淘汰候选数据集的范围把它们分成两类：
   * 在设置了过期时间的数据中进行淘汰，包括 `volatile-random`、`volatile-ttl`、`volatile-lru`、`volatile-lfu`（Redis 4.0 后新增）四种。
   * 在所有数据范围内进行淘汰，包括 allkeys-lru、allkeys-random、allkeys-lfu（Redis 4.0 后新增）三种。

![redis_drop_strategy.png](redis_drop_stratgy.png)

默认情况下，Redis 在使用的内存空间超过 `maxmemory` 值时，并不会淘汰数据，也就是设定的 noeviction 策略。对应到 Redis 缓存，也就是指，一旦缓存被写满了，再有写请求来时，Redis 不再提供服务，而是直接返回错误。

我们再分析下 volatile-random、volatile-ttl、volatile-lru 和 volatile-lfu 这四种淘汰策略。它们筛选的候选数据范围，被限制在已经设置了过期时间的键值对上。也正因为此，即使缓存没有写满，这些数据如果过期了，也会被删除。

我们使用 EXPIRE 命令对一批键值对设置了过期时间后，无论是这些键值对的过期时间是快到了，还是 Redis 的内存使用量达到了 `maxmemory` 阈值，Redis 都会进一步按照 `volatile-ttl`、`volatile-random`、`volatile-lru`、`volatile-lfu` 这四种策略的具体筛选规则进行淘汰。
* `volatile-ttl` 在筛选时，会针对设置了过期时间的键值对，根据过期时间的先后进行删除，越早过期的越先被删除。
* `volatile-random` 就像它的名称一样，在设置了过期时间的键值对中，进行随机删除。
* `volatile-lru` 会使用 LRU 算法（最近不使用）筛选设置了过期时间的键值对。
* `volatile-lfu` 会使用 LFU 算法（最少使用频率）选择设置了过期时间的键值对。

allkeys-lru、allkeys-random、allkeys-lfu 这三种淘汰策略的备选淘汰数据范围，就扩大到了所有键值对，无论这些键值对是否设置了过期时间。它们筛选数据进行淘汰的规则是：
* `allkeys-random` 策略，从所有键值对中随机选择并删除数据。
* `allkeys-lru` 策略，使用 LRU 算法在所有数据中进行筛选。
* `allkeys-lfu` 策略，使用 LFU 算法在所有数据中进行筛选。

在 Redis 中，LRU 算法被做了简化，以减轻数据淘汰对缓存性能的影响。
具体来说，Redis 默认会记录每个数据的最近一次访问的时间戳（由键值对数据结构 RedisObject 中的 lru 字段记录）。
然后，Redis 在决定淘汰的数据时，第一次会随机选出 N 个数据，把它们作为一个候选集合。接下来，Redis 会比较这 N 个数据的 lru 字段，把 lru 字段值最小的数据从缓存中淘汰出去。

Redis 提供了一个配置参数 `maxmemory-samples`，这个参数就是 Redis 选出的数据个数 N。例如，我们执行如下命令，可以让 Redis 选出 100 个数据作为候选数据集：
```Shell
CONFIG SET maxmemory-samples 100
```