# MySQL笔记day3

## 间隙锁
以表结构`t`为例子
```SQL
CREATE TABLE `t` ( 
`id` int(11) NOT NULL, 
`c` int(11) DEFAULT NULL, 
`d` int(11) DEFAULT NULL, 
PRIMARY KEY (`id`), KEY `c` (`c`)
) ENGINE=InnoDB;
insert into t values(0,0,0),(5,5,5),(10,10,10),(15,15,15),(20,20,20),(25,25,25);
```
顾名思义，间隙锁，锁的就是两个值之间的空隙。比如下图所示，初始化插入了 6 个记录，这就产生了 7 个间隙。
![gap_lock](mysql_gap_lock1.png)

**这样，当你执行 select * from t where d=5 for update 的时候，就不止是给数据库中已有的 6 个记录加上了行锁，还同时加了 7 个间隙锁。这样就确保了无法再插入新的记录。**

也就是说这时候，在一行行扫描的过程中，不仅将给行加上了行锁，还给行两边的空隙，也加上了间隙锁。

在MySQL中，数据行是可以加上锁的实体，数据行之间的间隙，也是可以加上锁的实体。但是间隙锁跟我们之前碰到过的锁都不太一样。

比如行锁，分成读锁和写锁。下图就是这两种类型行锁的冲突关系。

|    | 读锁 | 写锁 |
|----|----|----|
| 读锁 | 兼容 | 冲突 |
| 写锁 | 冲突 | 冲突 |

但是间隙锁不一样，跟间隙锁存在冲突关系的，是“往这个间隙中插入一个记录”这个操作。间隙锁之间都不存在冲突关系。

间隙锁和行锁合称 next-key lock，每个 next-key lock 是前开后闭区间。也就是说，我们的表 t 初始化以后，如果用 select * from t for update 要把整个表所有记录锁起来，就形成了 7 个 next-key lock，分别是 (-∞,0]、(0,5]、(5,10]、(10,15]、(15,20]、(20, 25]、(25, +supremum]。

## Next-Key Lock加锁规则
1. 原则 1：加锁的基本单位是 next-key lock。且范围一定是左开右闭区间。
2. 原则 2：查找过程中访问到的对象才会加锁。
3. 优化 1：索引上的等值查询，给唯一索引加锁的时候，next-key lock 退化为行锁。
4. 优化 2：索引上的等值查询，向右遍历时且最后一个值不满足等值条件的时候，next-key lock 退化为间隙锁。
5. 一个 bug：主键索引上的范围查询会访问到不满足条件的第一个值为止。(MySQL8.0.18之前)

下面整理一些加锁案例，以下表t为例子
```SQL
CREATE TABLE `t` ( 
`id` int(11) NOT NULL, 
`c` int(11) DEFAULT NULL, 
`d` int(11) DEFAULT NULL, 
PRIMARY KEY (`id`), KEY `c` (`c`)
) ENGINE=InnoDB;

insert into t values(0,0,0),(5,5,5),
(10,10,10),(15,15,15),(20,20,20),(25,25,25);
```

### 1.唯一索引等值查询间隙锁

| session A                                    | session B                   | session C                         |
|----------------------------------------------|-----------------------------|-----------------------------------|
| begin;<br/>update t set d = d+1 where id = 7 |                             |                                   |
|                                              | insert into t values(8,8,8) |                                   |
|                                              |                             | update t set d=d+1 where id = 10; |

由于表中没有id=7的数据，而id=7最左侧的数据是id=5，最右侧的数据id=10，所以最初加锁范围是(5,10],但是并不满足**优化2**，所以退化成间隙锁，也就是锁定(5,10)，所以session B会被阻塞住，session C正常执行


### 2.非唯一索引等值查询

| session A                                                   | session B                          | session C                   |
|-------------------------------------------------------------|------------------------------------|-----------------------------|
| begin;<br/>select id from t where c = 5 lock in share mode; |                                    |                             |
|                                                             | update t set d = d+1 where id = 5; |                             |
|                                                             |                                    | insert into t values(7,7,7) |
加锁分析
1. session A执行时会加上Next-Key Lock，由于最左侧的数据是c=0,所以锁定的范围是(0,5],
2. 因为c不是唯一索引，所以需要向下遍历到第一个不符合条件的值才能停止，所以还会继续加锁，锁定范围是(5,10];
3. 根据优化2，c=10不满足c=5，所以会退化成间隙锁(5,10);
4. 最终锁定的范围，是(0,5]和(5,10)
5. 此时，我们注意到，增加的锁是在c列上的，而session B是用的主键索引，所以session B更新id=5的数据不受影响
6. session C插入c=7的数据，处于锁定的范围，所以会被阻塞。

### 3.唯一索引的范围查询

| session A                                                         | session B                                                         | session C                        |
|-------------------------------------------------------------------|-------------------------------------------------------------------|----------------------------------|
| begin;<br/>select * from t where id >= 10 and id < 11 for update; |                                                                   |                                  |
|                                                                   | insert into t values(8,8,8);<br/> insert into t values(13,13,13); |                                  |
|                                                                   |                                                                   | update t set d=d+1 where id = 15 |
加锁分析
1. session A检测到id=10的查询，所以加锁为(5,10],又因为**优化1**，退化成id=10这一行的行锁
2. 然后从id=10的位置往后遍历，找到第一个不满足条件的id=15的数据，此时又加上了(10,15]
3. 最终，锁定范围是id=10，和id=(10,15],session B的第一个执行会成功，第二个会失败，session C也会失败
4. 特别的，当MySQL 8.0.18后，修复了唯一索引的范围查询的一个bug，此时加锁范围不是(10,15],而是(10,15),在这个版本下，session C是正常执行的

### 4.非唯一索引的范围查询
| session A                                                       | session B                     | session C                       |
|-----------------------------------------------------------------|-------------------------------|---------------------------------|
| begin;<br/>select * from t where c >= 10 and c < 11 for update; |                               |                                 |
|                                                                 | insert into t values(8,8,8);  |                                 |
|                                                                 |                               | update t set d=d+1 where c = 15 |

加锁分析，这里加锁场景跟上面的一样，只是索引从id换成了c，那么在检查到id=10，加锁(5,10]后，由于不是唯一索引，所以不会退化成行锁。

最终加锁范围是(5,15],session B和session C都会失败

### 非唯一索引上的等值例子
这里我们先写入一条数据
```SQL
mysql> insert into t values(30,10,30);
```
这里插入了一条c=10，id=30的数据，也就说c=10存在两条记录(id不同)

| session A                                | session B                       | session C                       |
|------------------------------------------|---------------------------------|---------------------------------|
| begin;<br/>delete from t where c = 10;   |                                 |                                 |
|                                          | insert into t values(12,12,12); |                                 |
|                                          |                                 | update t set d=d+1 where c = 15 |
加锁分析
1. 由于是等值查询，先查到(c=10,id=10)的数据，此时加上next-key lock(5,10]
2. 后续继续查询，又查询到一条c=10的记录，加上next-key lock(10,10]
3. 继续查询，直到第一个不满足c=10的记录，此时退化成间隙锁(10,15)
4. 最终锁定范围是(5,15),所以session B不会锁定，session C正常执行

## binlog 写入机制
binlog 的写入逻辑比较简单：事务执行过程中，先把日志写到 binlog cache，事务提交的时候，再把 binlog cache 写到 binlog 文件中。

一个事务的 binlog 是不能被拆开的，因此不论这个事务多大，也要确保一次性写入。这就涉及到了 binlog cache 的保存问题。系统给 binlog cache 分配了一片内存，每个线程一个，参数 binlog_cache_size 用于控制单个线程内 binlog cache 所占内存的大小。如果超过了这个参数规定的大小，就要暂存到磁盘。
![binlog_write](binlog_write.png)

可以看到，每个线程有自己 binlog cache，但是共用同一份 binlog 文件。
* 图中的 write，指的就是指把日志写入到文件系统的 page cache，并没有把数据持久化到磁盘，所以速度比较快。\
* 图中的 fsync，才是将数据持久化到磁盘的操作。一般情况下，我们认为 fsync 才占磁盘的 IOPS。

write 和 fsync 的时机，是由参数 sync_binlog 控制的：
1. sync_binlog=0 的时候，表示每次提交事务都只 write，不 fsync；
2. sync_binlog=1 的时候，表示每次提交事务都会执行 fsync；
3. sync_binlog=N(N>1) 的时候，表示每次提交事务都 write，但累积 N 个事务后才 fsync。

因此，在出现 IO 瓶颈的场景里，将 sync_binlog 设置成一个比较大的值，可以提升性能。在实际的业务场景中，考虑到丢失日志量的可控性，一般不建议将这个参数设成 0，比较常见的是将其设置为 100~1000 中的某个数值。

但是，将 sync_binlog 设置为 N，对应的风险是：如果主机发生异常重启，会丢失最近 N 个事务的 binlog 日志。

## redo log 写入机制
redo log的写入会优先写入到redo log buffer，然后经过一段时间后，写到磁盘中，期间一条redo log会经历以下状态
![redo_log_statue](redo_log_statue.png)

1. 存在 redo log buffer 中，物理上是在 MySQL 进程内存中，就是图中的红色部分；
2. 写到磁盘 (write)，但是没有持久化（fsync)，物理上是在文件系统的 page cache 里面，也就是图中的黄色部分；
3. 持久化到磁盘，对应的是 hard disk，也就是图中的绿色部分。

日志写到 redo log buffer 是很快的，wirte 到 page cache 也差不多，但是持久化到磁盘的速度就慢多了。
为了控制 redo log 的写入策略，InnoDB 提供了 innodb_flush_log_at_trx_commit 参数，它有三种可能取值：

1. 设置为 0 的时候，表示每次事务提交时都只是把 redo log 留在 redo log buffer 中 ;
2. 设置为 1 的时候，表示每次事务提交时都将 redo log 直接持久化到磁盘；
3. 设置为 2 的时候，表示每次事务提交时都只是把 redo log 写到 page cache。

InnoDB 有一个后台线程，每隔 1 秒，就会把 redo log buffer 中的日志，调用 write 写到文件系统的 page cache，然后调用 fsync 持久化到磁盘。

实际上，除了后台线程每秒一次的轮询操作外，还有两种场景会让一个没有提交的事务的 redo log 写入到磁盘中。
* 一种是，redo log buffer 占用的空间即将达到 innodb_log_buffer_size 一半的时候，后台线程会主动写盘。注意，由于这个事务并没有提交，所以这个写盘动作只是 write，而没有调用 fsync，也就是只留在了文件系统的 page cache。
* 另一种是，并行的事务提交的时候，顺带将这个事务的 redo log buffer 持久化到磁盘。假设一个事务 A 执行到一半，已经写了一些 redo log 到 buffer 中，这时候有另外一个线程的事务 B 提交，如果 innodb_flush_log_at_trx_commit 设置的是 1，那么按照这个参数的逻辑，事务 B 要把 redo log buffer 里的日志全部持久化到磁盘。这时候，就会带上事务 A 在 redo log buffer 里的日志一起持久化到磁盘。

通常我们说 MySQL 的“双 1”配置，指的就是 sync_binlog 和 innodb_flush_log_at_trx_commit 都设置成 1。也就是说，一个事务完整提交前，需要等待两次刷盘，一次是 redo log（prepare 阶段），一次是 binlog。

## 组提交
前面提到过，如果采用“双1”配置，意味着每次事务提交就会有两次写磁盘行为，这样性能开销会很大。组提交就是用来优化的手段

这里，先介绍日志逻辑序列号（log sequence number，LSN）的概念。
LSN 是单调递增的，用来对应 redo log 的一个个写入点。每次写入长度为 length 的 redo log， LSN 的值就会加上 length。

假设有三个并发事务 (trx1, trx2, trx3) 在 prepare 阶段，都写完 redo log buffer，持久化到磁盘的过程，对应的 LSN 分别是 50、120 和 160。

1. trx1 是第一个到达的，会被选为这组的 leader；
2. 等 trx1 要开始写盘的时候，这个组里面已经有了三个事务，这时候 LSN 也变成了 160；
3. trx1 去写盘的时候，带的就是 LSN=160，因此等 trx1 返回时，所有 LSN 小于等于 160 的 redo log，都已经被持久化到磁盘；
4. 这时候 trx2 和 trx3 就可以直接返回了。

所以，一次组提交里面，组员越多，节约磁盘 IOPS 的效果越好。

如果你想提升 binlog 组提交的效果，可以通过设置 binlog_group_commit_sync_delay 和 binlog_group_commit_sync_no_delay_count 来实现。

* binlog_group_commit_sync_delay 参数，表示延迟多少微秒后才调用 fsync;
* binlog_group_commit_sync_no_delay_count 参数，表示累积多少次以后才调用 fsync。

这两个条件是或的关系，也就是说只要有一个满足条件就会调用 fsync。

## MySQL在IO瓶颈时的优化
1. 设置 binlog_group_commit_sync_delay 和 binlog_group_commit_sync_no_delay_count 参数，减少 binlog 的写盘次数。这个方法是基于“额外的故意等待”来实现的，因此可能会增加语句的响应时间，但没有丢失数据的风险。
2. 将 sync_binlog 设置为大于 1 的值（比较常见是 100~1000）。这样做的风险是，主机掉电时会丢 binlog 日志。
3. 将 innodb_flush_log_at_trx_commit 设置为 2。这样做的风险是，主机掉电的时候会丢数据。

不建议把 innodb_flush_log_at_trx_commit 设置成 0。因为把这个参数设置成 0，表示 redo log 只保存在内存中，这样的话 MySQL 本身异常重启也会丢数据，风险太大。而 redo log 写到文件系统的 page cache 的速度也是很快的，所以将这个参数设置成 2 跟设置成 0 其实性能差不多，但这样做 MySQL 异常重启时就不会丢数据了，相比之下风险会更小。
