# MySQL笔记day4

## MySQL主从备份基本原理
如下图所示，就是MySQL基本的主从备份原理

![mysql_copy](mysql_copy.png)

在状态 1 中，客户端的读写都直接访问节点 A，而节点 B 是 A 的备库，只是将 A 的更新都同步过来，到本地执行。这样可以保持节点 B 和 A 的数据是相同的。

当需要切换的时候，就切成状态 2。这时候客户端读写访问的都是节点 B，而节点 A 是 B 的备库。

接下来，我们再看看节点 A 到 B 这条线的内部流程是什么样的。下图画出的就是一个 update 语句在节点 A 执行，然后同步到节点 B 的完整流程图。

![mysql_slave](mysql_slave.png)

1. 在备库 B 上通过 change master 命令，设置主库 A 的 IP、端口、用户名、密码，以及要从哪个位置开始请求 binlog，这个位置包含文件名和日志偏移量。
2. 在备库 B 上执行 start slave 命令，这时候备库会启动两个线程，就是图中的 io_thread 和 sql_thread。其中 io_thread 负责与主库建立连接。
3. 主库 A 校验完用户名、密码后，开始按照备库 B 传过来的位置，从本地读取 binlog，发给 B。
4. 备库 B 拿到 binlog 后，写到本地文件，称为中转日志（relay log）。
5. sql_thread 读取中转日志，解析出日志里的命令，并执行。

## binlog的三种格式
binlog的格式有三种，分别是：statement，row和mixed。下面来讨论下这三种格式的差异。

### 数据准备

mysql版本为`5.7.36`

数据库建表语句和初始化数据语句如下：

```sql
CREATE TABLE `t` ( 
  `id` int(11) NOT NULL, 
  `b` int(11) DEFAULT NULL, 
  `c` int(11) DEFAULT NULL, 
   KEY `b` (`b`), KEY `c` (`c`)，
) ENGINE=InnoDB;

insert into t values(1,1,1);
insert into t values(2,2,2);
insert into t values(3,3,4);
insert into t values(4,4,5);
insert into t values(5,5,4);
```

另外，如果你想查看当前MySQL使用的何种binlog格式，可以使用以下语句

```sql
show variables like 'binlog_format';
```


我的配置内容如下：

```text
# enable bin log
log_bin = master
# server id
server_id = 1
```
如果你想指定特定格式的binlog，可以在`my.cnf`文件中进行配置

```text
# set binlog format
binglog_format = [STATEMENT|ROW|MIXED]
```

### STATEMENT格式

我们先来看看在STATEMENT格式下，binlog是怎么记录delete操作的

```sql
delete from /* this is a comment */ t where id = 2;
```

注意如果你是在`mysql-clients`下进行操作，需要加上-c参数，防止客户端自动去掉注释代码

此时，我们查询下binlog日志

```sql
mysql> show binlog events in 'master.000001';
```

可以看到日志内容如下：

| master.000001 | 543  | Query | 1    | 622  | BEGIN                                                        |
| ------------- | ---- | ----- | ---- | ---- | ------------------------------------------------------------ |
| master.000001 | 622  | Query | 1    | 746  | use `test`; delete from /* this is a comment */ t where id = 2 |
| master.000001 | 746  | Query | 1    | 826  | COMMIT                                                       |

可以看到，**在STATEMENT格式内容下：SQL语句的内容被完完整整的记录了下来**

**特别的，我们再次重复执行这条sql，发现同样的的内容被再次记录了一次**

### ROW格式

我们再来看看在ROW格式下，binlog又是如何记录的，还是同样的sql

```sql
delete from /* this is a comment */ t where id = 2;
```

注意，这里id=2的记录实际上已经不存在了，也就是说这条sql的影响行数为0

此时我们查询binlog，并没有增加多余的日志，此时我们删除一条存在的记录

```sql
delete from t where id = 3;
```

再次查询binlog时，发现有新增的日志内容：

| master.000002 | 219  | Query       | 1    | 291  | BEGIN                           |
| ------------- | ---- | ----------- | ---- | ---- | ------------------------------- |
| master.000002 | 291  | Table_map   | 1    | 337  | table_id: 108 (test.t)          |
| master.000002 | 337  | Delete_rows | 1    | 385  | table_id: 108 flags: STMT_END_F |
| master.000002 | 385  | Xid         | 1    | 416  | COMMIT                          |

这里可以看到，**ROW格式的binlog，不再存储sql，而是转换成一系列操作事件**。在这个例子里，delete对应的事件就是Delete_rows，当然前提是影响了真实存在的数据

### STATEMENT格式和ROW格式的缺陷

在讨论MIXED格式之前，我们不妨先讨论下STATEMENT格式和ROW格式的优缺点。

STATEMENT格式很简单，直接存储了当前数据库上的所有sql操作，但是这种操作并不是完全安全的，我们考虑下以下场景

```sql
delete from t where b>=4 and c<=5 limit 1;
```

这条语句增加了limit参数，而且c和b两个条件都满足索引，那么两个索引都有机会被命中，那么就会出现以下的情况：

* 如果选择了索引b，那么找到第一条b>=4并且满足条件的记录后，并且删除后不再继续查找。最终删除了id=4的这条记录
* 如果选择了索引c，那么找到第一条c<=5并且满足条件的记录后，并且删除后不再继续查找。最终删除了id=5的这条记录

这样就很明确了，**由于可能使用的索引不一致，导致最终执行时影响数据不一致，这就是STATEMENT格式binlog的致命缺陷**



ROW格式则避免了这种问题的发生，**ROW格式在记录日志时则会记录真实的主键id，那么在主从同步的时候，操作的数据肯定是相同主键的数据**，不会出现上述情况，但是ROW格式并非完美无缺，由于ROW格式下的每个操作事件只会记录一个主键id，那么，如果我同时操作10万条数据，那么就会生成10万条操作事件，这样会占用更大的空间，也会间接导致主从同步的延迟增大。

### MIXED格式

前面提及了，STATEMENT格式在某些场景下会出现数据不一致的风险，ROW格式很好的解决了这一问题，但是代价确实牺牲了更大的空间。所以，MySQL就取了个折中方案，也就是有了MIXED格式的 binlog。MIXED格式的意思是，**MySQL自己会判断这条 SQL 语句是否可能引起主备不一致，如果有可能，就用 ROW格式，否则就用STATEMENT格式**。

## MySQL多线程复制 {id="mysql_1"}
所有的多线程复制机制，都是要把图 1 中只有一个线程的 sql_thread，拆成多个线程，也就是都符合下面的这个模型：

![mysql_multi_worker](mysql_multi_worker.png)

coordinator 就是原来的 sql_thread, 不过现在它不再直接更新数据了，只负责读取中转日志和分发事务。真正更新日志的，变成了 worker 线程。而 work 线程的个数，就是由参数 slave_parallel_workers 决定的。

coordinator 在分发的时候，需要满足以下这两个基本要求：
1. 不能造成更新覆盖。这就要求更新同一行的两个事务，必须被分发到同一个 worker 中。
2. 同一个事务不能被拆开，必须放到同一个 worker 中。

### 按表分发策略
按表分发事务的基本思路是，如果两个事务更新不同的表，它们就可以并行。因为数据是存储在表里的，所以按表分发，可以保证两个 worker 不会更新同一行。当然，如果有跨表的事务，还是要把两张表放在一起考虑的。

每个 worker 线程对应一个 hash 表，用于保存当前正在这个 worker 的“执行队列”里的事务所涉及的表。hash 表的 key 是“库名. 表名”，value 是一个数字，表示队列中有多少个事务修改这个表。

在有事务分配给 worker 时，事务里面涉及的表会被加到对应的 hash 表中。worker 执行完成后，这个表会被从 hash 表中去掉。

而coordinator分配事务的策略，是基于对应事务是否与worker存在冲突，这里的冲突是指，事务中需要修改的表，在对应worker的hash表中有值。
1. 如果跟所有 worker 都不冲突，coordinator 线程就会把这个事务分配给最空闲的 woker;
2. 如果跟多于一个 worker 冲突，coordinator 线程就进入等待状态，直到和这个事务存在冲突关系的 worker 只剩下 1 个；
3. 如果只跟一个 worker 冲突，coordinator 线程就会把这个事务分配给这个存在冲突关系的 worker。

这个按表分发的方案，在多个表负载均匀的场景里应用效果很好。但是，如果碰到热点表，比如所有的更新事务都会涉及到某一个表的时候，所有事务都会被分配到同一个 worker 中，就变成单线程复制了。

### 按行分发策略
要解决热点表的并行复制问题，就需要一个按行并行复制的方案。按行复制的核心思路是：如果两个事务没有更新相同的行，它们在备库上可以并行执行。显然，这个模式要求 binlog 格式必须是 row。

这时候，我们判断一个事务 T 和 worker 是否冲突，用的就规则就不是“修改同一个表”，而是“修改同一行”。

按行复制和按表复制的数据结构差不多，也是为每个 worker，分配一个 hash 表。只是要实现按行分发，这时候的 key，就必须是“库名 + 表名 + 唯一键的值”。