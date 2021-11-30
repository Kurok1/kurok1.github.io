## binlog的三种格式

众所周知，MySQL主从同步的是基于binlog来实现的。Master节点上对数据库的操作，会生成binlog同步发送至各个Slave节点，各个Slave节点基于binlog，解析成本地的relax log，再基于relax log完成数据的同步。

而binlog的格式也有三种，分别是：statement，row和mixed。下面来讨论下这三种格式的差异。



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

```tex
# enable bin log
log_bin = master
# server id
server_id = 1
```
如果你想指定特定格式的binlog，可以在`my.cnf`文件中进行配置

```tex
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