# MySQL笔记day1

## MySQL基础架构
![mysql_architecture.png](mysql_architecture.png)

### 连接器
客户端会连接到这个MySQL数据库上，这时候接待你的就是连接器。连接器负责跟客户端建立连接、获取权限、维持和管理连接。

客户端如果太长时间没动静，连接器就会自动将它断开。这个时间是由参数` wait_timeout `控制的，默认值是 8 小时。

如果在连接被断开之后，客户端再次发送请求的话，就会收到一个错误提醒： Lost connection to MySQL server during query。这时候如果你要继续，就需要重连，然后再执行请求了。
数据库里面，长连接是指连接成功后，如果客户端持续有请求，则一直使用同一个连接。短连接则是指每次执行完很少的几次查询就断开连接，下次查询再重新建立一个。

### 分析器
接下来就是真正执行语句了。首先，MySQL 需要知道你要做什么，因此需要对 SQL 语句做解析。分析器先会做“词法分析”。

你输入的是由多个字符串和空格组成的一条 SQL 语句，MySQL 需要识别出里面的字符串分别是什么，代表什么。

MySQL 从你输入的"select"这个关键字识别出来，这是一个查询语句。
它也要把字符串“T”识别成“表名 T”，把字符串“ID”识别成“列 ID”。做完了这些识别以后，就要做“语法分析”。
根据词法分析的结果，语法分析器会根据语法规则，判断你输入的这个 SQL 语句是否满足 MySQL 语法。如果你的语句不对，就会收到“You have an error in your SQL syntax”的错误提醒。

比如下面这个语句 select 少打了开头的字母“s”。
```SQL
mysql> elect * from t where ID=1;

ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near 'elect * from t where ID=1' at line 1
```
一般语法错误会提示第一个出现错误的位置，所以你要关注的是紧接“use near”的内容。

### 优化器
经过了分析器，MySQL 就知道你要做什么了。在开始执行之前，还要先经过优化器的处理。

优化器是在表里面有**多个索引的时候，决定使用哪个索引**；或者在一个语句有多表关联（join）的时候，决定各个表的连接顺序。

比如你执行下面这样的语句，这个语句是执行两个表的 join：
```SQL
mysql> select * from t1 join t2 using(ID) where t1.c=10 and t2.d=20;
```
* 既可以先从表 t1 里面取出 c=10 的记录的 ID 值，再根据 ID 值关联到表 t2，再判断 t2 里面 d 的值是否等于 20。
* 也可以先从表 t2 里面取出 d=20 的记录的 ID 值，再根据 ID 值关联到 t1，再判断 t1 里面 c 的值是否等于 10。

这两种执行方法的逻辑结果是一样的，但是执行的效率会有不同，而优化器的作用就是决定选择使用哪一个方案。优化器阶段完成后，这个语句的执行方案就确定下来了，然后进入执行器阶段。

### 执行器
MySQL 通过分析器知道了你要做什么，通过优化器知道了该怎么做，于是就进入了执行器阶段，开始执行语句。

* 开始执行的时候，要先判断一下你对这个表 T 有没有执行查询的权限，如果没有，就会返回没有权限的错误，如下所示
```SQL
mysql> select * from T where ID=10;ERROR 1142 (42000): SELECT command denied to user 'b'@'localhost' for table 'T'
```
* 如果有权限，就打开表继续执行。打开表的时候，执行器就会根据表的引擎定义，去使用这个引擎提供的接口。比如我们这个例子中的表 T 中，ID 字段没有索引，那么执行器的执行流程是这样的(InnoDB)：
  1. 调用 InnoDB 引擎接口取这个表的第一行，判断 ID 值是不是 10，如果不是则跳过，如果是则将这行存在结果集中；
  2. 调用引擎接口取“下一行”，重复相同的判断逻辑，直到取到这个表的最后一行。
  3. 执行器将上述遍历过程中所有满足条件的行组成的记录集作为结果集返回给客户端。
 
至此，这个语句就执行完成了。

## Redo Log
在 MySQL 里，如果每一次的更新操作都需要写进磁盘，然后磁盘也要找到对应的那条记录，然后再更新，整个过程 IO 成本、查找成本都很高。

为了解决这个问题，MySQL 的设计者就用了 WAL 技术来解决这个问题，WAL 的全称是 Write-Ahead Logging，它的关键点就是先写日志，再写磁盘。

具体来说，当有一条记录需要更新的时候，InnoDB 引擎就会先把记录写到 redo log里面，并更新内存，这个时候更新就算完成了。

同时，InnoDB 引擎会在适当的时候，将这个操作记录更新到磁盘里面，而这个更新往往是在系统比较空闲的时候做。

InnoDB 的 redo log 是固定大小的，比如可以配置为一组 4 个文件，每个文件的大小是 1GB，那么总共就可以记录 4GB 的操作。从头开始写，写到末尾就又回到开头循环写，如下面这个图所示。
![mysql_redo_log.png](mysql_redo_log.png)

write pos 是当前记录的位置，一边写一边后移，写到第 3 号文件末尾后就回到 0 号文件开头。checkpoint 是当前要擦除的位置，也是往后推移并且循环的，擦除记录前要把记录更新到数据文件。

## Bin Log
MySQL 整体来看，其实就有两块：
* 一块是 Server 层，它主要做的是 MySQL 功能层面的事情；
* 还有一块是引擎层，负责存储相关的具体事宜。上面我们聊到的 redo log 是 InnoDB 引擎特有的日志，而 Server 层也有自己的日志，称为 binlog（归档日志）。

这两种日志有以下三点不同。
1. redo log 是 InnoDB 引擎特有的；binlog 是 MySQL 的 Server 层实现的，所有引擎都可以使用。redo log 是物理日志，记录的是“在某个数据页上做了什么修改”；
2. binlog 是逻辑日志，记录的是这个语句的原始逻辑，比如“给 ID=2 这一行的 c 字段加 1 ”。
3. redo log 是循环写的，空间固定会用完；binlog 是可以追加写入的。“追加写”是指 binlog 文件写到一定大小后会切换到下一个，并不会覆盖以前的日志。

## UPDATE的两阶段提交
接下来来看看一个update语句是如何具体执行的，以这个sql为例子
```SQL
update T set c=c+1 where ID=2;
```
1. 执行器先找引擎取 ID=2 这一行。ID 是主键，引擎直接用树搜索找到这一行。如果 ID=2 这一行所在的数据页本来就在内存中，就直接返回给执行器；否则，需要先从磁盘读入内存，然后再返回。
2. 执行器拿到引擎给的行数据，把这个值加上 1，比如原来是 N，现在就是 N+1，得到新的一行数据，再调用引擎接口写入这行新数据。
3. 引擎将这行新数据更新到内存中，同时将这个更新操作记录到 redo log 里面，此时 redo log 处于 prepare 状态。然后告知执行器执行完成了，随时可以提交事务。
4. 执行器生成这个操作的 binlog，并把 binlog 写入磁盘。
5. 执行器调用引擎的提交事务接口，引擎把刚刚写入的 redo log 改成提交（commit）状态，更新完成。
![mysql_commit_update.png](mysql_commit_update.png)

上面的步骤，将redo log的写入拆成了两个步骤：prepare和commit，也就是**两阶段提交**

### 为什么必须要两阶段提交
两阶段提交本质上是为了数据在binlog和redo log上保证一致性，因为binlog和redo log是两种完全不同的逻辑，所以只能采用两阶段提交的方式

如果不采用两阶段提交，那么会出现什么问题呢？暂还是以上面的update sql为例子

* **先写redo log 再写bin log**，假设在redo log写完后，MySQL系统crash，此时该操作未记录到binlog中，后续恢复数据库时，由于binlog缺少这条语句，导致最终T表中c的值没有任何变化。
* **先写binlog 再写 redo log**，假设先写的binlog，写完后MySQL系统crash，由于redo log未写，对客户端来说，该此事务是失败的，后续恢复数据库时，由于binlog已经记录，所以导致T表的c值被更新了

可以看到，如果不使用“两阶段提交”，那么数据库的状态就有可能和用它的日志恢复出来的库的状态不一致。

## 索引
假设表T的建表语句如下：
```SQL
mysql> create table T
(id int primary key, 
k int not null, 
name varchar(16),
index (k)
)engine=InnoDB;
```
### 索引结构
InnoDB引擎下，索引的结构为B+树,大致结构如下图所示
![mysql_index](mysql_index.png)

主键索引的叶子节点存的是整行数据。在 InnoDB 里，主键索引也被称为聚簇索引（clustered index）。

非主键索引的叶子节点内容是主键的值。在 InnoDB 里，非主键索引也被称为二级索引（secondary index）。

* 如果语句是 select * from T where ID=500，即主键查询方式，则只需要搜索 ID 这棵 B+ 树；
* 如果语句是 select * from T where k=5，即普通索引查询方式，则需要先搜索 k 索引树，得到 ID 的值为 500，再到 ID 索引树搜索一次。这个过程称为**回表**。
### 索引维护
B+ 树为了维护索引有序性，在插入新值的时候需要做必要的维护。以上面这个图为例，如果插入新的行 ID 值为 700，则只需要在 R5 的记录后面插入一个新记录。

如果新插入的 ID 值为 400，就相对麻烦了，需要逻辑上挪动后面的数据，空出位置。而更糟的情况是，如果 R5 所在的数据页已经满了，根据 B+ 树的算法，这时候需要**申请一个新的数据页**，然后挪动部分数据过去。这个过程称为**页分裂**。在这种情况下，性能自然会受影响。

当然有分裂就有合并。**当相邻两个页由于删除了数据，利用率很低之后，会将数据页做合并**。合并的过程，可以认为是分裂过程的逆过程。

所以为了尽可能避免**页分裂**和**页合并**， 所以推荐使用自增主键id
### 最左前缀原则
假设我现在有张用户表，建表sql如下：
```SQL
 CREATE TABLE `tuser` (
  `id` int(11) NOT NULL,
  `id_card` varchar(32) DEFAULT NULL,
  `name` varchar(32) DEFAULT NULL,
  `age` int(11) DEFAULT NULL,
  `ismale` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `id_card` (`id_card`),
  KEY `name_age` (`name`,`age`)
) ENGINE=InnoDB
```
如果为每一种查询都设计一个索引，索引是不是太多了。如果我现在要按照某个用户身份证号去查他的家庭地址呢？虽然这个查询需求在业务中出现的概率不高，但总不能让它走全表扫描吧？

我们以（name，age）这个索引来分析、
![mysql_index1](mysql_index1.png)
可以看到，索引项是按照索引定义里面出现的字段顺序排序的。

当你的逻辑需求是查到所有名字是“张三”的人时，可以快速定位到 ID4，然后向后遍历得到所有需要的结果。

如果你要查的是所有名字第一个字是“张”的人，你的 SQL 语句的条件是"where name like ‘张 %’"。这时，你也能够用上这个索引，查找到第一个符合条件的记录是 ID3，然后向后遍历，直到不满足条件为止。

可以看到，不只是索引的全部定义，只要满足最左前缀，就可以利用索引来加速检索。这个最左前缀可以是联合索引的最左 N 个字段，也可以是字符串索引的最左 M 个字符。这个规则就是最左前缀匹配原则

因为最左前缀，所以当已经有了 (a,b) 这个联合索引后，一般就不需要单独在 a 上建立索引了。因此，第一原则是，如果通过调整顺序，可以少维护一个索引，那么这个顺序往往就是需要优先考虑采用的。

那么，如果既有联合查询，又有基于 a、b 各自的查询呢？查询条件里面只有 b 的语句，是无法使用 (a,b) 这个联合索引的，这时候你不得不维护另外一个索引，也就是说你需要同时维护 (a,b)、(b) 这两个索引。