---
layout: default
title: 浅谈MySQL事务隔离模式
description: 浅谈MySQL事务隔离模式
---

## 事务隔离

总所周知，数据库事务具有4大特性，A(`原子性`)C(`一致性`)I(`隔离性`)D(`持久性`)。`ACD`都好理解，目的都是为了保证事务整个进行过程的完整性。而隔离性则是定义了数据库系统中一个事务中操作的结果在何时以何种方式对其他并发事务操作可见。

## 错误读现象举例
`ANSI/ISO SQL 92`标准描述了三种不同的一个事务读取另外一个事务可能修改的数据的"读现象"。当然，这三种现象都是不期望看见的。

假设我们有如下表`user`



| id        | name          | age |
|:----------|:--------------|:----|
| 1         | user1         | 11  |
| 2         | user2         | 12  |
| 4         | user3         | 20  |
| 6         | user4         | 18  |

### 脏读

假设我们有两个事务A,B。在下列时间线下进行操作

| 时间节点 | 事务A操作                                            | 事务B操作                                |
| -------- | ---------------------------------------------------- | ---------------------------------------- |
| T0       | begin                                                |                                          |
| T1       | `select age from user where id = 2;/*读取到age=12*/` | begin                                    |
| T2       |                                                      | `update user set age = 19 where id = 2;` |
| T3       | `select age from user where id = 2;/*读取到age=19*/` |                                          |
| T4       |                                                      | rollback                                 |

在这个例子中，事务B进行回滚后，数据库`user`表中就已经没有`id=2,age=19`的数据了



### 不可重复读

还是同样的例子，时间线操作如下:

假设我们有两个事务A,B。在下列时间线下进行操作

| 时间节点 | 事务A操作                                            | 事务B操作                                |
| -------- | ---------------------------------------------------- | ---------------------------------------- |
| T0       | begin                                                |                                          |
| T1       | `select age from user where id = 2;/*读取到age=12*/` | begin                                    |
| T2       |                                                      | `update user set age = 19 where id = 2;` |
| T3       |                                                      | commit                                   |
| T4       | `select age from user where id = 2;/*读取到age=19*/` |                                          |

在这个例子中，事务B提交成功，因此他对id为1的行的修改就对其他事务可见了。但是事务A在此前已经从这行读到了另外一个`age`的值



### 幻读

`幻读`是`不可重复读`的特殊场景。在事务执行过程中，当两个完全相同的查询语句执行得到不同的结果集。这种现象称为`幻读`

| 时间节点 | 事务A操作                                            | 事务B操作                                          |
| -------- | ---------------------------------------------------- | -------------------------------------------------- |
| T0       | begin                                                |                                                    |
| T1       | `select * from user where id > 2;/*读取到两行记录*/` | begin                                              |
| T2       |                                                      | `INSERT INTO users VALUES ( 3, 'username', 27 ); ` |
| T3       |                                                      | commit                                             |
| T4       | `select * from user where id > 2;/*读取到3行记录*/`  |                                                    |

当事务A两次执行*SELECT ... WHERE*检索一定范围内数据，事务B在这个表中创建了(如`INSERT`)了一行新数据，这条新数据正好满足事务B的“WHERE”子句。从而导致了同一个语句两次结果不一致



## MySQL的隔离级别实现

同其他关系型数据库一样，MySQL同样也实现了4种隔离级别

- Read Uncommitted 读未提交
- Read Commited 读已提交
- Repeatable Read 可重复读
- Serializable 串行化



### Read Uncommited 读未提交

这种隔离级别意味着当前事务能读到其他事务未提交的数据，没有任何限制，理论上性能最好的一种，但是会带来`脏读`的问题，所以这种级别一般不在我们的业务考量之中。



### MVCC在MySQL的InnoDB中的实现

在讨论`Read Commited`和`Repeatable Read`之前，先讨论下MMVC在InnoDB中的实现。MySQL正是以后者为基础实现了`Read Commited`和`Repeatable Read`。

在InnoDB中，会在每行数据后添加两个额外的隐藏的值来实现MVCC，在[MySQL官方文档](https://dev.mysql.com/doc/refman/8.0/en/innodb-multi-versioning.html)中是这样对这两个值描述的：

> - A 6-byte `DB_TRX_ID` field indicates the transaction identifier for the last transaction that inserted or updated the row. Also, a deletion is treated internally as an update where a special bit in the row is set to mark it as deleted.
> - A 7-byte `DB_ROLL_PTR` field called the roll pointer. The roll pointer points to an undo log record written to the rollback segment. If the row was updated, the undo log record contains the information necessary to rebuild the content of the row before it was updated.

`DB_TRX_ID`用于存储最后一次修改该行记录的事务id，在MySQL中，事务id是随时间自增长的，也就是说可以通过比较大小的方式，比较当前事务操作的当前行是不是最新的，或者说在当前事务开启之前，是否有其他事务修改过这一行数据。

`DB_ROLL_PTR`用于指向当前行在`Undo Log`中所处的最新位置.

`Undo Log`记录了数据库中所有未提交事务的反向操作，比如一个事务，`insert`了一条记录，但是没有`commit`,这时候在`Undo Log`中记录了一条对应的`delete`语句,在比如我`update`一条记录，则`Undo Log`则会记录更新前的原值

#### Undo Log 版本链

前面提到了每行记录都使用了`DB_ROLL_PTR`用于指向当前行在`Undo Log`中所处的最新位置。那么怎么根据`DB_ROLL_PTR`在`Undo Log`中寻找到所有的记录（不管是否提交）呢，答案就是版本链。

例如，我们有这样一行数据

| id   | name | DB_TRX_ID | DB_ROLL_PRT |
| ---- | ---- | --------- | ----------- |
| 1    | name | 10        | 地址A       |

现在有个事务，id为20，执行如下sql并提交

```sql
-- trx_id = 20
set autocommit = 0;
update table set name = 'name1' where id = 1;
commit;
```

提交完成后，`Undo Log`变成如下结构

| 地址   | id   | name  | DB_TRX_ID | DB_ROLL_PRT |
| ------ | ---- | ----- | --------- | ----------- |
|        | 1    | name1 | 20        | B           |
| ...... |      |       |           |             |
| B      | 1    | name  | 10        | A           |

这时候，再进来一个事务，id为30，执行下列sql不提交

```sql
-- trx_id = 30
set autocommit = 0;
update table set name = 'name2' where id = 1;
```

执行完成后，`Undo Log`变成如下结构

| 地址   | id   | name  | DB_TRX_ID | DB_ROLL_PRT |
| ------ | ---- | ----- | --------- | ----------- |
|        | 1    | name2 | 30        | C           |
| ...... |      |       |           |             |
| C      | 1    | name1 | 20        | B           |
| ...... |      |       |           |             |
| B      | 1    | name  | 10        | A           |

可以发现，`Undo Log`利用`DB_ROLL_PRT`构建了一个版本链，当然这个版本链不会无限增长，当事务已提交后，**对应的记录放入待清理的链表，在合适的时机删除**



#### Read View

说完了`Undo Log`我们再来看看`Read View`。`Read View`中主要就是有个列表来存储我们系统中当前活跃着的读写事务，也就是begin了还未提交的事务。

`Read View`中主要就是有个列表来存储我们系统中当前活跃着的读写事务，也就是begin了还未提交的事务。通过这个列表来判断记录的某个版本是否对当前事务可见。其中最主要的与可见性相关的属性如下：

`up_limit_id`：当前已经提交的事务号的下一个事务号，事务号 < `up_limit_id` ，对于当前`Read View`都是可见的。理解起来就是创建`Read View`视图的时候，之前已经提交的事务对于该事务肯定是可见的。

`low_limit_id`：当前最大的事务号 + 1，事务号 >= low_limit_id，对于当前`Read View`都是不可见的。理解起来就是在创建`Read View`视图之后创建的事务对于该事务肯定是不可见的。

`trx_ids`：为活跃事务id列表，即`Read View`初始化时当前未提交的事务列表。

而在MySQL中实现`Read Commited`和R`epeatable Read`就是根据不同的`Read View`策略来实现的

### Read Commited 读已提交

读已提交，故名思意，就是当前事务可以读取到其他事务已提交的数据。具体实现很简单：

1. 构建一个新的`Read View`
2. 根据当前事务id，查到当前行对应的`Undo Log`版本链
3. 根据`Read View`查到`up_limit_id`
4. 根据第1步查到的`Undo Log`版本链和`up_limit_id`,找到最新的已提交的`Undo Log`，就是当前行锁能看到的数据

每次`select`的时候，都会重复上述步骤。通过这样的策略，可以规避掉脏数据的问题，但是没法解决不可重复读和幻读的问题

- 针对不可重复读，由于每次`select`的时候都会重新构建`Read View`，所以每次读取的数据一定都是最新版本的数据
- 幻读同上

虽然`Read Commited`无法避免不可重复读和幻读的问题，但在实际应用场景中，我们一般都是能接受不可重复读和幻读的情形，在这种场景下，我们每次操作的都是数据库的最新值，只要不是脏数据，其实也能接受。

### Repeatable Read 可重复读

`Read Commited`无法解决不可重复读的场景，是因为每次`select`的时候都会重新构建一个新的`Read View`。那么我们设想一样，如果对同样一个行，不管查询多少次，所使用的`Read View`都是第一次查询所构建的，是否就能解决不可重复读呢，答案是肯定的。

实际上，`Repeatable Read`这一隔离级别就是这样实现的，第一次查询时候构建一次`Read View`,后面不管对这个行查询多少次，所使用的`Read View`都是第一次构建的，那么自然而然就解决的不可重复读的问题

#### 还会产生幻读吗？

实际上如果我们在MySQL的`Repeatable Read`环境下跑一遍上面幻读的例子，是不会出现幻读的现象的，但是这不意味着我们解决了幻读，考虑如下场景：

| 时间节点 | 事务A操作                                                    | 事务B操作                                          |
| -------- | ------------------------------------------------------------ | -------------------------------------------------- |
| T0       | begin                                                        |                                                    |
| T1       | `select * from user where id > 2;/*读取到两行记录*/`         | begin                                              |
| T2       |                                                              | `INSERT INTO users VALUES ( 3, 'username', 27 ); ` |
| T3       |                                                              | commit                                             |
| T4       | `update user set age = 21 where id > 2;/*这里更新了3条记录*/` |                                                    |
| T5       | `select * from user where id > 2;/*读取到3行记录*/`          |                                                    |

通过上述场景，我们可以看到，在T5时刻发生了幻读场景。

这是因为默认情况下`Repeatable Read`做`select`的时候实际上是一种快照读的模式，当使用`update`,`insert`,`delete`的时候，就会衍化成当前读模式，下次在进行同样的`select`时则会出现幻读。(其实这种情况下的幻读属于"误伤", 因为`update`自身要避免幻读，所以强制使用了`next-key`锁的方式，读取到了其他事务已提交的数据)



### Serializable 串行化

这种模式下，不管有多少事务在跑，都是以一种串行的方式去执行，相互之间不会干扰，属于最高级别的安全，但是性能相当低下，相当于一个单核cpu在跑

------

参考文献：

- [一文理解Mysql MVCC](https://zhuanlan.zhihu.com/p/66791480)
- [事务隔离级别-wiki](https://en.wikipedia.org/wiki/Isolation_(database_systems))

[back](./)