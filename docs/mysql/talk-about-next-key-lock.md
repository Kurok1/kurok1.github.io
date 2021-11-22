## 浅谈Mysql中的`Next-Key Lock`

`Innodb`引擎在实现行锁机制时，提供了三种算法：

* `Record Lock` 记录锁，用于锁定指定的某条数据库记录
* `Gap Lock` 间隙锁，锁定一个范围区间，但是不包括记录本身
* `Next-Key Lock` 前两者的集合，加锁范围为一个前开后闭的区间。值得注意的是，数据库加行锁默认都是加的`Next-Key Lock`,然后根据场景考虑退化

### 准备工作

`MySQL`版本为`5.7.34`

建表语句以及数据语句：

```sql
CREATE TABLE `t` (
  `id` int(11) NOT NULL,
  `c` int(11) DEFAULT NULL,
  `d` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `c` (`c`)
) ENGINE=InnoDB;

insert into t 
values(0,0,0),(2,2,2),(10,10,10),(15,15,15),(20,20,20),(25,25,25);
```



### 场景一: 等值查询间隙锁

| Session A      | Session B | Session C |
| -------------- | --------- | :-------: |
| begin;update t set d = d+1 where id = 4; |||
|                | insert into t values(8,8,8);//**blocked** |           |
|                |           | update t set d = d+1 where id = 10;//*successed* |

可以观察到，Session B在执行过程中被阻塞住了。通过查询`INFORMATION_SCHEMA.INNODB_LOCKS`表可以看到，Session B被`Gap Lock`锁锁定，而Session C则正常执行

#### 分析

1. Session A执行update语句，因此先加上`Next-Key Lock`，由于id=4的记录并不存在，因此锁的范围为(id=2,id=10]
2. 而id=10并不满足id=4的条件，因此最终退化成为`Gap Lock`范围为(id=2,id=10)
3. Session B插入的记录id=8，满足`Gap Lock`的范围条件，因此被阻塞。Session C更新的记录id=10，超出这个范围，因此正常执行

---

### 场景二: 非唯一索引记录锁

| Session A                                              | Session B                                       | Session C                                 |
| ------------------------------------------------------ | ----------------------------------------------- | ----------------------------------------- |
| begin;select id from t where c = 2 lock in share mode; |                                                 |                                           |
|                                                        | update t set d = d+1 where id = 2;//*successed* |                                           |
|                                                        |                                                 | insert into t values(7,7,7);//**blocked** |

Session B正常执行，而Session C则被阻塞

#### 分析：

1. Session A执行时，给表t的索引c加上`Next-Key Lock`，范围为(0,2]
2. 由于c为普通索引，因此还需要继续查找到下一条c!=2的记录(c=10),此时`Next-Key Lock`退化成间隙锁，范围为(2,10)

3. 由于是在索引c上加锁，因此不影响Session B的执行（Session B使用主键索引，也没有涉及到字段c的变更）。Session C满足索引c上的间隙锁条件，被阻塞

#### 注意：

Session A采用的加锁方式为`in share mode`,同时查询字段不涉及其他正常字段，因此不会在主键索引上加锁。

如果需要为主键索引加锁，可以采用以下两种方式：

1. `lock in share mode`更改为`for update`。这样MySQL会认为接下来你可能为进行数据的更新，因此顺道给主键加上索引
2. 查询字段引入索引中不存在的字段，比如例子中，Session A的查询语句可以修改成`select id,d from t where c = 2 lock in share mode`



### 场景三：主键索引范围锁

| Session A                                                 | Session B                                    | Session C                                          |
| --------------------------------------------------------- | -------------------------------------------- | -------------------------------------------------- |
| begin;select * from t where id >=10 and id<11 for update; |                                              |                                                    |
|                                                           | insert into t values(8,8,8);//*successed*    |                                                    |
|                                                           | insert into t values(13,13,13);//**blocked** |                                                    |
|                                                           |                                              | update t set d = d + 1 where id = 15;//**blocked** |

#### 分析

1. Session A执行时，先找到第一个id=10的行，加上`Next-Key Lock`,范围为(2,10],由于存在等值条件，因此退化成记录锁（只锁定id=10这一行）
2. 后续继续查找范围，直到找到id=15的行，加上`Next-Key Lock`,范围为(10,15]。

因此，Session A在执行过程中，给表t的主键索引加了两把锁：

1. id=10的`Record Lock`	
2. 范围为(10,15]的`Next-Key Lock`

Session B的第二条语句满足条件，因此被阻塞，Session C也是如此。



### 场景四：非唯一索引范围锁

| Session A                                             | Session B                                 | Session C                                   |
| ----------------------------------------------------- | ----------------------------------------- | ------------------------------------------- |
| begin;select * from t where c>=10 and c<11 for update |                                           |                                             |
|                                                       | insert into t values(8,8,8);//**blocked** |                                             |
|                                                       |                                           | update t set d=d+1 where c=15;//**blocked** |

这个场景下跟场景三类似，但是不同的点在于：**c并非主键索引，所以在加上了(2,10]这个`Next-Key Lock`后并不会退化成`Record Key`。所以最终Session A在执行过程中给索引c的(2,10],(10,15]两个范围加上了`Next-Key Lock`**



### 场景五：非唯一索引上的间隙锁

这里，先给表t新插入一条记录

```sql
insert into t values(30,10,30);
```

新插入一条数据后，表t存在有两条c=10但是主键id不同的行。

| Session A                        | Session B                                    | Session C                                   |
| -------------------------------- | -------------------------------------------- | ------------------------------------------- |
| begin;delete from t where c = 10 |                                              |                                             |
|                                  | insert into t values(12,12,12);//**blocked** |                                             |
|                                  |                                              | update t set d=d+1 where c=15;//*successed* |

#### 分析

1. Session A在执行过程中，根据索引c，访问到第一个c=10的记录，此时，给(「c=2,id=2」,「c=10,id=10」)这个范围加上了`Next-Key Lock`
2. Session继续查找，查找到「c=15，id=15」的记录，由于是等值查询（c=10），因此退化成(「c=10,id=10」，「c=15,id=10」)这个范围的间隙锁，最终该索引的范围如下：

| 记录值\是否加锁 |      |      | 🔒    | 🔒     | 🔒     | 🔒    |       |       |       |
| --------------- | ---- | ---- | ---- | ----- | ----- | ---- | ----- | ----- | ----- |
|                 | id=0 | id=2 |      | id=10 | id=30 |      | id=15 | id=20 | id=25 |
|                 | c=0  | c=2  |      | c=10  | c=10  |      | c=15  | c=20  | c=25  |

很明显，Session B命中了范围，故阻塞。Session C正常执行



### 场景六：limit加锁

在场景5的基础上，执行下列过程

| Session A                                 | Session B                                    |
| ----------------------------------------- | -------------------------------------------- |
| begin;delete from t where c = 10 limit 2; |                                              |
|                                           | insert into t values(12,12,12);//*successed* |

表t中c=10的数据有两条，因此从逻辑上看`limit 2`加不加都是一样的，但是加锁的效果却不同。这是因为条件中明确了`limit 2`这个条件，所以遍历到第二条c=10的记录（「c=10,id=30」）时就停止加锁了,所以最终的锁定范围如下所示：

| 记录值\是否加锁 |      |      | 🔒    | 🔒     | 🔒     |       |       |       |
| --------------- | ---- | ---- | ---- | ----- | ----- | ----- | ----- | ----- |
|                 | id=0 | id=2 |      | id=10 | id=30 | id=15 | id=20 | id=25 |
|                 | c=0  | c=2  |      | c=10  | c=10  | c=15  | c=20  | c=25  |

所以Session B正常执行

### 场景七：死锁案例

| Session A                                               | Session B                                   |
| ------------------------------------------------------- | ------------------------------------------- |
| begin;select id from t where c = 10 lock in share mode; |                                             |
|                                                         | update t set d=d+1 where c=10;//**blocked** |
| insert into values(8,8,8);                              |                                             |
|                                                         | Dead Lock Happened                          |

#### 分析

1. Session A执行后，根据前面的场景二，Session A在索引c上的(2,10]加上了`Next-Key Lock`,给(10,15)加上了`Gap Lock`
2. Session B的update语句，在索引c上需要加`Next-Key Lock`，范围为(5,10]，与Session A冲突，进入锁等待状态。
3. Session A再次执行insert语句，此时被Session B锁住，死锁发生。

---

## 总结

1. 加锁的基本单位是 next-key lock。next-key lock 是前开后闭区间。
2. 查找过程中访问到的对象才会加锁。
3. 索引上的等值查询，给唯一索引加锁的时候，next-key lock 退化为行锁。
4. 索引上的等值查询，向右遍历时且最后一个值不满足等值条件的时候，next-key lock 退化为间隙锁。