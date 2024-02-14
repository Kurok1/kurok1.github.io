# ReentrantLock

顾名思义，可重入锁。当某一线程持有锁后，允许该线程重复获取（实际有次数限制，这个限制为int类型的最大值，即2^32-1）,其余线程只能待该线程完成所有的锁释放才能参与竞争锁。

可重入锁的功能与Java中`synchronized`关键字功能类似，可以在一定程度上认为`ReentrantLock`是`synchroinized`关键字在JavaAPI层面上的实现。

`ReentrantLock`支持两种模式，即公平锁和非公平锁，其差异主要体现在当前线程释放锁后，等待线程在竞争策略上的差异
* 公平锁，等待线程严格按照先进先出的方式竞争，即越早等待的线程最先获取锁
* 非公平锁，允许各个等待线程同时争抢锁，不准守先进先出的规则

某种程度上来看，公平锁的方式下，每个线程平均等待时间最少，但其吞吐量不如非公平锁，同时非公平锁还可能会导致线程长时间阻塞

在实现层面上，`ReentrantLock`抽象实现了[AQS](juc.md),且所有的等待节点都是独占模式。在加锁和释放锁的逻辑上,已经由AQS代为实现,因此只需关注**尝试加锁**和**尝试释放**的逻辑即可。


## 尝试释放锁
在释放锁的处理上，两种模式下是一样的
```Java
        protected final boolean tryRelease(int releases) {
            //获取当前状态，当前状态就是重入锁的次数，大于1表示被持有了
            int c = getState() - releases;
            //判断尝试释放的线程是否是当前持有锁的线程，不是则说明锁的状态异常
            if (Thread.currentThread() != getExclusiveOwnerThread())
                throw new IllegalMonitorStateException();
            boolean free = false;//标记是否完全释放了，可重入锁，只有当释放次数和加锁次数完全一致是，才会唤醒下一个等待节点
            if (c == 0) {
                free = true;
                //释放锁，清除持有线程
                setExclusiveOwnerThread(null);
            }
            setState(c);
            //是否完全释放掉了锁
            return free;
        }
```
## 非公平模式下的尝试加锁
```Java
        final boolean nonfairTryAcquire(int acquires) {
            final Thread current = Thread.currentThread();
            int c = getState();
            //获取当前状态，=0才表示没有任何线程持有锁，此时当前线程可以获取
            if (c == 0) {//第一次进入
                //cas操作改变state，原子操作
                if (compareAndSetState(0, acquires)) {
                    //设置当前持有线程
                    setExclusiveOwnerThread(current);
                    //获取锁成功
                    return true;
                }
            }
            else if (current == getExclusiveOwnerThread()) {//同一线程重复进入
                //状态累加
                int nextc = c + acquires;
                //这里因为Integer.MAX_VALUE + 1 = Integer.MIN_VALUE,防止越界处理
                if (nextc < 0) // overflow
                    throw new Error("Maximum lock count exceeded");
                //写入状态
                setState(nextc);
                return true;
            }
            //已经有线程持有了，返回失败
            return false;
        }
```

## 公平模式下的尝试加锁
```Java
        protected final boolean tryAcquire(int acquires) {
            final Thread current = Thread.currentThread();
            int c = getState();
            if (c == 0) {//已经没有线程持有锁了，且是第一次进入
                //这里与非公平模式下的区别在于增加了hasQueuedPredecessors判断，也就是要求当前等待线程对应的节点前方，没有等待的节点
                if (!hasQueuedPredecessors() &&
                    //CAS操作写入状态
                    compareAndSetState(0, acquires)) {
                    //设置当前独占线程
                    setExclusiveOwnerThread(current);
                    return true;
                }
            }
            else if (current == getExclusiveOwnerThread()) {
                //同一线程重复持有，处理逻辑同上
                int nextc = c + acquires;
                if (nextc < 0)
                    throw new Error("Maximum lock count exceeded");
                setState(nextc);
                return true;
            }
            return false;
        }
        
        public final boolean hasQueuedPredecessors() {
            Node h, s;
            if ((h = head) != null) {//说明已经初始化完成
                //找到从head开始的第一个未取消的节点
                if ((s = h.next) == null || s.waitStatus > 0) {
                    s = null; 
                    //从头往前，找到最后一个未取消状态的节点
                    //在前面提及过，AQS中的取消操作只会改变节点的next指向，所以这里只能从尾往前遍历
                    for (Node p = tail; p != h && p != null; p = p.prev) {
                        if (p.waitStatus <= 0)
                            s = p;
                    }
                }
                //判断第一个非取消的节点是否跟当前调用线程一致，不一致，说明在当前线程之前，还有其他线程处于等待队列中
                if (s != null && s.thread != Thread.currentThread())
                    return true;
            }
            //未初始化队列，更加说明前方没有等待节点
            return false;
        }

```