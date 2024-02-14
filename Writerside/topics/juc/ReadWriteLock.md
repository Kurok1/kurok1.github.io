# ReadWriteLock

前面介绍了[ReentrantLock](ReentrantLock.md),这是一种可重入的独占锁，即同一时间，只能有一个线程独占资源。但是某些场景下，这种独占模式并不是最佳选择。

比如我们需要读取某个变量，在没有其他线程写入的情况下，我们可以认为这个变量是一个不可变的，当一个不可变的变量，采用独占锁的模式去读取它，这样就显得有些浪费了，因此我们需要使用共享模式，即允许多个线程同时持有资源。这就是读写锁(ReentrantReadWriteLock)。

读写锁中，读锁(ReadLock)采用共享模式，写锁(WriteLock)采用独占模式。二者是冲突关系，当写锁被某个线程持有时，读锁无法加锁，同样的，读锁被持有时，写锁也无法加锁。

| 尝试加锁 | 当前锁定模式 | 是否可以成功加锁 |
|------|--------|----------|
| 读锁   | 读锁     | 是        |
| 读锁   | 写锁     | 否        |
| 写锁   | 读锁     | 否        |
| 写锁   | 写锁     | 否        |


## 读写状态描述
由于读写锁绑定的资源状态是同一个，因此需要某种机制能同时表达读状态和写状态。

在`ReentrantReadWriteLock`,采用了分段的方式计算读写状态，即将int类型的变量，前16位作为写状态的持有，后16位作为读状态的持有
```Java

        /*
         * Read vs write count extraction constants and functions.
         * Lock state is logically divided into two unsigned shorts:
         * The lower one representing the exclusive (writer) lock hold count,
         * and the upper the shared (reader) hold count.
         */

        //分片界限，即31-16位表示写锁，15-0表示读锁
        static final int SHARED_SHIFT   = 16;
        static final int SHARED_UNIT    = (1 << SHARED_SHIFT);
        //最大持有次数
        static final int MAX_COUNT      = (1 << SHARED_SHIFT) - 1;
        static final int EXCLUSIVE_MASK = (1 << SHARED_SHIFT) - 1;

        /** Returns the number of shared holds represented in count. */
        static int sharedCount(int c)    { return c >>> SHARED_SHIFT; }
        /** Returns the number of exclusive holds represented in count. */
        static int exclusiveCount(int c) { return c & EXCLUSIVE_MASK; }
```
这里的设计比较巧妙，但是一定程度上牺牲了可重入次数


## 尝试加锁
[AQS](AQS.md)已经完成实现了独占和共享模式的等待入队，因此这里只需要实现独占和共享模式下的尝试加锁逻辑即可

### 独占(写锁)
```Java
        protected final boolean tryAcquire(int acquires) {
            /*
             * Walkthrough:
             * 1. If read count nonzero or write count nonzero
             *    and owner is a different thread, fail.
             * 2. If count would saturate, fail. (This can only
             *    happen if count is already nonzero.)
             * 3. Otherwise, this thread is eligible for lock if
             *    it is either a reentrant acquire or
             *    queue policy allows it. If so, update state
             *    and set owner.
             */
            Thread current = Thread.currentThread();
            int c = getState();
            //判断是否有写锁存在
            int w = exclusiveCount(c);
            if (c != 0) {//有线程持有锁，可以是持有读锁，也可以是持有写锁
                //w!=0表示，有线程持有读锁
                //当有线程持有写锁时，要求当前线程跟持有线程一致
                if (w == 0 || current != getExclusiveOwnerThread())
                    return false;
                //写锁可持有次数不能超过最大值
                if (w + exclusiveCount(acquires) > MAX_COUNT)
                    throw new Error("Maximum lock count exceeded");
                //重入
                setState(c + acquires);
                return true;
            }
            //c==0,表示没有任何线程持有锁，这里需要判断，可能会导致并发线程同时竞争写锁
            //writerShouldBlock，表示判断是否需要阻塞，需要区分公平模式和非公平模式，
            //非公平模式下，不需要阻塞，任何线程都可以尝试竞争
            //公平模式下，要求AQS中，没有之前参与等待的队列
            if (writerShouldBlock() ||
                //比较状态，如果非公平模式下CAS操作失败，表示加锁失败，后续加入到等待队列中
                !compareAndSetState(c, c + acquires))
                return false;
            //获取锁成功，标记线程持有
            setExclusiveOwnerThread(current);
            return true;
        }
```

### 共享(读锁)
```Java
        protected final int tryAcquireShared(int unused) {
            /*
             * Walkthrough:
             * 1. If write lock held by another thread, fail.
             * 2. Otherwise, this thread is eligible for
             *    lock wrt state, so ask if it should block
             *    because of queue policy. If not, try
             *    to grant by CASing state and updating count.
             *    Note that step does not check for reentrant
             *    acquires, which is postponed to full version
             *    to avoid having to check hold count in
             *    the more typical non-reentrant case.
             * 3. If step 2 fails either because thread
             *    apparently not eligible or CAS fails or count
             *    saturated, chain to version with full retry loop.
             */
            Thread current = Thread.currentThread();
            int c = getState();
            //1.存在写锁并且写锁的持有线程不是自身，失败
            if (exclusiveCount(c) != 0 &&
                getExclusiveOwnerThread() != current)
                return -1;
            //读锁持有数量
            int r = sharedCount(c);
            //同样需要判断是否需要阻塞，类似读锁的逻辑，公平模式下要求AQS没有等待的线程，非公平模式下允许竞争
            //读锁持有次数没有超过最大限制，并且没有并发读锁争抢
            if (!readerShouldBlock() &&
                r < MAX_COUNT &&
                compareAndSetState(c, c + SHARED_UNIT)) {
                
                //加锁成功，利用ThreadLocal，对每个线程进行持有计数统计
                if (r == 0) {
                    firstReader = current;
                    firstReaderHoldCount = 1;
                } else if (firstReader == current) {
                    firstReaderHoldCount++;
                } else {
                    HoldCounter rh = cachedHoldCounter;
                    if (rh == null ||
                        rh.tid != LockSupport.getThreadId(current))
                        cachedHoldCounter = rh = readHolds.get();
                    else if (rh.count == 0)
                        readHolds.set(rh);
                    rh.count++;
                }
                return 1;
            }
            //针对CAS失败操作的自旋，因为读锁竞争失败了，还是允许他继续竞争的
            return fullTryAcquireShared(current);
        }
        
        static final class HoldCounter {
            int count;          // initially 0
            // Use id, not reference, to avoid garbage retention
            final long tid = LockSupport.getThreadId(Thread.currentThread());
        }

        /**
         * ThreadLocal subclass. Easiest to explicitly define for sake
         * of deserialization mechanics.
         */
        static final class ThreadLocalHoldCounter
            extends ThreadLocal<HoldCounter> {
            public HoldCounter initialValue() {
                return new HoldCounter();
            }
        }

        /**
         * The number of reentrant read locks held by current thread.
         * Initialized only in constructor and readObject.
         * Removed whenever a thread's read hold count drops to 0.
         */
        private transient ThreadLocalHoldCounter readHolds;
```

## 释放锁
### 独占模式(写锁释放)
```Java
        protected final boolean tryRelease(int releases) {
            if (!isHeldExclusively())//是否是当前线程持有锁
                throw new IllegalMonitorStateException();
            int nextc = getState() - releases;//释放后的写锁状态
            boolean free = exclusiveCount(nextc) == 0;//是否需要完全释放
            if (free)
                //释放锁，清除独占表示
                setExclusiveOwnerThread(null);
            //写入状态
            setState(nextc);
            return free;
        }
        
        protected final boolean isHeldExclusively() {
            // While we must in general read state before owner,
            // we don't need to do so to check if current thread is owner
            return getExclusiveOwnerThread() == Thread.currentThread();
        }
```
### 共享模式(读锁释放)
```Java
        protected final boolean tryReleaseShared(int unused) {
            //释放前，先根据ThreadLocal找到对应读线程，持有计数-1
            Thread current = Thread.currentThread();
            if (firstReader == current) {
                // assert firstReaderHoldCount > 0;
                if (firstReaderHoldCount == 1)
                    firstReader = null;
                else
                    firstReaderHoldCount--;
            } else {
                HoldCounter rh = cachedHoldCounter;
                if (rh == null ||
                    rh.tid != LockSupport.getThreadId(current))
                    rh = readHolds.get();
                int count = rh.count;
                if (count <= 1) {
                    readHolds.remove();
                    if (count <= 0)
                        throw unmatchedUnlockException();
                }
                --rh.count;
            }
            //自旋CAS操作，更新状态
            for (;;) {
                int c = getState();
                int nextc = c - SHARED_UNIT;
                if (compareAndSetState(c, nextc))
                    // Releasing the read lock has no effect on readers,
                    // but it may allow waiting writers to proceed if
                    // both read and write locks are now free.
                    return nextc == 0;
            }
        }
```