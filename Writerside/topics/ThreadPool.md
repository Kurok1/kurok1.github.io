# ThreadPool

## 为什么需要线程池
在操作系统中，线程的创建和销毁都是开销很大的操作。因此从复用的角度来看，在满足最大并发度的情况下，应该尽可能的将更多的线程复用。同时出于线程生命周期管理的角度，能更好的控制线程的生命周期。因此线程池则必不可少了。

## 内建实现ThreadPoolExecutor
```Java
    public ThreadPoolExecutor(int corePoolSize,
                              int maximumPoolSize,
                              long keepAliveTime,
                              TimeUnit unit,
                              BlockingQueue<Runnable> workQueue,
                              ThreadFactory threadFactory,
                              RejectedExecutionHandler handler) {
        if (corePoolSize < 0 ||
            maximumPoolSize <= 0 ||
            maximumPoolSize < corePoolSize ||
            keepAliveTime < 0)
            throw new IllegalArgumentException();
        if (workQueue == null || threadFactory == null || handler == null)
            throw new NullPointerException();
        this.corePoolSize = corePoolSize;
        this.maximumPoolSize = maximumPoolSize;
        this.workQueue = workQueue;
        this.keepAliveTime = unit.toNanos(keepAliveTime);
        this.threadFactory = threadFactory;
        this.handler = handler;
    }
```
通过构造函数，我们可以看到一个线程池的基本组成
* corePoolSize 线程池核心线程数
* maximumPoolSize 线程池允许的最大线程池
* keepAliveTime 线程允许空闲的最长时间，0表示永久
* unit 空间时间时间单位
* workQueue 任务等待队列
* threadFactory 线程创建工厂，工作线程都是由这个接口创建
* RejectedExecutionHandler 任务拒绝策略，当任务无法继续执行时，执行拒绝策略

## Worker构造
```Java
    private final class Worker
        extends AbstractQueuedSynchronizer
        implements Runnable
    {
        final Thread thread;
        Runnable firstTask;
        volatile long completedTasks;

        Worker(Runnable firstTask) {
            setState(-1); // inhibit interrupts until runWorker
            this.firstTask = firstTask;
            this.thread = getThreadFactory().newThread(this);
        }

        public void run() {
            runWorker(this);
        }

        protected boolean isHeldExclusively() {
            return getState() != 0;
        }

        protected boolean tryAcquire(int unused) {
            if (compareAndSetState(0, 1)) {
                setExclusiveOwnerThread(Thread.currentThread());
                return true;
            }
            return false;
        }

        protected boolean tryRelease(int unused) {
            setExclusiveOwnerThread(null);
            setState(0);
            return true;
        }

        public void lock()        { acquire(1); }
        public boolean tryLock()  { return tryAcquire(1); }
        public void unlock()      { release(1); }
        public boolean isLocked() { return isHeldExclusively(); }

        void interruptIfStarted() {
            Thread t;
            if (getState() >= 0 && (t = thread) != null && !t.isInterrupted()) {
                try {
                    t.interrupt();
                } catch (SecurityException ignore) {
                }
            }
        }
    }
```
这是线程池中，每个工作线程的具体实现，每一个worker会绑定第一个具体的实现线程。
同时Worker实现了AQS的独占模式。

## Worker的生命周期 {id="worker_1"}
### 新增
线程池只会在两种情况下新增Worker
* 当前工作线程数小于核心线程数，每次新增任务时都会增加Worker，直到与核心线程数相等
* 等待队列已满，且当前工作线程数未达到最大线程，此时新增任务都会增加Worker，直到与最大线程数相等

```Java
    private boolean addWorker(Runnable firstTask, boolean core) {//core区分创建的是核心线程还是非核心线程
        retry:
        for (int c = ctl.get();;) {
            //检查线程池状态和队列状态，线程池已经关闭则无需增加
            //等待队列为空，也不需要增加，当前任务塞到等待队列即可
            if (runStateAtLeast(c, SHUTDOWN)
                && (runStateAtLeast(c, STOP)
                    || firstTask != null
                    || workQueue.isEmpty()))
                return false;
            
            for (;;) {
                if (workerCountOf(c)
                    >= ((core ? corePoolSize : maximumPoolSize) & COUNT_MASK))
                    return false;
                //判断当前工作线程数是否达到最大线程
                //CAS增加工作线程数
                if (compareAndIncrementWorkerCount(c))
                    break retry;
                c = ctl.get();  // Re-read ctl
                //判断是否并发关闭
                if (runStateAtLeast(c, SHUTDOWN))
                    continue retry;
                // else CAS failed due to workerCount change; retry inner loop
            }
        }

        boolean workerStarted = false;
        boolean workerAdded = false;
        Worker w = null;
        try {
            //构造Worker，每个Worker的创建都必须拥有其初始任务，否则会开始计算空闲时间
            w = new Worker(firstTask);
            final Thread t = w.thread;
            if (t != null) {
                //此时线程可见
                final ReentrantLock mainLock = this.mainLock;
                //主锁锁定，不允许并发添加Worker
                mainLock.lock();
                try {
                    int c = ctl.get();
                    //判断线程池是否在运行
                    if (isRunning(c) ||
                        (runStateLessThan(c, STOP) && firstTask == null)) {
                        if (t.getState() != Thread.State.NEW)//当前线程不是新创建出来的，说明线程可能被替换过
                            throw new IllegalThreadStateException();
                        workers.add(w);//加入工作线程集合
                        workerAdded = true;
                        int s = workers.size();
                        //这里只是记录指标
                        if (s > largestPoolSize)
                            largestPoolSize = s;
                    }
                } finally {
                    mainLock.unlock();
                }
                if (workerAdded) {
                    //添加成功则立刻启动线程，此时会执行Worker#run方法
                    t.start();
                    workerStarted = true;
                }
            }
        } finally {
            if (! workerStarted)
                //失败回调处理
                addWorkerFailed(w);
        }
        return workerStarted;
    }
```
### 运行
每个Worker在创建后，就会立刻开始执行其绑定的任务，然后此时如果等待队列中还有任务，则从队列中取出任务执行
```Java
    final void runWorker(Worker w) {
        Thread wt = Thread.currentThread();
        Runnable task = w.firstTask;//取出第一个任务
        w.firstTask = null;
        w.unlock(); //worker的状态重置，因为Worker在创建的时候，state=-1，此操作是将状态改回0
        boolean completedAbruptly = true;
        try {
            while (task != null || (task = getTask()) != null) {
                w.lock();//锁定，防止其余线程并发执行runWorker方法
                if ((runStateAtLeast(ctl.get(), STOP) ||
                     (Thread.interrupted() &&
                      runStateAtLeast(ctl.get(), STOP))) &&
                    !wt.isInterrupted())
                    //如果检查到线程池关闭，则通知内部线程打断
                    wt.interrupt();
                try {
                    //执行前回调
                    beforeExecute(wt, task);
                    try {
                        //任务执行
                        task.run();
                        //执行后回调
                        afterExecute(task, null);
                    } catch (Throwable ex) {
                        //执行后异常回调
                        afterExecute(task, ex);
                        throw ex;
                    }
                } finally {
                    task = null;
                    //统计任务指标
                    w.completedTasks++;
                    //解锁
                    w.unlock();
                }
            }
            //没有出现异常
            completedAbruptly = false;
        } finally {
            //worker运行过程中出现异常后，回调,此时Worker可能会销毁
            processWorkerExit(w, completedAbruptly);
        }
    }
```
### 销毁
线程的销毁会出现在以下三种情况
* Worker执行过程中发生异常
* Worker是非核心线程，执行完成后需要销毁
* 核心线程多于核心线程数（核心线程数允许在运行期动态变更），此时执行完任务后也需要销毁

```Java
    private void processWorkerExit(Worker w, boolean completedAbruptly) {
        //之前发生异常导致退出，提前减少Worker数量
        if (completedAbruptly) // If abrupt, then workerCount wasn't adjusted
            decrementWorkerCount();

        final ReentrantLock mainLock = this.mainLock;
        //主锁锁定，防止并发删除和增加
        mainLock.lock();
        try {
            //worker执行任务数统计
            completedTaskCount += w.completedTasks;
            workers.remove(w);//移除
        } finally {
            mainLock.unlock();
        }
        //判断是否关闭了线程池，如果关闭了则要打断所有线程
        tryTerminate();

        int c = ctl.get();
        if (runStateLessThan(c, STOP)) {
            //线程池未关闭
            if (!completedAbruptly) {
                //正常执行完成后
                //allowCoreThreadTimeOut表示是否允许线程存在空闲时期
                //false表示允许，那么线程在空闲后，还是允许继续保留
                //true表示不允许，空闲一段时间后便会移除
                //这里的含义是指，如果允许空闲驻留，那么需要保证worker数量平衡到corePoolSize
                //如果不允许，说明需要他们自身空循环后销毁
                int min = allowCoreThreadTimeOut ? 0 : corePoolSize;
                //走到这里，说明任务基本已经执行完成了，此时需要判断在这期间是否有并发任务加入，如果有，那么最少要保留一个Worker
                if (min == 0 && ! workQueue.isEmpty())
                    min = 1;
                //如果已经超过了最小要求，无需后面加Worker
                if (workerCountOf(c) >= min)
                    return; // replacement not needed
            }
            //再次新增一个Worker去执行
            addWorker(null, false);
        }
    }
```

## 加入任务
```Java
    public void execute(Runnable command) {
        if (command == null)
            throw new NullPointerException();

        int c = ctl.get();
        if (workerCountOf(c) < corePoolSize) {
            //如果Worker数量小于核心线程，先增加线程，把当前任务绑定到新Worker中
            if (addWorker(command, true))
                return;
            c = ctl.get();
        }
        //线程池正在运行，且加入等待队列成功了
        if (isRunning(c) && workQueue.offer(command)) {
            int recheck = ctl.get();
            //并发检查，如果没有运行就移除任务，执行拒绝策略
            if (! isRunning(recheck) && remove(command))
                reject(command);
            else if (workerCountOf(recheck) == 0)//运行中，但是worker数=0，需要新增worker，次数任务还在队列中
                addWorker(null, false);
        }
        else if (!addWorker(command, false))//运行中，但是添加到队列失败，可能是队列满了，需要新增一个非核心线程来执行
            //创建失败，只能拒绝了
            reject(command);
    }
```
## 任务拒绝
Java内建了4种拒绝策略
1. `AbortPolicy` 中断策略，直接抛出异常
2. `DiscardPolicy` 抛弃策略，即任务直接丢掉
3. `DiscardOldestPolicy` 抛弃最老任务，即抛弃掉队列中队首的任务，然后再次尝试将当前任务加入执行
4. `CallerRunsPolicy` 直接执行策略，由任务提交线程来执行任务

## 关闭线程池
