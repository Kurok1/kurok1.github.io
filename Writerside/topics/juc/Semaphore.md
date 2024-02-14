# Semaphore

>In computer science, a semaphore is a variable or abstract data type used to control access to a common resource by multiple threads and avoid critical section problems in a concurrent system such as a multitasking operating system. 
>Semaphores are a type of synchronization primitive. A trivial semaphore is a plain variable that is changed (for example, incremented or decremented, or toggled) depending on programmer-defined conditions.
> 
>[wiki](https://en.wikipedia.org/wiki/Semaphore_(programming))

信号量的概念是由荷兰计算机科学家艾兹赫尔·戴克斯特拉（Edsger W. Dijkstra）发明的，广泛的应用于不同的操作系统中。

在系统中，给予每一个进程一个信号量，代表每个进程目前的状态，未得到控制权的进程会在特定地方被强迫停下来，等待可以继续进行的信号到来。

在Java中，Semaphore在实现层面上是一个共享锁，在等待队列的处理上也支持公平和非公平模式。但与普通的共享锁(ReentrantReadWriteLock.ReadLock)不同的是,Semaphore的释放操作等价于加锁，加锁等于等于释放
## 初始化
```Java
    abstract static class Sync extends AbstractQueuedSynchronizer {
        private static final long serialVersionUID = 1192457210091910933L;

        Sync(int permits) {
            setState(permits);
        }
    }
```
`Semaphore`在实现上也借鉴了AQS，在初始化阶段设置好线程持有量。

## 尝试获取(tryAcquireShared)
acquire操作会对资源进行扣减处理，当资源剩余量等于0时，后续线程无法继续获取，进入等待队列
### 非公平模式
```Java
        final int nonfairTryAcquireShared(int acquires) {
            for (;;) {
                //可用量
                int available = getState();
                //剩余量
                int remaining = available - acquires;
                //只有剩余量>=0时，才允许获取资源，否则失败
                //CAS操作不断自旋
                if (remaining < 0 ||
                    compareAndSetState(available, remaining))
                    return remaining;
            }
        }
```
### 公平模式
在原有非公平模式上，要求判断没有其他更早的等待线程
```Java
        protected int tryAcquireShared(int acquires) {
            for (;;) {
                //判断没有等待线程
                if (hasQueuedPredecessors())
                    return -1;
                int available = getState();
                int remaining = available - acquires;
                if (remaining < 0 ||
                    compareAndSetState(available, remaining))
                    return remaining;
            }
        }
```

## 尝试释放(tryReleaseShared)
```Java
        protected final boolean tryReleaseShared(int releases) {
            for (;;) {
                int current = getState();
                //资源反馈
                int next = current + releases;
                //防止int越界
                if (next < current) // overflow
                    throw new Error("Maximum permit count exceeded");
                //CAS自旋改变状态
                if (compareAndSetState(current, next))
                    return true;
            }
        }
```
