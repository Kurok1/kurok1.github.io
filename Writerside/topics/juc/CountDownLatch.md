# CountDownLatch

`CountDownLatch`是一个线程同步工具。

当主线程下发任务到工作线程时，需要等待所有工作线程同时执行完成，那么就可以使用`CountDownLatch`
```Java
 class Driver { // ...
   void main() throws InterruptedException {
     CountDownLatch startSignal = new CountDownLatch(1);
     CountDownLatch doneSignal = new CountDownLatch(N);
     for (int i = 0; i < N; ++i) // create and start threads
       new Thread(new Worker(startSignal, doneSignal)).start();

     doSomethingElse();            // don't let run yet
     //告诉所有子线程开始工作
     startSignal.countDown();      // let all threads proceed
     doSomethingElse();
     //等待全部子线程完成
     doneSignal.await();           // wait for all to finish
   }
 }

 class Worker implements Runnable {
   private final CountDownLatch startSignal;
   private final CountDownLatch doneSignal;
   Worker(CountDownLatch startSignal, CountDownLatch doneSignal) {
     this.startSignal = startSignal;
     this.doneSignal = doneSignal;
   }
   public void run() {
     try {
       //等待主线程信号开始
       startSignal.await();
       doWork();
       //提示主线程当前任务完成
       doneSignal.countDown();
     } catch (InterruptedException ex) {} // return;
   }

   void doWork() { ... }
 }
```

## 初始化
```Java
    private static final class Sync extends AbstractQueuedSynchronizer {
        private static final long serialVersionUID = 4982264981922014374L;

        Sync(int count) {
            setState(count);
        }

        int getCount() {
            return getState();
        }

        protected int tryAcquireShared(int acquires) {
            //这里申明必须完全释放后才能尝试获取，也就是说无法做到可重用
            return (getState() == 0) ? 1 : -1;
        }

        protected boolean tryReleaseShared(int releases) {
            // Decrement count; signal when transition to zero
            //因为是共享模式，解锁时无需判断是否是独占线程
            for (;;) {
                int c = getState();
                if (c == 0)
                    return false;
                int nextc = c - 1;
                if (compareAndSetState(c, nextc))
                    return nextc == 0;
            }
        }
    }

    private final Sync sync;

    /**
     * Constructs a {@code CountDownLatch} initialized with the given count.
     *
     * @param count the number of times {@link #countDown} must be invoked
     *        before threads can pass through {@link #await}
     * @throws IllegalArgumentException if {@code count} is negative
     */
    public CountDownLatch(int count) {
        if (count < 0) throw new IllegalArgumentException("count < 0");
        this.sync = new Sync(count);
    }
```
可以看到，在初始化节点，就对AQS的资源状态进行的写入，也就是说，`CountDownLatch`本质上是一个**等待解锁的共享锁**

## CountDown(解锁)
```Java
public void countDown() {
        sync.releaseShared(1);
    }
```
可以看到，countDown的过程本质就是释放锁

## Await(加锁)
```Java
public void await() throws InterruptedException {
        sync.acquireSharedInterruptibly(1);
    }
```
await的本质就是加锁，当Sync未完全释放时，处于等待队列中，完全锁放后，当前线程获得锁，得以继续

## 可重用思考
`CountDownLatch`本身并不支持复用，因为`CountDownLatch`本质来说就是一个共享锁(已完成加锁)，await只会在完全释放后才去加锁，此时无论`CountDownLatch`设置的n是多少，都会退化成`CountDownLatch(1)`。

因此如果要实现可重用，可以改动await的代码，例如:
```Java
public void await() throws InterruptedException {
        sync.acquireSharedInterruptibly(n);
    }
```

