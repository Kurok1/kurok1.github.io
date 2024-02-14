# CyclicBarrier

跟[CountDownLatch]一样，`CyclicBarrier`也是一个线程同步工具，这里用个简单例子描述二者区别：

假如，有3个员工加1名领导吃饭，领导需要在所有人吃完饭后，统一打车回公司

用`CountDownLatch`的做法
* 每有一个吃完饭就走到门口，等所有人走到门口，领导打车回家

用`CyclicBarrier`的做法
* 大家在饭桌上互相等待对方吃完，所有人吃完后，一起到门口打车回家

## Usage
```Java
class Solver {
  final int N;
  final float[][] data;
  final CyclicBarrier barrier;
  class Worker implements Runnable {
    int myRow;
    Worker(int row) { myRow = row; }
    public void run() {
      while (!done()) {
        processRow(myRow);
        try {
          barrier.await();
        } catch (InterruptedException ex) {
          return;
        } catch (BrokenBarrierException ex) {
          return;
        }
      }
    }
  }
  public Solver(float[][] matrix) {
    data = matrix;
    N = matrix.length;
    Runnable barrierAction = () -> mergeRows(...);
    barrier = new CyclicBarrier(N, barrierAction);
    List<Thread> threads = new ArrayList<>(N);
    for (int i = 0; i < N; i++) {
      Thread thread = new Thread(new Worker(i));
      threads.add(thread);
      thread.start();
    }
    // wait until done
    for (Thread thread : threads)
      thread.join();
  }
 }
```

## 初始化
```Java
    public CyclicBarrier(int parties, Runnable barrierAction) {
        if (parties <= 0) throw new IllegalArgumentException();
        //计数
        this.parties = parties;
        //等待计数
        this.count = parties;
        //计数=0后，执行的操作
        this.barrierCommand = barrierAction;
    }
```

## await(等待)
```Java
private int dowait(boolean timed, long nanos)
        throws InterruptedException, BrokenBarrierException,
               TimeoutException {
        final ReentrantLock lock = this.lock;
        lock.lock();
        try {
            //当前迭代
            final Generation g = generation;
            //当前迭代已完成，异常
            if (g.broken)
                throw new BrokenBarrierException();
            //当前线程被打断，返回状态
            if (Thread.interrupted()) {
                breakBarrier();
                throw new InterruptedException();
            }
            //计数-1
            int index = --count;
            if (index == 0) {  // tripped
                计数=0，表示所有线程都进入了等待状态，执行操作
                boolean ranAction = false;
                try {
                    final Runnable command = barrierCommand;
                    if (command != null)
                        command.run();
                    ranAction = true;
                    //执行完成，进入下一次迭代
                    nextGeneration();
                    return 0;
                } finally {
                    if (!ranAction)
                        //未能执行完成，关闭当前迭代，同时唤醒所有等待线程
                        breakBarrier();
                }
            }

            // loop until tripped, broken, interrupted, or timed out
            for (;;) {
                try {
                    //阻塞，等待唤醒
                    if (!timed)
                        trip.await();
                    else if (nanos > 0L)
                        nanos = trip.awaitNanos(nanos);
                } catch (InterruptedException ie) {
                    //判断迭代是否关闭，未关闭则关闭，并且保留当前线程打断状态
                    if (g == generation && ! g.broken) {
                        breakBarrier();
                        throw ie;
                    } else {
                        // We're about to finish waiting even if we had not
                        // been interrupted, so this interrupt is deemed to
                        // "belong" to subsequent execution.
                        Thread.currentThread().interrupt();
                    }
                }
                //唤醒后发现迭代关闭，说明有其他线程提前被唤醒了，失败
                if (g.broken)
                    throw new BrokenBarrierException();

                //迭代变化了，返回失败
                if (g != generation)
                    return index;
                
                //等待超时。
                if (timed && nanos <= 0L) {
                    breakBarrier();
                    throw new TimeoutException();
                }
            }
        } finally {
            lock.unlock();
        }
    }
    
    private void breakBarrier() {
        //当前迭代标记损坏
        generation.broken = true;
        count = parties;
        //唤醒其他线程，停止等待状态
        trip.signalAll();
    }
```
可以看到实现了一个很简单的计数器扣减功能，当前线程扣减成功中，进入阻塞状态，等待其他线程扣减，等待过程中，有任何一线程发生打断，超时等，自身也结束阻塞状态

## 重置
```Java
    public void reset() {
        final ReentrantLock lock = this.lock;
        lock.lock();
        try {
            breakBarrier();   // break the current generation
            nextGeneration(); // start a new generation
        } finally {
            lock.unlock();
        }
    }
```
可以看到基于迭代模式下，`CyclicBarrier`实现了可重用