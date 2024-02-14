# AQS

AQS是一个基本框架，用于实现各类同步操作，例如同步锁，信号量等。

AQS基于一个int变量作为锁定字段(state)，包含一个先进先出的等待队列，用户可以通过改变state的值来控制当前同步状态
>Provides a framework for implementing blocking locks and related synchronizers (semaphores, events, etc) that rely on first-in-first-out (FIFO) wait queues.
>
>This class is designed to be a useful basis for most kinds of synchronizers that rely on a single atomic int value to represent state.

## CLH队列
CLH队列是一个先进先出的双向队列，用于保存等待线程队列
### 节点
队列中的每个等待线程由`java.util.concurrent.locks.AbstractQueuedSynchronizer.Node`描述，类结构如下。

同时，每个节点支持两种模式，即共享(SHARED)和独占(EXCLUSIVE)
* 独占表示当前state只能被一个线程持有
* 共享表示当前state可以被多个线程持有
```Java
        /** Marker to indicate a node is waiting in shared mode */
        /**
         * 标记当前节点是一个处于共享模式下的节点
         */
        static final Node SHARED = new Node();
        /** Marker to indicate a node is waiting in exclusive mode */
        static final Node EXCLUSIVE = null;
        /**
         * 当前节点状态
         */
        volatile int waitStatus;

        /**
         * 前继节点
         */
        volatile Node prev;

        /**
         * 后继节点
         */
        volatile Node next;

        /**
         * 节点绑定的线程
         */
        volatile Thread thread;
        
        /**
         * 等待队列下一个节点，如果等于SHARED，则表示当前节点处于共享模式中
         */
        Node nextWaiter;
```

### 节点状态描述

| 状态 | 描述                                                |
|----|---------------------------------------------------|
| -1 | SIGNAL, 当前线程被park住了，需要等待unpark操作                  |
| 1  | CANCELLED，表示当前线程取消了等待，比如在等待过程中发生了interrupt操作或者超时了 |
| -2 | CONDITION，表示当前队列暂时挂起进入了`Conditional`队列            |
| -3 | PROPAGATE，共享节点状态，如果当前节点的下一个节点也是共享的，此状态将传递下去 |

## 初始化
CLH采用懒加载的机制，在第一个等待队列加入时，初始化队列头节点
```Java
    private final void initializeSyncQueue() {
        Node h;
        if (HEAD.compareAndSet(this, null, (h = new Node())))
            tail = h;
    }
```

## 入队
### 添加节点
```Java
    private Node addWaiter(Node mode) {
        //创建一个指定独占/共享模式的节点
        Node node = new Node(mode);

        for (;;) {
            Node oldTail = tail;
            //如果尾节点为null，表示队列需要进行初始化
            if (oldTail != null) {
                //当前节点前继节点就是之前的尾节点
                node.setPrevRelaxed(oldTail);
                //将当前节点作为尾节点（CAS操作）
                if (compareAndSetTail(oldTail, node)) {
                    CAS操作成功后，原尾节点设置成当前节点
                    oldTail.next = node;
                    return node;
                }
            } else {
                //初始化操作
                initializeSyncQueue();
            }
        }
    }
```
### 独占模式下入队
```Java
    public final void acquire(int arg) {
        if (!tryAcquire(arg) && //try方法需要用户自行实现，判断当前状态是否支持当前线程持有
            acquireQueued(addWaiter(Node.EXCLUSIVE), arg)) //入队
            selfInterrupt();  //如果发生了打断，则需要传递下去
    }
    
    /**
     * 此时node已经加入过等待队列
     */
    final boolean acquireQueued(final Node node, int arg) {
        //记录当前线程是否被打断过
        boolean interrupted = false;
        try {
            for (;;) {
                //获取当前节点的前继节点
                final Node p = node.predecessor();
                if (p == head && tryAcquire(arg)) {//如果前置节点是head，说明已经到了第一个，那么需要尝试获取状态
                    //将当前节点作为头节点，当前节点的thread，prev清空
                    setHead(node);
                    //前置节点取消引用
                    p.next = null; // help GC
                    //返回等待过程中是否被中断过
                    return interrupted;
                }
                
                //前节点不是头节点或者获取状态失败，此时需要判断是否需要park
                //需要park的条件 前节点状态==SIGNAL(前节点被已park了)
                //如果检查出前节点状态为CANCELED，那么需要往前遍历，找到第一个状态不为CANCELED的节点，将其作为node的前继节点
                //如果前节点状态为其他，则需要进一步判断，这里就不需要进行park操作了，继续循环到下一步
                //这里主要作用
                if (shouldParkAfterFailedAcquire(p, node))
                    //写入是否被中断过
                    interrupted |= parkAndCheckInterrupt();
            }
        } catch (Throwable t) {
            //当前节点等待过程中发生了任何异常，取消等待，待后续有节点加入时被移除
            cancelAcquire(node);
            if (interrupted)
                //保留打断状态
                selfInterrupt();
            //异常传播
            throw t;
        }
    }
```

### 共享模式下入队
```Java
    public final void acquireShared(int arg) {
        if (tryAcquireShared(arg) < 0)//尝试共享获取状态，用户自行判断
            doAcquireShared(arg); //共享模式下入队操作
    }
    
    private void doAcquireShared(int arg) {
        // 添加第一个共享模式的节点
        final Node node = addWaiter(Node.SHARED);
        // 判断线程是否被打断过
        boolean interrupted = false;
        try {
            for (;;) {
                //找到前继节点
                final Node p = node.predecessor();
                if (p == head) {//前节点是头节点，此时再次尝试获取状态
                    int r = tryAcquireShared(arg);
                    if (r >= 0) {//表示允许获取
                        //将当前节点设置成头节点，并将waitingState状态传播下去
                        setHeadAndPropagate(node, r);
                        原先头节点取消关联
                        p.next = null; // help GC
                        return;
                    }
                }
                //同独占模式一样，判断是否需要进行park，避免空循环
                if (shouldParkAfterFailedAcquire(p, node))
                    interrupted |= parkAndCheckInterrupt();
            }
        } catch (Throwable t) {
            //出现异常，取消获取
            cancelAcquire(node);
            throw t;
        } finally {
            if (interrupted)
                //传递打断状态
                selfInterrupt();
        }
    }
```
### 共享模式下状态传播
```Java
    private void setHeadAndPropagate(Node node, int propagate) {
        Node h = head; // Record old head for check below
        //当前节点设置成head
        setHead(node);
        if (propagate > 0 || h == null || h.waitStatus < 0 ||
            (h = head) == null || h.waitStatus < 0) {
            Node s = node.next;
            //检查下一个节点是否是共享模式，如果是则唤醒下一个节点，因为共享模式下允许多个节点同时持有状态
            if (s == null || s.isShared())
                doReleaseShared();
        }
    }
    
    private void doReleaseShared() {
        for (;;) {
            Node h = head;
            if (h != null && h != tail) {
                int ws = h.waitStatus;
                if (ws == Node.SIGNAL) {
                    //如果头节点被park住了，尝试unpark操作，因为头节点如果是刚放上去的，此时状态还是被park，此时需要解除状态，执行到这里说明head的设置已经完成了
                    if (!h.compareAndSetWaitStatus(Node.SIGNAL, 0))
                        continue;            // loop to recheck cases
                    unparkSuccessor(h);
                }
                //头节点unpark后，状态设置成PROPAGATE
                else if (ws == 0 &&
                         !h.compareAndSetWaitStatus(0, Node.PROPAGATE))
                    continue;                // loop on failed CAS
            }
            //在这个过程中head此时有变化了，重新执行流程
            if (h == head)                   // loop if head changed
                break;
        }
    }
    
    private void unparkSuccessor(Node node) {
        //重置状态，如果是>0说明可能被取消了
        int ws = node.waitStatus;
        if (ws < 0)
            node.compareAndSetWaitStatus(ws, 0);

        //遍历当前节点的下一个节点，找到第一个非取消状态的节点
        Node s = node.next;
        if (s == null || s.waitStatus > 0) {
            s = null;
            //从尾往前遍历，这是因为取消操作会改变每个节点的next值，但是不会改变节点的prev
            for (Node p = tail; p != node && p != null; p = p.prev)
                if (p.waitStatus <= 0)
                    s = p;
        }
        //唤醒第一个非取消状态的节点线程，参与竞争
        if (s != null)
            LockSupport.unpark(s.thread);
    }
```


## 取消
取消操作一般发生在请求状态超时或者等待竞争中出现异常，代码实现
```Java
    private void cancelAcquire(Node node) {
        // Ignore if node doesn't exist
        if (node == null)
            return;

        //既然取消，先取消节点上的线程绑定
        node.thread = null;

        //从所有前节点中，找到第一个不为取消状态的节点
        Node pred = node.prev;
        while (pred.waitStatus > 0)
            node.prev = pred = pred.prev;

        // predNext is the apparent node to unsplice. CASes below will
        // fail if not, in which case, we lost race vs another cancel
        // or signal, so no further action is necessary, although with
        // a possibility that a cancelled node may transiently remain
        // reachable.
        //往前第一个不为取消状态的节点开始
        //predNext即为连续取消节点的第一个
        Node predNext = pred.next;

        //当前节点状态变更
        node.waitStatus = Node.CANCELLED;

        // If we are the tail, remove ourselves.
        if (node == tail && compareAndSetTail(node, pred)) {//将pred设置成尾节点
            //本身就是尾节点了，取消next
            pred.compareAndSetNext(predNext, null);
            
        } else {
            //如果当前节点不是尾节点
            int ws;
            if (pred != head &&
                ((ws = pred.waitStatus) == Node.SIGNAL ||
                 (ws <= 0 && pred.compareAndSetWaitStatus(ws, Node.SIGNAL))) &&
                pred.thread != null) {
                //pred不是head，把pred的next指向当前节点的next
                Node next = node.next;
                if (next != null && next.waitStatus <= 0)
                    pred.compareAndSetNext(predNext, next);
            } else {
                //pred是head，说明当前节点的前面所有节点都是取消状态，直接唤醒下一个节点
                unparkSuccessor(node);
            }
            //当前节点的下一个引用取消
            node.next = node; // help GC
        }
        
        //注意此时只取消了next的关联，prev的关联会在后续shouldParkAfterFailedAcquire方法中取消
    }
    
    private static boolean shouldParkAfterFailedAcquire(Node pred, Node node) {
        int ws = pred.waitStatus;
        if (ws == Node.SIGNAL)
            /*
             * This node has already set status asking a release
             * to signal it, so it can safely park.
             */
            return true;
        if (ws > 0) {
            //这里会改变prev的状态，清理状态为取消的节点
            do {
                node.prev = pred = pred.prev;
            } while (pred.waitStatus > 0);
            pred.next = node;
        } else {
            /*
             * waitStatus must be 0 or PROPAGATE.  Indicate that we
             * need a signal, but don't park yet.  Caller will need to
             * retry to make sure it cannot acquire before parking.
             */
            pred.compareAndSetWaitStatus(ws, Node.SIGNAL);
        }
        return false;
    }
```

