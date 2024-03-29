# 垃圾收集器-串行GC,并行GC

## Serial收集器
这个收集器是一个单线程工作的收集器,它的整个垃圾收集的过程都是在单线程环境下完成的,其新生代采用[标记复制算法](JVM3.md#mark-copy),老年代使用[标记整理算法](JVM3.md#mark-compact).

Serial收集器是单线程工作的,工作效率低下,已经不适合当前服务端的应用环境了.
但在客户端模式下,或者说机器资源不多(这里一般指cpu核心数较少且内存不高)的情况下,由于没有其他线程的交互的额外开销,Serial自然就可以得到最高的收集效率.
因此Serial收集器对于运行在客户端模式下的虚拟机来说是个很好的选择

## ParNew收集器
ParNew收集器实质上是Serial收集器的多线程版本实现.
除了使用多线程进行垃圾回收外,其余的行为也与Serial相似,比如:
* Serial可用的所有控制参数
* 收集算法
* 对象分配规则
* 回收策略

ParNew收集器除了支持多线程并行收集外,其他与Serial收集器相比没有什么创新之处.
注意: 在单核心处理器的环境中,ParNew的表现是不如Serial的,甚至由于存在线程交互的开销,该收集器在通过超线程技术实现的伪双核处理器环境中都不能百分之百超越Serial收集器

-XX:ParallelGCThreads参数控制ParNew参与GC的线程数量

## 并发GC与并行GC的区别
* 并行GC: 并行描述的是多条垃圾收集器线程之间的关系,说明同一时间有多条这样的线程在协同工作,默认此时用户线程处于等待状态(STW)
* 并发: 并发描述的是垃圾收集器线程与用户线程之间的关系,说明同一时间垃圾收集器和用户线程都在运行. 由于垃圾收集器线程占用了一部分系统资源,因此用户线程的吞吐量会收到一定的影响

## Parallel Scavenge收集器
PS(Parallel Scavenge)收集器是一款新生代收集器,它同样是基于[标记复制算法](JVM3.md#mark-copy)实现的,也支持多线程并行收集

PS收集器的特点是它与其他收集器的关注点不同, PS收集器的目标则是达到一个可控制的吞吐量.
所谓吞吐量就算处理器用于运行用户代码与处理器消耗总时间的比值.

如果虚拟机完成某个任务,用户代码+垃圾收集时间耗费了100分钟,垃圾收集时间耗费1分钟,那么吞吐量就是99%. 
停顿时间越短越适合需要保证服务响应质量的程序,良好的响应速度可以提升用户体验.
高吞吐量则可以代表可以高效率的利用处理器资源.

PS收集器提供了两个参数用于精确控制吞吐量:
* -XX:MaxGCPauseMillis. 该参数允许的值是一个大于0的毫秒数, 收集器将尽力保证内存回收花费的时间不超过用户设定值.

但这并不意味着设定一个很小的值就能使得系统的垃圾收集速度变得更快. 垃圾收集速度的变快是以新生代空间为代价换取的: 系统会将新生代调的更小点,收集300MB的新生代速度肯定是比收集500MB的速度更快的,但这会导致垃圾收集的频率更频繁

* -XX:GCTimeRatio, 该参数要求设置为一个正整数,表示用户期望虚拟机消耗在GC上的时间不要超过程序运行时间的1/(1+N).

默认值为99, 含义是保证用户线程运行的时间是GC执行器执行时间的99倍,也可以认为收集器的时间消耗不超过总运行时间的1%

由于与吞吐量密切相关,PS收集器也经常被称为"吞吐量优先收集器". 除了上面提及的两个参数外,PS收集器还有一个参数值得注意:
* -XX:+UseAdaptiveSizePolicy, 这是个开关参数,打开后,虚拟机会根据当前运行环境的影响,自动调整堆中各个区域的相对大小,以提供最合适的停顿时间或者最大吞吐量

## Serial Old收集器
这是Serial收集器的老年代版本,同样的,一般用于客户端模式下的虚拟机垃圾回收

## Parallel Old收集器
Parallel Old收集器是[PS收集器](#parallel-scavenge)的老年代版本,支持多线程并行收集,基于[标记整理算法](JVM3.md#mark-compact)实现