# 虚拟机类加载

## 类的生命周期
一个类从被加载到虚拟机内存开始，到卸载出内存开始，它的整个生命周期将会经历加载（Loading），验证（Verification），准备（Preparation），解析（Resolution），初始化（Initialization），使用（Using）和卸载（Unloading）。
其中，验证，准备和解析三个阶段统称为连接（Linking）。

## 类的加载
在加载阶段，Java虚拟机需要完成以下三件事情：
1. 通过一个类的全限定名称，获取定义此类的二进制字节流
2. 将这个字节流所代表的数据结构，转化为方法区的数据结构
3. 在内存中生成一个代表这个类的Class对象，作为方法区这个类的入口

对于数组类型，情况有所不同，数组类型本身不通过类加载器创建，它是由Java虚拟机直接在内存的中创建出来，数组的元素类型仍然会依靠类加载器加载。

## 类的初始化
在《Java虚拟机规范》中，并没有严格限制在什么时刻对类进行加载，这点可以交由虚拟机实现方自由把控。但是对于初始化阶段，明确规定了有以下情况必须立即对类进行初始化：
1. 遇到new，getstatic，putstatic或invokestatic这四条字节码指令时，如果没有初始化对应的类，则应该立即初始化，简单来说就是第一次使用到这个类时，必须要保证初始化。
2. 使用java.lang.reflect包下的方法对类型进行反射调用时，如果没有初始化，则要进行初始化
3. 当初始化一个类时，如果其父类没有初始化，则需优先初始化其父类
4. 当虚拟机启动时，针对指定的main方法所在的类，必须进行初始化
5. 当使用JDK7加入的动态语言支持时，比如java.lang.invoke.MethodHandle实例时，需要优先初始化其绑定的类
6. 当一个接口实现了default方法（JDK8之后），其对应实现类初始化时，应该优先初始化当前接口

在初始化阶段，Java虚拟机将不再主导应用程序代码，此时初始化过程大多由用户自己来指定。更直接点表达就是执行类构造器`<clinit>()`方法，也就是`static`代码块

## 类加载器
Java虚拟机有意将类加载阶段中“通过一个类的全限定名称，获取定义此类的二进制字节流”这个动作放到Java虚拟机外部去实现，以方便应用程序自己决定如何去获取所需的类。这就是类加载器（ClassLoader）

对于任意一个类，都必须由加载它的类型加载器和这个类本身一起共同确定其在Java虚拟机中的唯一性。跟通俗一点就是：**比较两个类是否相等，前提条件是这两个类由同一个类加载器加载。**

### 双亲委派模型
一般来说，Java程序都会使用到以下3个系统提供的类加载器：
* Bootstrap Class Loader，这个类加载器负责加载<JAVA_HOME>/lib目录下，或者被参数`-Xbootclasspath`所指定的路径中存放的，能被Java虚拟机所识别的类库到内存中。这个类加载器有Java虚拟机创建，应用程序无法获取。
* Extension Class Loader，这个类加载器负责加载<JAVA_HOME>/lib/ext目录下，或者系统变量`java.ext.dirs`所指定的路径的所以类库，允许程序开发者将一些通用性的类库，放置在ext目录下用于拓展JDK的功能。
* Application Class Loader，这个类加载器负责加载用户类路径(ClassPath)上的所有类库。

除此以外，用户还可以加入自定义的类加载器来进行拓展，一般用来实现类隔离相关的功能。

![classloader](classloader.jpeg)

各个类加载器之间的层级关系如图所示，这也就是“双亲委派模型”。双亲委派模型要求除了顶层的启动类加载器外，其余的所有类加载器都有其关联的夫加载器。不过这里的关系不是用继承来实现的，而是通过手动设置达到组合的关系。

双亲委派模型的工作过程是：**如果一个类加载器收到类类加载的请求，它首先不会自己去尝试加载这个类，而是把请求委派给父类加载器去完成，每一个层次的类加载器都是如此。
因此所有的类加载请求都会传送到最顶层的Bootstrap ClassLoader，只有当夫加载器无法完成这个加载请求（无法完整一般是指在其搜索范围内没有找到对应的类）时候，子加载器才会自己尝试去完成加载。**

使用双亲委派模型来组织类加载器之间的关系，有个明显的好处就是Java中的类伴随着它类加载器也拥有了一种层级的关系，同时可以保证一些关键类在整个Java运行环境中之存在一份。
```Java
    protected Class<?> loadClass(String name, boolean resolve)
        throws ClassNotFoundException
    {
        synchronized (getClassLoadingLock(name)) {
            // First, check if the class has already been loaded
            Class<?> c = findLoadedClass(name);
            if (c == null) {
                long t0 = System.nanoTime();
                try {
                    if (parent != null) {
                        c = parent.loadClass(name, false);
                    } else {
                        c = findBootstrapClassOrNull(name);
                    }
                } catch (ClassNotFoundException e) {
                    // ClassNotFoundException thrown if class not found
                    // from the non-null parent class loader
                }

                if (c == null) {
                    // If still not found, then invoke findClass in order
                    // to find the class.
                    long t1 = System.nanoTime();
                    c = findClass(name);

                    // this is the defining class loader; record the stats
                    sun.misc.PerfCounter.getParentDelegationTime().addTime(t1 - t0);
                    sun.misc.PerfCounter.getFindClassTime().addElapsedTimeFrom(t1);
                    sun.misc.PerfCounter.getFindClasses().increment();
                }
            }
            if (resolve) {
                resolveClass(c);
            }
            return c;
        }
    }
```