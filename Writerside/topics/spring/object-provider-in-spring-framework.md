# Spring Framework中的`ObjectProvider`

IoC是一个非常重要的设计原则，其核心思想是：组件无需关心其所依赖组件的生命周期，从而降低代码的耦合度

Ioc的实现方式一般来说有两种：

* 依赖注入（Dependency Injection）, 即利用外部容器完成组件依赖的构建，自动注入到目标对象中。
* 依赖查找（Dependency Lookup）, 即外部容器提供查找接口，由使用者手动获取依赖注入到目标对象中。

### `Spring Framework`中的依赖查找实现

前面提及到，依赖查找是利用Ioc容器所提供的查找接口，完成依赖的注入，在Spring Framework中，其Ioc容器的主要实现为

`org.springframework.beans.factory.BeanFactory`,其所提供的相关查询接口如下：

```java
org.springframework.beans.factory.BeanFactory#getBean(java.lang.String);
org.springframework.beans.factory.BeanFactory#getBean(java.lang.String, java.lang.Class<T>);
org.springframework.beans.factory.BeanFactory#getBean(java.lang.String, java.lang.Object...);
org.springframework.beans.factory.BeanFactory#getBean(java.lang.Class<T>);
org.springframework.beans.factory.BeanFactory#getBean(java.lang.Class<T>, java.lang.Object...);
```

`BeanFactory`接口主要还是提供单一类型的bean查找方法，特别的，针对集合类型，`Spring Framework`提供了`org.springframework.beans.factory.ListableBeanFactory`这一接口，用于支持集合类型的查找，相关接口如下：

```java
//只提供beanName
org.springframework.beans.factory.ListableBeanFactory#getBeanDefinitionNames();
org.springframework.beans.factory.ListableBeanFactory#getBeanNamesForType(org.springframework.core.ResolvableType);
org.springframework.beans.factory.ListableBeanFactory#getBeanNamesForType(org.springframework.core.ResolvableType, boolean, boolean);
org.springframework.beans.factory.ListableBeanFactory#getBeanNamesForType(java.lang.Class<?>);
org.springframework.beans.factory.ListableBeanFactory#getBeanNamesForType(java.lang.Class<?>, boolean, boolean);
org.springframework.beans.factory.ListableBeanFactory#getBeanNamesForAnnotation(java.lang.Class<? extends Annotation>);

//提供bean实例
org.springframework.beans.factory.ListableBeanFactory#getBeansOfType(java.lang.Class<T>)；
org.springframework.beans.factory.ListableBeanFactory#getBeansOfType(java.lang.Class<T>, boolean, boolean)；
org.springframework.beans.factory.ListableBeanFactory#getBeansWithAnnotation(java.lang.Class<? extends Annotation>);
```



### `ObjectProvider`简介

`ObjectProvider`是从Spring 4.3版本开提供的接口，在Spring 5.1中开始出现于`BeanFactory`接口之中，作为类型安全的依赖查找接口的返回值

提供给使用者，我们先看其接口构造

```java
public interface ObjectProvider<T> extends ObjectFactory<T>, Iterable<T> {

	T getObject(Object... args) throws BeansException;

	@Nullable
	T getIfAvailable() throws BeansException;

	default T getIfAvailable(Supplier<T> defaultSupplier) throws BeansException {
		T dependency = getIfAvailable();
		return (dependency != null ? dependency : defaultSupplier.get());
	}

	default void ifAvailable(Consumer<T> dependencyConsumer) throws BeansException {
		T dependency = getIfAvailable();
		if (dependency != null) {
			dependencyConsumer.accept(dependency);
		}
	}

	@Nullable
	T getIfUnique() throws BeansException;

	default T getIfUnique(Supplier<T> defaultSupplier) throws BeansException {
		T dependency = getIfUnique();
		return (dependency != null ? dependency : defaultSupplier.get());
	}

	default void ifUnique(Consumer<T> dependencyConsumer) throws BeansException {
		T dependency = getIfUnique();
		if (dependency != null) {
			dependencyConsumer.accept(dependency);
		}
	}

	@Override
	default Iterator<T> iterator() {
		return stream().iterator();
	}

	default Stream<T> stream() {
		throw new UnsupportedOperationException("Multi element access not supported");
	}

	default Stream<T> orderedStream() {
		throw new UnsupportedOperationException("Ordered element access not supported");
	}

}
```

其夫类接口为`ObjectFactory`和`Iterable`,分别提供了无参版本的`getObject`方法和对集合类型的支持。

在Spring 5.0版本中，额外对其提供了函数式接口的支持。

### `ObjectProvider`的使用场景

前面提及到`ObjectProvier`主要是**作为类型安全的依赖查找接口的返回结果**来使用，下面简单描述下其使用场景

#### 1.屏蔽`BeansException`

`BeanException`是Spring Framework提供的一类异常集合，主要用于描述查询bean时所遇到的异常情况，其相关实现有：

* `NoSuchBeanDefinitionException`, 没有找到相关Bean的定义
* `NoUniqueBeanDefinitionException`, 存在多种Bean的定义
* `BeanInstantiationException`, Bean实例化时出现异常

等等。通常我们在使用依赖查询接口时，往往需要注意这些异常，而ObjectProvider可以替我们自动屏蔽了这些异常，比如`ObjectProvider#getIfAvailable`。

```
	public static void main(String[] args) {
		//no define User bean
		BeanFactory beanFactory = SpringApplication.run(Application.class, args);
		//使用BeanFactory#getBean直接获取
		display("BeanFactory#getBean", () -> {beanFactory.getBean(User.class);});
		// ObjectProvider also ObjectFactory
		display("ObjectFactory#getObject", () -> {beanFactory.getBeanProvider(User.class).getObject();});
		//ObjectProvider#getIfAvailable
		display("ObjectProvider#getIfAvailable", () -> {beanFactory.getBeanProvider(User.class).getIfAvailable();});
	}


	private static void display(String source, Runnable runnable) {
		try {
			System.err.println(source);
			runnable.run();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
```

输出结果如下

```
BeanFactory#getBean
org.springframework.beans.factory.NoSuchBeanDefinitionException
ObjectFactory#getObject
org.springframework.beans.factory.NoSuchBeanDefinitionException
ObjectProvider#getIfAvailable
```

可以明显看到，`ObjectProvider#getIfAvailable`**并没有抛出相关异常**，可以看成是安全的依赖查询。



#### 2.依赖的延迟查找

前面提及到了`ObjectProvider#getIfAvailable`提供了一种安全的依赖查询方式。同时注意到，在`ObjectProvider`中对`getIfAvailable`进行了方法重载,以便做到延迟查找的效果。

```java
	public static void main(String[] args) {
		//no define User bean
		BeanFactory beanFactory = SpringApplication.run(CbtDataApplication.class, args);
		ObjectProvider<User> userProvider = beanFactory.getBeanProvider(User.class);
		User user = userProvider.getIfAvailable(User::new);
		System.out.println(user);
	}
```

我们利用了`ObjectProvider#getIfAvailable(java.util.function.Supplier<T>)`这一重载方法，当User这个Bean未创建时，可以重新创建一个Bean用于返回，这就是延迟查找。

#### 3.Bean的Stream操作

对应集合类型的bean，如果想对其进行Stream操作，仅靠`BeanFactory`是不行的，需要依赖`ListableBeanFactory`来操作。在Spring 5.1之后，`ObjectProvider`增加了`Iterable`的接口继承，使得可以对集合类型的Bean也可以进行Stream操作。

```
	public static void main(String[] args) {
		BeanFactory beanFactory = SpringApplication.run(CbtDataApplication.class, args);
		ObjectProvider<String> userProvider = beanFactory.getBeanProvider(String.class);
		userProvider.stream().forEach(System.out::println);
	}

	@Bean
	@Primary
	public String hello() {
		return "hello";
	}

	@Bean
	public String message() {
		return "message";
	}
```

可以看到，不需要`ListableBeanFactory`，也可以实现对集合类型对象的Stream操作

