## Spring 依赖来源

Spring的依赖来源一般来说有4种：

* Spring BeanDefinition
* 单例注册
* 游离对象
* 外部化配置

### Spring BeanDefinition

这是Spring最为主要的依赖来源，这里又可以细分成两类

#### 1.内建BeanDefinition

Spring内部自己提供的内建Bean，其注册点位于`org.springframework.context.annotation.AnnotationConfigUtils#registerAnnotationConfigProcessors(org.springframework.beans.factory.support.BeanDefinitionRegistry, java.lang.Object)`

```java
public static Set<BeanDefinitionHolder> registerAnnotationConfigProcessors(
			BeanDefinitionRegistry registry, @Nullable Object source) {

		DefaultListableBeanFactory beanFactory = unwrapDefaultListableBeanFactory(registry);
		if (beanFactory != null) {
			if (!(beanFactory.getDependencyComparator() instanceof AnnotationAwareOrderComparator)) {
				beanFactory.setDependencyComparator(AnnotationAwareOrderComparator.INSTANCE);
			}
			if (!(beanFactory.getAutowireCandidateResolver() instanceof ContextAnnotationAutowireCandidateResolver)) {
				beanFactory.setAutowireCandidateResolver(new ContextAnnotationAutowireCandidateResolver());
			}
		}

		Set<BeanDefinitionHolder> beanDefs = new LinkedHashSet<>(8);

		if (!registry.containsBeanDefinition(CONFIGURATION_ANNOTATION_PROCESSOR_BEAN_NAME)) {
			RootBeanDefinition def = new RootBeanDefinition(ConfigurationClassPostProcessor.class);
			def.setSource(source);
			beanDefs.add(registerPostProcessor(registry, def, CONFIGURATION_ANNOTATION_PROCESSOR_BEAN_NAME));
		}

		if (!registry.containsBeanDefinition(AUTOWIRED_ANNOTATION_PROCESSOR_BEAN_NAME)) {
			RootBeanDefinition def = new RootBeanDefinition(AutowiredAnnotationBeanPostProcessor.class);
			def.setSource(source);
			beanDefs.add(registerPostProcessor(registry, def, AUTOWIRED_ANNOTATION_PROCESSOR_BEAN_NAME));
		}

		//more...

		return beanDefs;
	}
```

#### 2.自定义BeanDefinition

这是Spring面向开发用户提供的Bean注册方法，直接注册方式为：

`org.springframework.beans.factory.support.BeanDefinitionRegistry#registerBeanDefinition`

包括其衍生方式：

* 基于XML配置文件
* 基于注解驱动
* 基于@Bean的方法注册

同时需要注意的是，在Spring Context正确启动后，其已注册的BeanDefinition会进入冻结状态

`org.springframework.beans.factory.config.ConfigurableListableBeanFactory#freezeConfiguration`



### 单例注册来源

Spring提供了一种注册单例Bean的简单方式:

`org.springframework.beans.factory.config.SingletonBeanRegistry#registerSingleton`

这种方式可以快速注册一个单例Bean到Spring Context中，即便Spring Context已经完成正确启动。但是需要注意的是，注册的单例Bean不会为其**注入依赖**



### 游离对象（非Spring管理的依赖）

这类依赖非常特殊，会注册在Spring Context中，但是不会被Spring管理，即只能作为依赖属性注入到正常的Spring Bean之中，不会被外部所依赖查找。注册方式为：

`org.springframework.beans.factory.config.ConfigurableListableBeanFactory#registerResolvableDependency`

值得注意的是，Spring内部也会采用这种方式，内置4种类型Bean

```java
// BeanFactory interface not registered as resolvable type in a plain factory.
// MessageSource registered (and found for autowiring) as a bean.
beanFactory.registerResolvableDependency(BeanFactory.class, beanFactory);
beanFactory.registerResolvableDependency(ResourceLoader.class, this);
beanFactory.registerResolvableDependency(ApplicationEventPublisher.class, this);
beanFactory.registerResolvableDependency(ApplicationContext.class, this);
```

### 外部化配置

Spring通过`@Value`注解，打通了外部化配置和Bean内部属性的关联，注意外部化配置是存在优先级顺序的，即同名的配置会被更高优先级的配置覆盖