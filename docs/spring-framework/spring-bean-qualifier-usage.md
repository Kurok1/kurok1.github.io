## Spring Bean的限定注入和分组



Spring Ioc容器之中，实现了依赖注入，我们可以通过`@Autowired`注解完成特定类型的Bean注入。

但是如果存在多个符合条件的Bean时，为了避免`NoUniqueBeanDefinitionException`异常，除了`@Primary`标记一个主Bean外，我们还可以通过`@Qualifier`指定某个单一Bean进行注入



### 限定注入

利用`@Qualifier`指定特定名称，达到限定注入

```java
@Configuration
public class Application {

    @Autowired
    @Qualifier("user3")
    private User user;

    public void displayUser() {
        System.out.printf(String.format("current injected user : %s\n", user.toString()));
    }

    public static void main(String[] args) {
        AnnotationConfigApplicationContext applicationContext = new AnnotationConfigApplicationContext();
        applicationContext.registerBean("user3", User.class, ()->User.of(3L, "user3"));

        applicationContext.register(Application.class);
        applicationContext.refresh();
        //启动ApplicationContext

        //查看注入后的结果
        Application application = applicationContext.getBean(Application.class);
        application.displayUser();
    }

    /**
     * 指定创建一个beanName=user1的UserBean
     * @return
     */
    @Bean("user1")
    public User user1() {
        return User.of(1L, "user1");
    }

    /**
     * 指定创建一个beanName=user2的UserBean
     * @return
     */
    @Bean("user2")
    public User user2() {
        return User.of(2L, "user2");
    }
}
```

可以看到，在启动时，共完成了3个UserBean的注册，最终，通过`@Qualifier("user3")`，限定user3的Bean完成注入，最后输出结果

```shell
current injected user : User{id=3, name='user3'}
```



### Bean分组

除了上面提及的限定注入外，`@Qualifier`还可以完成对Bean的逻辑分组。

#### 1.拓展`@Qualifier`

`@Qualifier`可以作用于注解上，因此我们派生出一个新的注解，用于标记特定类型的UserBean

```java
@Target({ElementType.FIELD, ElementType.METHOD, ElementType.PARAMETER, ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Inherited
@Documented
@Qualifier
public @interface UserGroup {
}
```

2.UserBean分组

我们定义3个UserBean，其中两个被`@UserGroup`标记

```java
    @Bean("user1")
    public User user1() {
        return User.of(1L, "user1");
    }

    @Bean("user2")
    @UserGroup
    public User user2() {
        return User.of(2L, "user2");
    }

    @Bean("user3")
    @UserGroup
    public User user3() {
        return User.of(3L, "user3");
    }
```

分组注入

```java
@Configuration
public class Application {

    @Autowired
    @UserGroup
    private List<User> users;

    public void displayUsers() {
        System.out.printf(String.format("current injected users : %s\n", users.toString()));
    }

    public static void main(String[] args) {
        AnnotationConfigApplicationContext applicationContext = new AnnotationConfigApplicationContext();
        applicationContext.registerBean("user4", User.class, ()->User.of(4L, "user4"));

        applicationContext.register(Application.class);
        applicationContext.refresh();
        //启动ApplicationContext

        //查看UserGroup注入后的结果
        Application application = applicationContext.getBean(Application.class);
        application.displayUsers();

        //查看所有的UserBean
        System.out.println("all user bean registered:");
        applicationContext.getBeanProvider(User.class).stream().forEach(System.out::println);
    }
}
```

输出效果

```text
current injected users : [User{id=2, name='user2'}, User{id=3, name='user3'}]
all user bean registered:
User{id=4, name='user4'}
User{id=1, name='user1'}
User{id=2, name='user2'}
User{id=3, name='user3'}
```

可以看到，`@UserGroup`将user2和user3分成了一组，在注入时亦可以同时注入