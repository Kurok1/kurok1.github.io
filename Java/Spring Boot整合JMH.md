# 项目结构
```plain
project-root
├── pom.xml
├── src
│   ├── main
│   │   └── java
│   │       └── com.example.app
│   │           ├── Application.java
│   │           ├── controller
│   │           ├── service
│   │           ├── repository
│   │           └── config
│   │
│   ├── test
│   │   └── java
│   │       └── com.example.app
│   │           └── XxxTest.java   （功能 / 单测）
│   │
│   └── jmh
│       └── java
│           └── com.example.benchmark
│               ├── OrderServiceBenchmark.java
│               ├── UserServiceBenchmark.java
│               └── BenchmarkApplication.java
│
└── target

```
关键好处
✅ **不会被误当成单测执行**  
✅ **不会进生产包**  
✅ **可以单独配置 JVM 参数**  
✅ **CI 中可以只在特定条件运行**  
✅ **不会被 IDE 的 Test Runner 干扰**

# Maven支持JMH

## 禁用Spring-Boot插件
```xml
<!-- 禁用 Spring Boot repackage -->
<plugin>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-maven-plugin</artifactId>
    <executions>
        <execution>
            <id>repackage</id>
            <phase>none</phase>
        </execution>
    </executions>
</plugin>
```

## 使用`build-helper`插件
```xml
<!-- 让 Maven 识别 src/jmh/java -->
<plugin>
    <groupId>org.codehaus.mojo</groupId>
    <artifactId>build-helper-maven-plugin</artifactId>
    <version>3.5.0</version>
    <executions>
        <execution>
            <id>add-jmh-source</id>
            <phase>generate-test-sources</phase>
            <goals>
                <goal>add-source</goal>
            </goals>
            <configuration>
                <sources>
                    <source>src/jmh/java</source>
                </sources>
            </configuration>
        </execution>
        <!-- 2️⃣ 添加 JMH 资源 -->
        <execution>
            <id>add-jmh-resources</id>
            <phase>generate-test-resources</phase>
            <goals>
                <goal>add-resource</goal>
            </goals>
            <configuration>
                <resources>
                    <resource>
                        <directory>src/jmh/resources</directory>
                        <filtering>false</filtering>
                    </resource>
                </resources>
            </configuration>
        </execution>
    </executions>
</plugin>

```
注意：

- 用 `add-source`（不是 `add-test-source`）
    
- JMH 本质是 **独立的 main 程序**
## Maven Shade Plugin（JMH 主入口）
```xml
<plugin>
	<groupId>org.apache.maven.plugins</groupId>
	<artifactId>maven-shade-plugin</artifactId>
	<version>3.5.0</version>
	<executions>
		<execution>
			<id>shade-my-jar</id>
			<phase>package</phase>
			<goals>
				<goal>shade</goal>
			</goals>
			<configuration>
				<finalName>benchmarks</finalName>
				<transformers>
					<transformer
	implementation="org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
						<mainClass>org.openjdk.jmh.Main</mainClass>
	
					</transformer>
					<transformer
	implementation="org.apache.maven.plugins.shade.resource.AppendingTransformer">
	
						<resource>META-INF/spring.handlers</resource>
					</transformer>
					<transformer
	implementation="org.apache.maven.plugins.shade.resource.AppendingTransformer">
						<resource>META-INF/spring.schemas</resource>
					</transformer>
					<transformer
	implementation="org.apache.maven.plugins.shade.resource.AppendingTransformer">
						<resource>META-INF/spring.factories</resource>
					</transformer>
					<transformer
	implementation="org.apache.maven.plugins.shade.resource.AppendingTransformer">
						<resource>application-benchmark.yaml</resource>
					</transformer>
				</transformers>
				<filters>
					<filter>
						<artifact>*:*</artifact>
						<excludes>
							<exclude>META-INF/*.SF</exclude>
							<exclude>META-INF/*.DSA</exclude>
							<exclude>META-INF/*.RSA</exclude>
						</excludes>
					</filter>
				</filters>
			</configuration>
		</execution>
	</executions>

</plugin>
```
## BenchmarkApplication 放哪里？
推荐放在 `src/jmh/java`
```text
src/jmh/java
└── com.example.benchmark
    ├── BenchmarkApplication.java
    └── XxxServiceBenchmark.java

```
#为什么？

- 不污染生产 Application
    
- Benchmark 可以单独启 profile
    
- 可屏蔽 Web / 定时任务 / MQ
    

```java
@SpringBootApplication(
    scanBasePackages = "com.example.app"
)
public class BenchmarkApplication {
}
```

专用Profile
`application-benchmark.yml`
```yaml
spring:
  main:
    web-application-type: none

logging:
  level:
    root: warn

```
启动命令：
```shell
java -jar target/benchmark.jar \
  -Dspring.profiles.active=benchmark
```

# 目标指定
## 一、整体先看一句话版

```java
@BenchmarkMode(Mode.AverageTime)      // 测什么指标
@OutputTimeUnit(TimeUnit.MICROSECONDS) // 指标单位
@State(Scope.Benchmark)               // 测试状态的生命周期
@Warmup(iterations = 3)               // JVM 预热
@Measurement(iterations = 5)          // 正式测量
@Fork(1)                              // 启动多少个 JVM
```

👉 含义是：

> **在 1 个 JVM 进程中，先预热 3 轮，再测 5 轮，统计每次调用的平均耗时（微秒级），并在整个 benchmark 过程中共享同一份状态。**

---

## 二、逐个注解详细解释（重点）

---

## 1️⃣ `@BenchmarkMode(Mode.AverageTime)`

### 它决定：**你“关心什么指标”**

### 常用 Mode

|Mode|含义|典型用途|
|---|---|---|
|`AverageTime`|**平均每次调用耗时**|最常用|
|`Throughput`|每秒执行次数（ops/s）|QPS|
|`SampleTime`|随机采样耗时|看尾延迟|
|`SingleShotTime`|单次执行耗时|冷启动|
|`All`|所有模式|调研用|

### 示例对比

```java
@BenchmarkMode(Mode.AverageTime)
// 输出：每次调用平均耗时
```

```java
@BenchmarkMode(Mode.Throughput)
// 输出：xx ops/s
```

👉 **Service / 算法性能 → AverageTime**  
👉 **高并发吞吐 → Throughput**

---

## 2️⃣ `@OutputTimeUnit(TimeUnit.MICROSECONDS)`

### 它决定：**结果用什么时间单位展示**

可选：

- `NANOSECONDS`
    
- `MICROSECONDS` ✅（最常用）
    
- `MILLISECONDS`
    
- `SECONDS`
    

⚠️ **不会影响精度，只影响显示**

```text
0.345 us/op
```

而不是：

```text
0.000345 ms/op
```

---

## 3️⃣ `@State(Scope.Benchmark)`（非常重要）

### 它决定：**benchmark 的“上下文对象”怎么共享**

### Scope 对比

|Scope|含义|
|---|---|
|`Benchmark`|整个 benchmark 共享 1 个实例|
|`Thread`|每个线程 1 个实例|
|`Group`|线程组共享|

### 你的这个选择意味着：

```java
@State(Scope.Benchmark)
```

✔️ **整个 benchmark 只创建一个对象**  
✔️ Spring 容器 / Service 只初始化一次  
✔️ 不重复启动 Spring

👉 **Spring Boot benchmark 必须用这个**

---

## 4️⃣ `@Warmup(iterations = 3)`

### 它决定：**正式计时前，跑几轮“预热”**

为什么要预热？

JVM 有：

- JIT 编译
    
- 逃逸分析
    
- 方法内联
    
- 分支预测
    

如果不预热：

❌ 测的是“冷 JVM”  
❌ 数值极不稳定

### 含义

```java
@Warmup(iterations = 3)
```

= **预热 3 轮（默认每轮 1 秒）**

---

## 5️⃣ `@Measurement(iterations = 5)`

### 它决定：**正式统计跑几轮**

```java
@Measurement(iterations = 5)
```

= **测 5 轮**

最终结果是这 5 轮的统计值（平均 / 方差 / 误差）

---

## 6️⃣ `@Fork(1)`（容易被忽略）

### 它决定：**起几个独立的 JVM**

```java
@Fork(1)
```

= 启 1 个 JVM

### 为什么 Fork 很重要？

如果 `@Fork(0)`：

- benchmark 和 Maven / IDE 在同一个 JVM
    
- 类加载、GC、线程池被污染
    

### 常见配置

|Fork|用途|
|---|---|
|`1`|本地 / CI（推荐）|
|`2~3`|高精度对比|
|`0`|❌ 不要用|

---

## 三、这些注解“合在一起”的真实执行流程

你这组配置等价于：

```text
启动 1 个 JVM
↓
创建 1 个 Benchmark State
↓
Warmup 3 次（不计结果）
↓
Measurement 5 次（统计）
↓
输出平均耗时（微秒）
```

---

## 四、Spring Boot Benchmark 的“标准模板”

你现在用的这组注解：

```java
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
@Fork(1)
```

👉 **这是 Spring Boot 场景下的黄金配置**  
👉 90% 的服务性能问题，用它就够了

---

## 五、常见错误 & 后果（对照看）

|错误|后果|
|---|---|
|没有 Warmup|结果不稳定|
|Scope.Thread|Spring 重复启动|
|Fork=0|数据污染|
|用毫秒|看不到差异|
|测 Controller|无法复现真实负载|

# 测试结果查看&分析
```text
Result "com.sinochem.yunlian.dataservice.tms.route.jmh.BenchmarkService.run":
  102953175.150 ±(99.9%) 14862028.079 us/op [Average]
  (min, avg, max) = (97993608.500, 102953175.150, 107295204.709), stdev = 3859621.496
  CI (99.9%): [88091147.072, 117815203.229] (assumes normal distribution)


# Run complete. Total time: 00:13:59

REMEMBER: The numbers below are just data. To gain reusable insights, you need to follow up on
why the numbers are the way they are. Use profilers (see -prof, -lprof), design factorial
experiments, perform baseline and negative tests that provide experimental control, make sure
the benchmarking environment is safe on JVM/OS/HW level, ask for reviews from the domain experts.
Do not assume the numbers tell you what you want them to tell.

NOTE: Current JVM experimentally supports Compiler Blackholes, and they are in use. Please exercise
extra caution when trusting the results, look into the generated code to check the benchmark still
works, and factor in a small probability of new VM bugs. Additionally, while comparisons between
different JVMs are already problematic, the performance difference caused by different Blackhole
modes can be very significant. Please make sure you use the consistent Blackhole mode for comparisons.

Benchmark             Mode  Cnt          Score          Error  Units
BenchmarkService.run  avgt    5  102953175.150 ± 14862028.079  us/op
```

## 一、Benchmark 基本信息类指标

### 1. Benchmark

```
BenchmarkService.run
```

- 被测试的方法（基准用例）
    
- 全限定名：  
    `com.sinochem.yunlian.dataservice.tms.route.jmh.BenchmarkService.run`
    

---

### 2. Mode

```
avgt
```

- **测试模式**
    
- `avgt` = **Average Time**
    
- 表示：**每次操作的平均耗时**
    

常见 Mode 还有：

- `thrpt`：吞吐量（ops/s）
    
- `ss`：单次时间
    
- `sample`：采样时间
    
- `avgt`：平均时间（你这里用的是这个）
    

---

### 3. Units

```
us/op
```

- 单位：**微秒 / 每次操作**
    
- `us` = microseconds
    
- `op` = operation（一次 run 方法调用）
    

---

### 4. Cnt

```
5
```

- **测量次数（Measurement Iterations）**
    
- 表示实际参与统计的测量轮次为 5 次  
    （不包含 warmup）
    

---

## 二、核心性能指标（最重要）

### 5. Score（平均值）

```
102953175.150 us/op
```

- **平均执行时间**
    
- 含义：  
    👉 每次执行 `BenchmarkService.run` 方法，平均耗时约 **102,953,175 微秒**
    
- 换算：
    
    - ≈ **102.95 秒**
        
    - ≈ **1.7 分钟 / 次**
        

---

### 6. Error（误差 / 半置信区间）

```
± 14862028.079 us/op
```

- **统计误差范围**
    
- 通常表示 **Score 的置信区间的一半**
    
- 实际区间是：
    
    ```
    Score ± Error
    ```
    

---

## 三、统计分布指标（更详细）

### 7. min / avg / max

```
(min, avg, max) =
(97993608.500, 102953175.150, 107295204.709)
```

含义：

- **min**：最小耗时
    
    - 97,993,608 us
        
- **avg**：平均耗时
    
    - 102,953,175 us
        
- **max**：最大耗时
    
    - 107,295,204 us
        

👉 说明不同轮次之间有一定波动，但总体集中在 100s 左右。

---

### 8. stdev（标准差）

```
stdev = 3859621.496
```

- **标准差**
    
- 反映数据离散程度
    
- 数值越小，说明测试结果越稳定
    

这里：

- stdev ≈ **3.86 秒**
    
- 相对平均值（≈103 秒）来说，波动 **不算太大**
    

---

### 9. CI（Confidence Interval，置信区间）

```
CI (99.9%): [88091147.072, 117815203.229]
```

- **99.9% 置信区间**
    
- 含义是：
    
    > 有 99.9% 的概率，真实的平均值落在这个区间内
    

即：

- 最低：≈ **88.1 秒**
    
- 最高：≈ **117.8 秒**
    

⚠️ 前提：假设数据服从正态分布

---

## 四、汇总表中的指标（表格部分）

```
Benchmark             Mode  Cnt    Score          Error   Units
BenchmarkService.run  avgt    5  102953175.150 ± 14862028.079  us/op
```

这是前面所有信息的 **摘要版**，包含：

- Benchmark：基准方法
    
- Mode：测试模式
    
- Cnt：测量次数
    
- Score：平均值
    
- Error：误差
    
- Units：单位
    

---

## 五、指标一览速查表

|指标|含义|
|---|---|
|Benchmark|被测试的方法|
|Mode|测试模式（avgt = 平均时间）|
|Cnt|测量次数|
|Score|平均执行时间|
|Error|误差（置信区间的一半）|
|Units|单位（us/op）|
|min|最小耗时|
|avg|平均耗时|
|max|最大耗时|
|stdev|标准差|
|CI|置信区间|
