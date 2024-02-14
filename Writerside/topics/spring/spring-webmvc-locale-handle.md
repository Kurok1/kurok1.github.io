# Spring WebMVC 国际化处理

## 读取Locale信息 {id="locale_1"}

读取Locale信息的核心接口

`org.springframework.web.server.i18n.LocaleContextResolver`

该接口用于从Servlet请求中读取Locale信息，核心方法为：`org.springframework.web.server.i18n.LocaleContextResolver#resolveLocaleContext`

通过查看其对应其实现类，不难看出，Spring WebMVC支持的Locale设置方式一共有以下4种：

1. `org.springframework.web.server.i18n.AcceptHeaderLocaleContextResolver`基于header中的`Accept-Language`选项读取Locale信息，多个传值取第一个。

2. `org.springframework.web.servlet.i18n.CookieLocaleResolver`基于Cookie读取Locale设置信息，CookieName默认为`CookieLocaleResolver.class.getName() + ".LOCALE"`

   从cookie中读取完后，写入到ServletRequest的Attribute属性中，从而贯穿整个请求上下文

3. `org.springframework.web.servlet.i18n.SessionLocaleResolver`基于Session设置，实现类似`org.springframework.web.servlet.i18n.CookieLocaleResolver`

4. `org.springframework.web.servlet.i18n.FixedLocaleResolver`全局固定Locale信息，Spring Boot环境中通过配置key`spring.web.locale`设置全局Locale环境



#### Locale信息的存储 {id="locale_2"}

`org.springframework.context.i18n.LocaleContextHolder`

基于ThreadLocal的方式进行存储

```java
private static final ThreadLocal<LocaleContext> localeContextHolder =
			new NamedThreadLocal<>("LocaleContext");

	private static final ThreadLocal<LocaleContext> inheritableLocaleContextHolder =
			new NamedInheritableThreadLocal<>("LocaleContext");
```

同时允许跨线程存储

```java
	public static void setLocale(@Nullable Locale locale, boolean inheritable) {
		LocaleContext localeContext = getLocaleContext();
		TimeZone timeZone = (localeContext instanceof TimeZoneAwareLocaleContext ?
				((TimeZoneAwareLocaleContext) localeContext).getTimeZone() : null);
		if (timeZone != null) {
			localeContext = new SimpleTimeZoneAwareLocaleContext(locale, timeZone);
		}
		else if (locale != null) {
			localeContext = new SimpleLocaleContext(locale);
		}
		else {
			localeContext = null;
		}
		setLocaleContext(localeContext, inheritable);
	}
```



#### 视图渲染过程中Locale信息的存储 {id="locale_3"}

`org.springframework.web.servlet.support.RequestContext`在创建过程中，再次从HttpServletRequest中读取

```
		// Determine locale to use for this RequestContext.
		LocaleResolver localeResolver = RequestContextUtils.getLocaleResolver(request);
		if (localeResolver instanceof LocaleContextResolver) {
			LocaleContext localeContext = ((LocaleContextResolver) localeResolver).resolveLocaleContext(request);
			locale = localeContext.getLocale();
			if (localeContext instanceof TimeZoneAwareLocaleContext) {
				timeZone = ((TimeZoneAwareLocaleContext) localeContext).getTimeZone();
			}
		}
		else if (localeResolver != null) {
			// Try LocaleResolver (we're within a DispatcherServlet request).
			locale = localeResolver.resolveLocale(request);
		}
```



#### 疑惑点：

**为什么RequestContext中解析Locale采用的是`LocaleResolver#resolveLocaleContext()`而在FrameworkServlet中，采用的是`ServletRequest#getLocale()`**

猜测：

RequestContext是和RequestContextUtils是Spring1.0版本即存在的源代码，此时ServletRequest并没有getLocale()方法(Servlet2.3版本引入getLocale()，约2006年左右)。而Spring在后续版本更新中，仍然保留了`LocaleResolver`的实现。



Spring1.0版本中RequestContextUtils对Locale读取的实现

```java
/**
	 * Retrieves the current locale from the given request,
	 * using the LocaleResolver bound to the request by the DispatcherServlet.
	 * @param request current HTTP request
	 * @return the current locale
	 * @throws IllegalStateException if no LocaleResolver has been found
	 */
	public static Locale getLocale(HttpServletRequest request) throws IllegalStateException {
		return getLocaleResolver(request).resolveLocale(request);
	}
```

Spring3.0版本中，已经对`ServletRequest#getLocale()`有所支持

```java
/**
	 * Retrieves the current locale from the given request,
	 * using the LocaleResolver bound to the request by the DispatcherServlet
	 * (if available), falling back to the request's accept-header Locale.
	 * @param request current HTTP request
	 * @return the current locale, either from the LocaleResolver or from
	 * the plain request
	 * @see #getLocaleResolver
	 * @see javax.servlet.http.HttpServletRequest#getLocale()
	 */
	public static Locale getLocale(HttpServletRequest request) {
		LocaleResolver localeResolver = getLocaleResolver(request);
		if (localeResolver != null) {
			return localeResolver.resolveLocale(request);
		}
		else {
			return request.getLocale();
		}
	}
```





## 文本国际化处理(JSTL)

在JSTL中，本地化信息Locale的存储则是在`javax.servlet.jsp.jstl.fmt.LocalizationContext`中。



而`LocalizationContext`则是存储在ServletRequest的attribute中，贯穿于整个servlet的生命周期。



最终，通过`javax.servlet.jsp.jstl.fmt.LocaleSupport#getLocalizedMessage(javax.servlet.jsp.PageContext, java.lang.String, java.lang.Object[], java.lang.String)`完成国际化文案的处理



#### Locale注入时机

org.springframework.web.servlet.DispatcherServlet#render **开始渲染**

-> org.springframework.web.servlet.view.AbstractView#renderMergedOutputModel **Model属性渲染**

--> org.springframework.web.servlet.view.AbstractView#createRequestContext **创建RequestContext，传递Locale信息**

---> org.springframework.web.servlet.view.JstlView#exposeHelpers

----> org.springframework.web.servlet.support.JstlUtils#exposeLocalizationContext(javax.servlet.http.HttpServletRequest, org.springframework.context.MessageSource) **根据Request读取Locale，并结合MessageSource，适配`LocalizationContext`实现,注入到ServletRequest的attribute中**





#### SpringLocalizationContext适配实现

主要实现`LocalizationContext#getResourceBundle`，利用`org.springframework.context.support.MessageSourceResourceBundle`完成MessageSource和ResourceBundle之间的适配工作







