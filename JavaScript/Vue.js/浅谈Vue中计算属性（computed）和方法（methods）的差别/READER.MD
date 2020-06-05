# 浅谈Vue中计算属性（computed）和方法（methods）的差别

* methods方法和computed计算属性，两种方式的最终结果确实是完全相同

* 计算属性是基于它们的响应式依赖进行缓存的。只在相关响应式依赖发生改变时它们才会重新求值，多次访问计算属性会立即返回之前的计算结果，而不必再次执行函数。计算属性是基于他们的依赖进行缓存的，只有在相关依赖发生改变时，才会重新求值，

* methods方法，每当触发重新渲染时，调用方法将总会再次执行函数。

* 使用计算属性还是 methods 取决于你是否需要缓存，当遍历大数组和做大量计算时，应当使用 计算属性，除非你不希望得到缓存。
* 对于任何复杂逻辑，你都应当使用计算属性
 ##  例如

```html
<body>
	<div id="app">
		{{f}}
		<br>
		<hr>
		{{ff()}}
	</div>

	<script>
		var k = 10;
		var app = new Vue({
			el: '#app',
			data: {
				a: 5,
				b: 1
			},
			computed: {
				f() {
					console.log("f");
					return this.a + this.b + k;
				}
			},
			methods: {
				ff() {
					console.log("ff");
					return this.a + this.b + k;
				}
			}
		})
	</script>
</body>
```

## 运行结果如下：
![1](https://img-blog.csdnimg.cn/20200605161923228.png#pic_center)
![1](https://img-blog.csdnimg.cn/20200605161939896.png#pic_center)

### 接下来我们在控制台进行一系列操作帮助理解它们的区别
![1](https://img-blog.csdnimg.cn/20200605162133644.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3AxOTY3OTE0OTAx,size_16,color_FFFFFF,t_70#pic_center)
**我们发现，当我们改变全局变量k的值时，再次调用方法时结果会发生改变，而计算属性的值却并没有发生改变，（因为k并不是Vue实例里data属性的值）**