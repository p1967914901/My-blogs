# 实战：利用计算属性、指令等知识开发购物车
 **购物车需要展示 一个己加入购物车的商品列表，包含商品名称、商品单价、购买数量和操作等信息，还需要实时显示购买的总价。其中购买数量可以增加或减少，每类商品还可以从购物车中移除。也能添加商品到购物车中，最后一项是否选中该商品的功能，总价变为只计算选中商品的总价**
 实现效果如下:
 ![1](https://img-blog.csdnimg.cn/20200606132626333.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3AxOTY3OTE0OTAx,size_16,color_FFFFFF,t_70#pic_center)
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>购物车</title>
    <script src="https://cdn.staticfile.org/vue/2.2.2/vue.min.js"></script>
    <style>
        table{
            border: 1px solid #e9e9e9;
            border-collapse: collapse;
            border-spacing: 0;
            empty-cells: show;
        }
        th, td {
            padding: 8px 16px;
            border: 1px solid #e9e9e9;
            text-align: left;
        }
        th {
            background: #f7f7f7;
            color: #5c6b77;
            font-weight: 600;
            white-space: nowrap;
        }
    </style>
</head>
<body>
    <div id="app">
            商品名称:
            <input type="text" v-model="name">
            商品单价:
            <input type="text" v-model="price">
            购买数量:
            <input type="text" v-model="count">
            <button @click="newAdd()">添加</button>
            <table>
                <thead>
                    <th></th>
                    <th>商品名称</th>
                    <th>商品单价</th>
                    <th>购买数量</th>
                    <th>操作</th>
                    <th>是否选择</th>
                </thead>
                <tbody>
                    <tr v-for=" (item, index) in list">
                        <td>{{ index + 1 }}</td>
                        <td>{{ item.name }}</td>
                        <td>{{ item.price }}</td>
                        <td>
                            <button @click="Reduce(index)">-</button>
                            {{ item.count }}
                            <button @click="Add(index)">+</button>
                        </td>
                        <td>
                            <button @click="Remove(index)">移除</button>
                        </td>
                        <td>
                            <input type="checkbox" name="select" v-model="item.select"/>
                        </td>
                    </tr>
                </tbody>
            </table>
            <div>总价 {{ totalPrice }} 元</div>
        </template> 
        <div v-else>购物车为空</div>
    </div>
    <script>
        var app = new Vue({
            el: "#app",
            data: {
                list : [
                    {
                        id : 1,
                        name : "iPhone 7",
                        price : '6666',
                        count : 1,
                        select:true
                    },
                    {
                        id : 2,
                        name : "iPhone 7 pro",
                        price : '8888',
                        count : 1,
                        select:true
                    },
                    {
                        id : 3,
                        name : "iPad",
                        price : '7658',
                        count : 1,
                        select:true
                    },
                ],
                name:'',
                price:'',
                count:''
            },
            computed: {
                totalPrice : function() {
                    var s = 0;
                    for(let i = 0; i < this.list.length; i++)
                        if(this.list[i]["select"])
                            s += (this.list[i]["price"] * this.list[i]["count"]);
                    return s;
                }
            },
            methods : {
                Reduce(index) {
                    this.list[index].count = this.list[index].count == 1 ? 1 : this.list[index].count-1;
                },
                Add(index) {
                    this.list[index].count += 1;
                },
                Remove(index) {
                    this.list.splice(index, 1);
                },
                newAdd(){
                    this.list.push(
                        {
                            id : this.list.length,
                            name : this.name,
                            price : this.price,
                            count : this.count,
                            select: true
                        }
                    )
                }
            }
        })
    </script>
</body>
</html>
```