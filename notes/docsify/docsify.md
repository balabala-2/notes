## docsify配置

1. 全局安装 `docsify-cli` 工具，可以方便创建及本地预览文档网站： 

   `npm i docsify-cli -g`

2. 初始化: 

   `docsify init ./docs`

3. 运行本地服务器：

   `docsify serve docsify`

## More pages

示例目录结构如下

```
.
└── docs
    ├── README.md
    ├── guide.md
    └── zh-cn
        ├── README.md
        └── guide.md
```

配置路由为：

```
docs/README.md        => http://domain.com
docs/guide.md         => http://domain.com/#/guide
docs/zh-cn/README.md  => http://domain.com/#/zh-cn/
docs/zh-cn/guide.md   => http://domain.com/#/zh-cn/guide
```

### 侧边栏

```html
<!-- index.html -->
<script>
  window.$docsify = {
    //create _sidebar.md
    loadSidebar: true
    //create other
    loadSidebar: 'xxx.md'
  }
</script>
```

- _sidebar.md
`subMaxLevel = 2`表示能显示的子级标题层数
`{docsify-ignore}` 忽略特定标题；
`{docsify-ignore-all}` 忽略特定页面下的所有标题；

在`_siderbar.md`中，配置：

```html
<!-- docs/_sidebar.md -->

* [Home](/)
* [Guide](guide.md)
```

### Nested Sidebars嵌套侧边栏

specify `alias` to avoid unnecessary fallback.

```html
<script>
  window.$docsify = {
    loadSidebar: true,
    alias: {
      '/.*/_sidebar.md': '/_sidebar.md'
    }
  }
</script>
```
<a href="https://docsify.js.org/#/" style="color: #4384f6">docsify官方文档</a>




