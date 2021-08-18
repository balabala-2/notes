# Nginx

*Nginx*是一个高性能的[HTTP](https://baike.baidu.com/item/HTTP)和[反向代理](https://baike.baidu.com/item/反向代理/7793488)web服务器，同时也提供了IMAP/POP3/SMTP服务。

其将源代码以类[BSD许可证](https://baike.baidu.com/item/BSD许可证/10642412)的形式发布，因它的稳定性、丰富的功能集、简单的配置文件和低系统资源的消耗而闻名。

Nginx是一款轻量级的Web 服务器/反向代理服务器及电子邮件（IMAP/POP3）代理服务器，在BSD-like 协议下发行。其特点是`占有内存少，并发能力强`，事实上nginx的并发能力在同类型的网页服务器中表现较好，中国大陆使用nginx网站用户有：百度、京东、新浪、网易、腾讯、淘宝等。

- 正向代理：代理用户去访问服务器
- 反向代理：代理服务器去请求数据
- 负载均衡：将负载（工作任务）进行平衡、分摊到多个操作单元上进行运行
- 动静分离：指在web服务器架构中，将静态页面与动态页面或者静态内容接口和动态内容接口分开不同系统访问的架构设计方法，进而提升整个服务访问性能和可维护性。

## Nginx常用命令

> 验证配置是否正确: nginx  -t  
>
> 查看Nginx的详细的版本号：nginx  -V  
>
> 查看Nginx的简洁版本号：nginx  -v  
>
> 启动Nginx：start  nginx 
>
> 快速停止或关闭Nginx：nginx   -s   stop 
>
> 正常停止或关闭Nginx：nginx   -s   quit 
>
> 重新加载配置文件：nginx   -s  reload

## Nginx配置

在项目使用中，使用最多的三个核心功能是静态服务器、反向代理和负载均衡。

这三个不同的功能的使用，都跟Nginx的配置密切相关，Nginx服务器的配置信息主要集中在"nginx.conf"这个配置文件中，并且所有的可配置选项大致分为以下几个部分.

```nginx
全局配置

events { # 工作模式配置

}

http { # http设置
    http配置
    upstream DIYname{
        # 负载均衡配置
        serve 127.0.0.1:8080 weight=1;
        serve 127.0.0.1:8081 weight=1;
    }
    server { # 服务器主机配置（虚拟主机、反向代理等）
        listen 80;
        server_name localhost;
        
        //代理
        location / {  # 根路由配置（虚拟目录等）
            root html;
            index index.html index.htm;
            # 反向代理：只要是根目录下的请求就代理到下面的网址
            proxy_pass http://localhost
        }

        location path {
            ....
        }

        location otherpath {
            ....
        }
    }
 
    server {
        ....

        location {
            ....
        }
    }
}
```
**main模块**

> - user    用来指定nginx worker进程运行用户以及用户组，默认nobody账号运行
> -  worker_processes    指定nginx要开启的子进程数量，运行过程中监控每个进程消耗内存(一般几M~几十M不等)根据实际情况进行调整，通常数量是CPU内核数量的整数倍
> -  error_log    定义错误日志文件的位置及输出级别【debug / info / notice / warn / error / crit】
> -  pid    用来指定进程id的存储文件的位置
> -  worker_rlimit_nofile    用于指定一个进程可以打开最多文件数量的描述


**event模块**

>- worker_connections    指定最大可以同时接收的连接数量，这里一定要注意，最大连接数量是和worker processes共同决定的。
>- multi_accept    配置指定nginx在收到一个新连接通知后尽可能多的接受更多的连接
>- use epoll    配置指定了线程轮询的方法，如果是linux2.6+，使用epoll，如果是BSD如Mac请使用Kqueue

**http模块**

>- 作为web服务器，http模块是nginx最核心的一个模块，配置项也是比较多的，项目中会设置到很多的实际业务场景，需要根据硬件信息进行适当的配置。

## 配置代码

```nginx
#user  nobody;
worker_processes  1;
events {
    worker_connections  1024;
}
http {
    include       mime.types;
    default_type  application/octet-stream;
    sendfile        on;
    keepalive_timeout  65;

 #1 start
	upstream linuxidc {
			server localhost:7071;
			server localhost:7072;
			server localhost:7073;
	}
   server {
       listen      7070;
       server_name  localhost;
       location / {
          # root    C:/ngtest2;
         # index  index.html index.htm;
         proxy_pass http://linuxidc;
        }
    }
# 1 end
 #2 start
   server {
       listen      7071;
       server_name  localhost;
       location / {
           root    C:/ngtest1;
         #index  index.html index.htm;
         #proxy_pass https://tms;
         #proxy_pass https://www.baidu.com/;
        }
    }
   server {
       listen      7072;
       server_name  localhost;
       location / {
           root    C:/ngtest2;
         # index  index.html index.htm;
         #proxy_pass https://tms;
        }
    }
   server {
       listen      7073;
       server_name  localhost;
       location / {
          root    C:/ngtest3;
         # index  index.html index.htm;
         #proxy_pass https://tms;
        }
    }

# 2 end
 #3 start
    server {
        listen       8080;
        server_name  localhost;

        #charset koi8-r;

        #access_log  logs/host.access.log  main;
        
        #location / {
          #  root   C:\ngtest;
            #index  index.html index.htm;
            #proxy_pass https://www.baidu.com/;
       # }

        location /baidu {
            #root   html;
            #index  index.html index.htm;
            proxy_pass https://www.baidu.com/;
        }
        location /csdn {
            #root   html;
            #index  index.html index.htm;
            proxy_pass https://www.csdn.net/;
        }
        error_page   500 502 503 504  /50x.html;
        location = /50x.html {
            root   html;
        }
        # 3 end
    }
}
```



