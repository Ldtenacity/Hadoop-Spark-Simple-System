# -*- codeing = utf-8 -*-
from bs4 import BeautifulSoup  # 网页解析，获取数据
import re  # 正则表达式，进行文字匹配`
import urllib.request, urllib.error  # 制定URL，获取网页数据
import xlwt  # 进行excel操作
import time
import random
import re
import time
import requests
import threading
from lxml import etree
from bs4 import BeautifulSoup
from queue import Queue
from threading import Thread
import pandas as pd


findcomment = re.compile(r'<span class="short">(.*)</span>')
findtime=re.compile(r'<span class="comment-time" title="(.*)"')
findstar_list=re.compile(r'<span class="(.*)" title="(.*)"></span>')
findTitle = re.compile(r'<p class="pl2">&gt; <a href="(.*)">去 (.*) 的页面</a></p>')
io = 'C:\\Users\\13087\\Desktop\\movie\\Top250.xls'
df = pd.read_excel(io)




def askURL(url):
    pc_agent = [
        "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
        "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0);",
        "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
        "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
        "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
        "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36"
        "Mozilla/5.0 (X11; Linux x86_64; rv:76.0) Gecko/20100101 Firefox/76.0"
    ]
    agent = random.choice(pc_agent)
    head = {'User-Agent': agent}
    # 用户代理，表示告诉豆瓣服务器，我们是什么类型的机器、浏览器（本质上是告诉浏览器，我们可以接收什么水平的文件内容）

    request = urllib.request.Request(url, headers=head)
    html = ""
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode("utf-8")
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print(e.code)
        if hasattr(e, "reason"):
            print(e.reason)
    return html

# def login():
#     login_url="https://accounts.douban.com/passport/login?source=movie"
#     login_headers={}
def run(q):   ##q="id"

    while q.empty() is not True:
        datalist2 = []
        qq=q.get()

        j=0

        for i in range(0, 20):
            time.sleep(1)
            url = "https://movie.douban.com/subject/" + str(qq) + "/comments?start=" + str(
                i * 20) + "&limit=20&status=P&sort=new_score"
            print(url)
            html = askURL(url)
            soup = BeautifulSoup(html, "html.parser")
            # for item in soup.find_all('p', class_="pl2"):  # 查找符合要求的字符串
            #     j = j + 1
            #     #print(item)
            #     if j==1:
            #         #print(re.findall(r"\"keywords\">\n<meta content=\"(.+?)短评"))
            #         title = (re.findall(findTitle[1], str(item)))[0]
            #         print(title)
            for item in soup.find_all('div', class_="comment"):  # 查找符合要求的字符串

                data = []  # 保存一部电影所有信息

                comment = re.findall(findcomment, str(item))
                comment_time = re.findall(findtime, str(item))
                comment_star = re.findall(findstar_list, str(item))
                if len(comment_star) == 0:
                    num1 = 0.0
                else:
                    star = comment_star[0][0]
                    num = int(star[7:9])
                    num1 = num / 5
                # print(num1)
                # print(comment_time)
                #            print(comment)
                data.append(comment)
                data.append(comment_time)
                data.append(num1)
       #         print(data)
                datalist2.append(data)
        book = xlwt.Workbook(encoding="utf-8", style_compression=0)  # 创建workbook对象
        sheet = book.add_sheet('豆瓣电影Top1comment', cell_overwrite_ok=True)  # 创建工作表
        col = ("评论", "时间", "评分")
        i = 0
        sheet.write(0, 0, col[0])
        sheet.write(0, 1, col[1])
        sheet.write(0, 2, col[2])

        for item in datalist2:
            data = item
        #print(data)
            sheet.write(i + 1, 0, data[0])
            sheet.write(i + 1, 1, data[1])
            sheet.write(i + 1, 2, data[2])
            i = i + 1




        # a=df[df['id'].isin([int(qq)])]["影片中文名"]
        # print(a[0])
        a = df[df['id'].isin([int(qq)])].index.values[0]

        savepath2 = "豆瓣电影Top" +str(a+1) + "comment.xls"
        print(savepath2)
        book.save(savepath2)
        q.task_done()



def main():
    queue=Queue()
    # io='C:\\Users\\13087\\Desktop\\movie\\Top250.xls'
    # df = pd.read_excel(io)
    df_li=df.values.tolist()
    result=[]
    for s_li in df_li:
        result.append(s_li[8])

    for i in result:
        queue.put(str(i))

    for i in range(10):
        thread = Thread(target=run, args=(queue,))
        thread.daemon = True  # 随主线程退出而退出
        thread.start()
    queue.join()  # 队列消费完 线程结束

if __name__ == "__main__":  # 当程序执行时
    # 调用函数
     main()