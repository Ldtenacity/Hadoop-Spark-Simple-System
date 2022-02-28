# -*- codeing = utf-8 -*-
from bs4 import BeautifulSoup  # 网页解析，获取数据
import re  # 正则表达式，进行文字匹配
import urllib.request, urllib.error  # 制定URL，获取网页数据
import xlwt  # 进行excel操作
import time
#import sqlite3  # 进行SQLite数据库操作
import random
findLink = re.compile(r'<a href="(.*?)">')  # 正则表达式对象
findImgSrc = re.compile(r'<img.*src="(.*?)"', re.S)
findTitle = re.compile(r'<span class="title">(.*)</span>')
findRating = re.compile(r'<span class="rating_num" property="v:average">(.*)</span>')
findJudge = re.compile(r'<span>(\d*)人评价</span>')
findInq = re.compile(r'<span class="inq">(.*)</span>')
findBd = re.compile(r'<p class="">(.*?)</p>', re.S)
findcomment = re.compile(r'<span class="short">(.*)</span>')
findtime=re.compile(r'<span class="comment-time" title="(.*)"')
findstar_list=re.compile(r'<span class="(.*)" title="(.*)"></span>')
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////



def main():
    baseurl = "https://movie.douban.com/top250?start="  #要爬取的网页链接

    datalist = getData(baseurl)
    savepath = "豆瓣电影Top250.xls"    #当前目录新建XLS，存储进去
            #当前目录新建数据库，存储进去
    # 3.保存数据
    saveData(datalist,savepath)      #2种存储方式可以只选择一种
#    datalist2 = getData2(1292052)
#    print(datalist2)
#    savepath2 = "豆瓣电影Top1comment.xls"
#    print("save3.......")

#    saveData2(datalist2, savepath2)




def getData2(id):
    datalist2 = []
    print(id)
    for i in range(0,20):
        time.sleep(1)
        url="https://movie.douban.com/subject/"+str(id)+"/comments?start="+str(i*20)+"&limit=20&status=P&sort=new_score"
        print(url)
        html = askURL(url)
        soup = BeautifulSoup(html, "html.parser")
        for item in soup.find_all('div', class_="comment"):  # 查找符合要求的字符串
            data = []  # 保存一部电影所有信息
            comment=re.findall(findcomment,str(item))
            comment_time=re.findall(findtime,str(item))
            comment_star=re.findall(findstar_list,str(item))
            if len(comment_star)==0:
                num1=0.0
            else:
                star=comment_star[0][0]
                num=int(star[7:9])
                num1=num/5
            #print(num1)
            #print(comment_time)
#            print(comment)
            data.append(comment)
            data.append(comment_time)
            data.append(num1)
            datalist2.append(data)
            ##print(datalist2)

    print(datalist2)
    return datalist2

# 爬取网页
def getData(baseurl):
    datalist = []  #用来存储爬取的网页信息
    for i in range(0, 10):  # 调用获取页面信息的函数，10次
        url = baseurl + str(i * 25)
        html = askURL(url)  # 保存获取到的网页源码
        # 2.逐一解析数据
        soup = BeautifulSoup(html, "html.parser")
        for item in soup.find_all('div', class_="item"):  # 查找符合要求的字符串
            data = []  # 保存一部电影所有信息
            item = str(item)
            link = re.findall(findLink, item)[0]  # 通过正则表达式查找
            data.append(link)
            imgSrc = re.findall(findImgSrc, item)[0]
            data.append(imgSrc)
            titles = re.findall(findTitle, item)
            if (len(titles) == 2):
                ctitle = titles[0]
                data.append(ctitle)
                otitle = titles[1].replace("/", "")  #消除转义字符
                data.append(otitle)
            else:
                data.append(titles[0])
                data.append(' ')
            rating = re.findall(findRating, item)[0]
            data.append(rating)
            judgeNum = re.findall(findJudge, item)[0]
            data.append(judgeNum)
            inq = re.findall(findInq, item)
            if len(inq) != 0:
                inq = inq[0].replace("。", "")
                data.append(inq)
            else:
                data.append(" ")
            bd = re.findall(findBd, item)[0]
            bd = re.sub('<br(\s+)?/>(\s+)?', "", bd)
            bd = re.sub('/', "", bd)
            data.append(bd.strip())
            datalist.append(data)
            #/////////////////////////////////////////////////////////////////////////////////

            #/////////////////////////////////////////////////////////////////////////////////
    return datalist
def get_headers(use='pc'):
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
    headers = {'User-Agent': agent}
    return headers

# 得到指定一个URL的网页内容
def askURL(url):
    head = get_headers()
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

def saveData2(datalist2,savepath2):
    print("save2.......")
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)  # 创建workbook对象
    sheet = book.add_sheet('豆瓣电影Top1comment', cell_overwrite_ok=True)  # 创建工作表
    col = ("评论","时间","评分")
    i = 0
    sheet.write(0,0,col[0])
    sheet.write(0, 1, col[1])
    sheet.write(0, 2, col[2])
    for item in datalist2:
        data = item
        print(data)
        sheet.write(i+1,0,data[0])
        sheet.write(i+1,1,data[1])
        sheet.write(i+1,2,data[2])
        i=i+1


    if i==161 or i==162 or i==163 or i==188 or i==189 or i==191 or i==200:

        book.save(savepath2)


# 保存数据到表格
def saveData(datalist,savepath):
    print("save.......")
    book = xlwt.Workbook(encoding="utf-8",style_compression=0) #创建workbook对象
    sheet = book.add_sheet('豆瓣电影Top250', cell_overwrite_ok=True) #创建工作表
    #2.、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、
    col = ("电影详情链接","图片链接","影片中文名","影片外国名","评分","评价数","概况","相关信息","id","类别")
    for i in range(0,10):
        sheet.write(0,i,col[i])  #列名
    for i in range(0,250):
        # print("第%d条" %(i+1))       #输出语句，用来测试
        data = datalist[i]
        for j in range(0,8):
            sheet.write(i+1,j,data[j])  #数据
        if data[0][-9]=='/':
            sheet.write(i+1,8,data[0][-8:-1])
            id = data[0][-8:-1]
        else:
            sheet.write(i+1,8,data[0][-9:-1])
            id = data[0][-9:-1]
        l=data[7].split("  ")
        print(data[7])
        print(l)
        print(l[-1])
        sheet.write(i+1,9,l[-1])
        datalist2 = getData2(id)
        print(datalist2)
        savepath2 = "豆瓣电影Top"+str(i+1)+"comment.xls"
        print("savetop250.......")
        saveData2(datalist2, savepath2)
    print("save3.......")
    # for i in range(0,250):
    #     datalist = getData2()
    #     savepath = "豆瓣电影Top250.xls"
        #/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    book.save(savepath) #保存



if __name__ == "__main__":  # 当程序执行时
    # 调用函数
     main()
    # init_db("movietest.db")
     print("爬取完毕！")

