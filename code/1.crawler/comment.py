# -*- codeing = utf-8 -*-
from bs4 import BeautifulSoup  # 网页解析，获取数据
import re  # 正则表达式，进行文字匹配`
import urllib.request, urllib.error  # 制定URL，获取网页数据
import xlwt  # 进行excel操作
import time
#import sqlite3  # 进行SQLite数据库操作
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from datetime import datetime
from pandas import Series, DataFrame
import xlwt
import xlrd
from openpyxl import *


def main():
     for q in range(250):
         #time.sleep(1)
         filename = 'C:\\Users\\13087\\Desktop\\movie\\豆瓣电影Top'+str(q+1)+'comment.xls'
         print(filename)
         data = pd.read_excel(filename)
         datanota=data.dropna(axis=0,how='any')
         temp=np.array(datanota)
         l=temp.tolist()
         print(l)
         length=len(l)
         book = xlwt.Workbook(encoding="utf-8", style_compression=0)  # 创建workbook对象
         sheet = book.add_sheet('Sheet1', cell_overwrite_ok=True)  # 创建工作表
         sheet.write(0, 0, "评论")
         sheet.write(0, 1, "日期")
         sheet.write(0, 2, "评分")
         sum=0
         for i in range(length):
             s1=l[i][0]
             sheet.write(i + 1, 0, s1)
             s2=l[i][1]
             ss2=s2[:10]
             sheet.write(i + 1, 1, str(ss2))
             s3=int(l[i][2])
             sum=sum+s3
         avg=int(sum/length)
         for i in range(length):
             s3=int(l[i][2])
             if s3!=0:
                 sheet.write(i+1,2,s3)
             if s3==0:
                 sheet.write(i + 1, 2, avg)
         savepath='C:\\Users\\13087\\Desktop\\gwc\\豆瓣电影Top'+str(q+1)+'.xls'
         book.save(savepath)
         ex = pd.read_excel(savepath)
         savepath2='C:\\Users\\13087\\Desktop\\comment\\Top'+str(q+1)+'.csv'
         print(savepath2)
         ex.to_csv(savepath2, encoding="utf_8_sig")
















if __name__ == "__main__":
     main()


