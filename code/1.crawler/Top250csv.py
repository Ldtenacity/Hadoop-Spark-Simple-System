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

def cut(obj, sec):
    return [obj[i:i+sec] for i in range(0,len(obj),sec)]

def main():
    filename='C:\\Users\\13087\\Desktop\\Top250.xlsx'
    wb = load_workbook(filename)
    ws = wb.active
    ws.delete_cols(1)
    ws.delete_cols(1)
    ws.delete_cols(2)
    ws.delete_cols(4)
    wb.save('C:\\Users\\13087\\Desktop\\Top250new.xlsx')##去除唯一属性


    path = 'C:\\Users\\13087\\Desktop\\Top250new.xlsx'
    data = pd.DataFrame(pd.read_excel(path))#读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错

    book = xlwt.Workbook(encoding="utf-8", style_compression=0)  # 创建workbook对象
    sheet = book.add_sheet('Sheet1', cell_overwrite_ok=True)  # 创建工作表
    col = ("影片中文名", "评分", "评价数","id","类别","年份","拍摄地")
    for i in range(7):
        sheet.write(0,i,col[i])
    for i in range(250):
        s=data['相关信息'][i]
        l=s.split("  ")
        sheet.write(i+1,6,l[-2]) #拍摄地
        temp = l[-3]
        if l[-3][-1]!=")":
            sheet.write(i+1,5,str(temp[-4:]))
            ##print(temp[-4:])
        else:
            d=[]
            for j in temp:
                num=0
                if "0"<=j<="9":
                    d.append(j)
            dd="".join(d)
            new_dd=cut(dd,4)
            new_ddd=",".join(new_dd)
            sheet.write(i+1,5,str(new_ddd))#年份

        name=data["影片中文名"][i]
        sheet.write(i+1,0,name)
        score = data["评分"][i]
        sheet.write(i+1, 1,score)
        number = data["评价数"][i]
        sheet.write(i+1, 2,int(number))
        id = data["id"][i]
        sheet.write(i+1, 3,int(id))
        cato = data["类别"][i]
        sheet.write(i+1, 4,cato)
        savepath='C:\\Users\\13087\\Desktop\\豆瓣电影newTop250.xls'
    book.save(savepath)
    ex = pd.read_excel('C:\\Users\\13087\\Desktop\\豆瓣电影newTop250.xls')
    ex.to_csv('C:\\Users\\13087\\Desktop\\豆瓣电影newTop250.csv', encoding="gbk")



















if __name__ == "__main__":  # 当程序执行时
    # 调用函数
     main()