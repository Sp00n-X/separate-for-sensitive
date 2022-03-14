# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import re
import urllib.request,urllib.error
import sqlite3
import xlwt
import requests
import time
'''
url = "https://movie.douban.com/explore#!type=movie&tag=%E7%83%AD%E9%97%A8&sort=recommend&page_limit=20&page_start=0"
headers = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.146 Safari/537.36"}
data = bytes(urllib.parse.urlencode({'name':'geng'}),encoding="utf-8")
req = urllib.request.Request(url=url,data=data,headers=headers,method="POST")
response = urllib.request.urlopen(req)
print(response.read().decode("utf-8"))
'''
'''
se = requests.session()
headers = {'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'
'Accept-Encoding':'gzip, deflate'
'Accept-Language':'zh-CN,zh;q=0.9'
'Cache-Control':'max-age=0'
'Cookie':'NB_SRVID=srv106210'
'Host':'cdn.awwni.me'
'If-Modified-Since':' Fri, 19 Feb 2021 11:40:45 GMT'
'If-None-Match': "602fa3bd-18f19"
'Proxy-Connection':'keep-alive'
'Upgrade-Insecure-Requests': '1'
'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36'}
se.headers.update(headers)
'''
file = open("test d.txt") 
folder_path = 'D:\\PythonProjects\\for learn\\data\\drawing\\'

while 1:
  kk = file.readline()
  line = kk.strip('\n')
  if not line:
    break
  pass # do something
  print("开始下载：" + line)
  filename = line.split('/')[-1]
  filepath = folder_path + filename
  print(filepath)
  
  try:
      urllib.request.urlretrieve(line,filename=filepath)
      print("Downloading:")

  except Exception as e :
      print("下载时错误")
      print(e)
  print("下载完成")
file.close()
