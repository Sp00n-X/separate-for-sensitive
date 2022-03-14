# -*- coding: utf-8 -*-  
import os 

if __name__ == "__main__":
    with open("label.txt","a") as f:
        for files in os.listdir('D:\\PythonProjects\\for learn\\data2\\porn2\\'):
            fi = files
            sstr = r"/porn2/" + fi + " 0" +"\n"

            f.write(sstr)
