"""
    txt-csv-pray
"""
import numpy as np
import pandas as pd
import csv


# persons = [('Tom', 20, 180), ('Allen', 21, 182), ('Jerry', 22, 181)]
# headers = ('name', 'age', 'height')
# with open('persons.csv', 'w', encoding='utf-8', newline='') as f:
#     write = csv.writer(f)  # 创建writer对象
#     write.writerow(headers)
#     # 写内容，writerrow 一次只写入一行
#     for data in persons:
#         write.writerow(data)

def txt_csv(datas):
    heads = tuple(datas[0].split(";"))
    print(heads)
    datas = datas[1:]
    with open("../data/pre_data/pre_data1.csv", "w", encoding="utf-8", newline="") as f:

        write = csv.writer(f)
        write.writerow(heads)
        i =0
        for data in datas:
            line = data.rstrip('\n')
            sls = tuple(line.split(";"))
            write.writerow(sls)
            i +=1
            if i ==1000:
                break


if __name__ == "__main__":
    path = '../data/raw_data'
    file_name = "household_power_consumption.txt"
    with open(path+"/"+file_name,encoding="utf-8") as f:

       data_txt = f.readlines()
       txt_csv(data_txt)