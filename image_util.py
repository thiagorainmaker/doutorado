#
import os
import pandas as pd


print(os.path.isfile("datasets/kaggle/512/test/37951_left.jpeg"))
print(os.path.isfile("datasets/kaggle/512/test/1_right.jpeg"))
print(os.path.isfile("datasets/kaggle/512/test/1_right.jpeg"))


file = pd.read_csv("datasets/kaggle/test.csv")

for date, row in file.T.iteritems():

    if row[0] == "37951_left":
        print(row[0])
    if os.path.isfile("datasets/kaggle/512/test/"+row[0]+".jpg") == False and os.path.isfile("datasets/kaggle/512/test/"+row[0]+".jpeg") == False:
        print("ERRO")
        print(row[0])
        break
#print("all images is ok")
