# import tensorflow as tf
# import csv
# # def read_tsv(input_file, quotechar=None):
# # 	"""Reads a tab separated value file."""
# # 	lines = []
# # 	with tf.gfile.Open(input_file, "r") as f:
# # 		reader = csv.reader(f)
# # 		n = 0
# # 		for line in reader:
# # 			if n > 10:
# # 				return lines

# # 			if len(line) == 3:
# # 				temp = line[1]
# # 				temp.replace("?€?","")
# # 				lines.append(line)
# # 				print(111)
# # 			n+=1
# # 		print(len(lines))
# # 		return lines
# # ans = read_tsv("./data/train.csv")
# # writer = csv.writer()
# #     writer.writerows(lines)
# # import re
# lines = []
# # with tf.gfile.Open("./data/allsides.csv", "r") as f:
# # 	reader = csv.reader(f)
# # 	n,i = 0,0
# # 	j = 0
# # 	for line in reader:
# # 		if len(line) == 3 and line[2] in set(["l","r","c"]):
# # 				line[1] = re.sub(r'[^\w\s]','',line[1] )
# # 				lines.append(line[1:])
# # 				j+=1
# # 		n+=1
# # 	print(n,i,j)
# # print(111)

# # with open('./data/automl.csv', 'w',encoding = "utf-8",newline='') as writeFile:
# #     writer = csv.writer(writeFile)
# #     writer.writerows(lines)

# with tf.gfile.Open("./data/allNews.csv", "r") as f:
# 	reader = csv.reader(f)
# 	for line in reader:
# 		if len(line) == 2:
# 			newline = [line[1],line[0]]
# 			lines.append(newline)
# print(1112)
# assert(len(lines[0][0])==1)
# with open('./data/allNews.csv', 'w',encoding = "utf-8",newline='') as writeFile:
#     writer = csv.writer(writeFile)
#     writer.writerows(lines)

# writeFile.close()

# 		

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
print(sigmoid(-0.6))