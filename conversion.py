'''
    Convert csv file to format:
    word1 __label__l1 
    word2 __label__l2
    word3 __label__l3
'''

import csv

input_file = "./allsides.csv"
output_file = "./allsides.txt"

def main():
    docs = []
    labels = []

    with open(input_file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            docs.append(row[1])
            labels.append(row[2])

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(len(docs)):
            f.write("{0} __label__{1}\n".format(docs[i].replace("\n", ""), labels[i]))

main()