'''
    Convert csv file to format:
    'word1 word2 word3 __label__l1 __label__l2 __label__l3'
'''

import csv

input_file = "./allsides.csv"
output_file = "./allsides.txt"

def main():
    docs = []
    labels = []

    with open(input_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            docs.append(row[1])
            labels.append(row[2])

    with open(output_file, "w") as f:
        for word in docs:
            f.write(word + " ")

        f.write(" ")

        for label in labels:
            f.write("__label__" + label)

main()