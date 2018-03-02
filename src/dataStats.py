import csv

def readMyFile(filename):
    with open(filename, newline='', encoding='utf-8') as csvfile:
        dataReader = csv.reader(csvfile, dialect='excel')

        sum = 0
        chars = 0
        for row in dataReader:
            chars = chars + len(row[1])
            sum = sum + 1

        print(filename + " entries: " + str(sum))
        print(filename + " characters: " + str(chars))
        print(filename + " average: " + str(chars/sum))


readMyFile('../data/train.csv')
print()
readMyFile('../data/test.csv')