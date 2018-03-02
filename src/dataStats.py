import csv

def readMyFile(filename):
    with open(filename, newline='', encoding='utf-8') as csvfile:
        dataReader = csv.reader(csvfile, dialect='excel')

        sum = 0
        chars = 0
        words = 0
        for row in dataReader:
            sum = sum + 1
            chars = chars + len(row[1])
            words = words + len(row[1].split())

        print(filename + " entries: " + str(sum))
        print(filename + " characters: " + str(chars))
        print(filename + " average characters: " + str(chars / sum))
        print(filename + " words: " + str(words))
        print(filename + " average words: " + str(words / sum))


readMyFile('../data/train.csv')
print()
readMyFile('../data/test.csv')