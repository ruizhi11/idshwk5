from sklearn.ensemble import RandomForestClassifier
import numpy as mp

trainlist = []
testlist = []
class Domain:
    def __init__(self, _name, _label, _length, _numbers):
        self.name = _name
        self.label = _label
        self.length = _length
        self.numbers = _numbers

    def returnData(self):
        return [self.length, self.numbers]

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1

def countNumbers(str):
    num = 0
    for i in str:
        if i.isdigit():
            num = num + 1
    return num


def initTrain(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(',')
            name = tokens[0]
            label = tokens[1]
            length = len(tokens[0])
            numbers = countNumbers(tokens[0])
            trainlist.append(Domain(name, label, length, numbers))

def initTest(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            name = str(line)
            length = len(line)
            numbers = countNumbers(line)
            testlist.append(Domain(name, "", length, numbers))

def main():
    initTrain("train.txt")
    initTest("test.txt")
    featureMatrix = []
    labelList = []
    for item in trainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())

    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    with open("result.txt", 'w') as f:
        for i in testlist:
            if clf.predict([i.returnData()])[0] == 0:
                f.write(i.name+",notdga\n")
            else:
                f.write(i.name + ",dga\n")



if __name__ == '__main__':
    main()