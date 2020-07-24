import sklearn
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection
import numpy as np
import math

l = []

trainfile = open("../class-master_2/takasaki_temp.csv")
trainline = trainfile.read()
trainline = [[format (s1) for s1 in s0.split(",")] for s0 in trainline.strip().split("\n")]

testfile = open("../class-master_2/takasaki_temp_test.csv")
testline = testfile.read()
testline = [[format (s1) for s1 in s0.split(",")] for s0 in testline.strip().split("\n")]

labelfile = open("../class-master_2/takasaki_temp_test.label")
testlabel = labelfile.read()
testlabel = [[format (s1) for s1 in s0.split(",")] for s0 in testlabel.strip().split("\n")]

"""
def split_list(l, n):

    reference : https://www.python.ambitious-engineer.com/archives/1843

    for idx in range(0, len(l), n):
        yield l[idx:idx + n]

for i in range(0, len(trainline)):
    l.append(trainline[i][1])

result = list(split_list(l, 144))
#print(result[1][1])

"""
    
train_x,train_y, test_x, test_y = [],[],[],[]


for a in range(0, 1873):
    train_x.append(trainline[a][1])
    arr_trainx = np.array(train_x)

for b in range(1874, 2017):
    train_y.append(trainline[b][1])
    arr_trainy = np.array(train_y)

for c in range(0, len(testline)):
    test_x.append(testline[c][1])
    arr_testx = np.array(test_x)

for d in range(0, len(testlabel)):
    test_y.append(testlabel[d][1])
    arr_testy = np.array(test_y)
    
clf = RandomForestClassifier(max_depth=30, n_estimators=30, random_state=42)
clf.fit(arr_trainx.reshape(1,-1), arr_trainy.reshape(1,-1))

y_pred = clf.predict(arr_testx.reshape(1,-1))
accuracy = accuracy_score(arr_testy.reshape(1,-1), y_pred)
print('Accuracy: {}'.format(accuracy))



