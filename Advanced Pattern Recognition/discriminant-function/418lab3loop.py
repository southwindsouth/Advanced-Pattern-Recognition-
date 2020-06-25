import numpy as np
import random
import math

# Read file
random.seed(17)
f = open('./breast-cancer-wisconsin.data', 'r+')
dataset = f.readlines()
f.close()
print('input data','length:',len(dataset), ',part of data:',dataset[0:2])

random.shuffle(dataset)

# Separate string data according to commas
for i in range(len(dataset)):
    dataset[i] = dataset[i].split(',')
print('split input data','length:',len(dataset[i]),',part of data:',dataset[0:2])

# Convert string data to numeric data
data = []
for i in range(len(dataset)):
    for j in range(len(dataset[0])):
        data.append(int(dataset[i][j]))
# Rearrange to original dimension
new_data = np.asarray(data).reshape(len(dataset), 11)
print('numeric data(raw data)', 'type:', type(new_data[0][0]), ',shape:', new_data.shape)
print('part of data:', new_data[0:2])

# shuffle data with the random.shuﬄe method
# random.shuffle(new_data)
# print('shuffle data:',new_data)

#
data_set = []
data_label = []
x = new_data
x_data = x[:, 1:x.shape[1]-1]
x_label = x[:, x.shape[1]-1]

index = math.floor(x.shape[0]/5)
data_set0 = x_data[0:index]
data_set1 = x_data[index:index*2]
data_set2 = x_data[index*2:index*3]
data_set3 = x_data[index*3:index*4]
data_set4 = x_data[index*4:]
data_label0 = x_label[0:index]
data_label1 = x_label[index:index*2]
data_label2 = x_label[index*2:index*3]
data_label3 = x_label[index*3:index*4]
data_label4 = x_label[index*4:]
print('type',type(data_set1))


# five sets
print('five set', 'data1 shape',data_set0.shape,'data2 shape',data_set1.shape,'data3 shape',data_set2.shape,'data4 shape',data_set3.shape,'data5 shape',data_set4.shape,'part of data',data_set0[0:5])
# five labels
print('five set', 'data1 shape',data_label0.shape,'data2 shape',data_label1.shape,'data3 shape',data_label2.shape,'data4 shape',data_label3.shape,'data5 shape',data_label4.shape,'part of data',data_label0[0:4])


whole_data = np.array([data_set0,data_set1,data_set2,data_set3,data_set4])
whole_label = np.array([data_label0,data_label1,data_label2,data_label3,data_label4])
print('pingjie',whole_data.shape,whole_data[4].shape,whole_label.shape,whole_label[0].shape,whole_label[1].shape,whole_label[2].shape,whole_label[3].shape,whole_label[4].shape)

new = np.vstack((data_set0,data_set1,data_set2,data_set3,data_set4))

old = np.hstack((whole_label[0],whole_label[1],whole_label[2],whole_label[3],whole_label[4]))
# np.concatenate((d,b),axis=1)
print('test',new.shape,old.shape,new[0:5],old[0:5])

str = [0,1,2,3,4]
str.remove(0)
print('hhhh',str,str[0],str[3])
old = np.vstack((whole_label[str[0]].shape,whole_label[str[1]].shape,whole_label[str[2]].shape,whole_label[str[3]].shape))

accuracy_rate = []

for num in range(5):
    for liter in range(len(str)+1):
        str = [0, 1, 2, 3, 4]
        str.remove(liter)
        training_set = np.vstack((whole_data[str[0]],whole_data[str[1]],whole_data[str[2]],whole_data[str[3]]))[:]
        training_label = np.hstack((whole_label[str[0]],whole_label[str[1]],whole_label[str[2]],whole_label[str[3]]))[:]
        test_set = whole_data[liter][:]
        test_label = whole_label[liter][:]

        # Training stage
        # Augment the feature vector x with an additional constant dimension
        training_set = np.insert(training_set, training_set.shape[1], values=1, axis=1)

        # Scale linearly the attribute values xij into [−1,1] for each dimensional feature as follows:
        training_set = np.float64(training_set)
        test_set = np.float64(test_set)

        # method 1
        #     max_i = np.max(x,axis=0)
        #     min_i = np.min(x,axis=0)
        #     return (2*(x-min_i+10**-6)/(max_i-min_i+10**-6)-1)
        #     xij = function_A(X)

        # method 2
        for i in range(training_set.shape[1]):
            one_column = training_set[:,i]
            min = np.min(one_column)
            max = np.max(one_column)
            training_set[:,i] = 2*(one_column-min+10**-6)/(max-min+10**-6) - 1

        #  Reset the example vector x according its label y
        for i in range(training_set.shape[0]):
            if(training_label[i] == 4):
                training_set[i] = -training_set[i]

        # (XTX)−1XT1
        one =np.ones([training_set.shape[0], 1])
        k = np.matmul(training_set.transpose(), training_set)
        w = np.matmul(np.dot(np.linalg.inv(k), training_set.transpose()), one)

        test_set = np.insert(test_set, test_set.shape[1], values=1, axis=1)
        for i in range(10):
            one_column = test_set[:,i]
            min = np.min(one_column)
            max = np.max(one_column)
            test_set[:,i] = 2*(one_column-min+10**-6)/(max-min+10**-6) - 1
        new_label = np.dot(test_set, w)

        c = []
        for i in range(new_label.shape[0]):
            if(new_label[i] >= 0):
                c.append(2)
            else:
                c.append(4)

        num = np.asarray(c)-test_label
        rate = np.sum(num == 0)/test_label.shape[0]
        accuracy_rate.append(rate)

print('the average accuracy rate is:\n ',np.mean(accuracy_rate),'\n length of calculation:\n',len(accuracy_rate),'\nspecific average accuracy:\n',np.reshape(accuracy_rate,(25,1)))