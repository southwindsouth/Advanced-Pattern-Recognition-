import numpy as np
import random
import math

# Read file
random.seed(17)
f = open('./breast-cancer-wisconsin.data', 'r+')
dataset = f.readlines()
f.close()
print('original input data','\nlength:',len(dataset), '\npart of data[0:2]:',dataset[0:2])
# Separate string data according to commas
for i in range(len(dataset)):
    dataset[i] = dataset[i].split(',')
print('split input data','\nlength:',len(dataset),'\npart of data[0:2]:',dataset[0],'\n',dataset[1])
# Convert string data to numeric data
data = []
for i in range(len(dataset)):
    for j in range(len(dataset[0])):
        data.append(int(dataset[i][j]))
# Rearrange to original dimension
new_data = np.asarray(data).reshape(len(dataset), 11)
print('numeric data(raw data)', '\ntype:', type(new_data[0][0]), '\nshape:', new_data.shape)
# shuffle data with the random.shuﬄe method
# random.shuffle(new_data)
# Split the dataset as ﬁve parts
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
# five sets
print('five dataset', '\ndata1 shape:',data_set0.shape,'\ndata2 shape:',data_set1.shape,'\ndata3 shape:',data_set2.shape,'\ndata4 shape:',data_set3.shape,'\ndata5 shape:',data_set4.shape,'\npart of data[0:5]:',data_set0[0:5])
# five labels
print('five labelset', '\ndata1 label:',data_label0.shape,'\ndata2 label:',data_label1.shape,'\ndata3 label:',data_label2.shape,'\ndata4 label:',data_label3.shape,'\ndata5 label:',data_label4.shape,'\npart of label[0:5]:',data_label0[0:5])
# Splicing data into a three-dimensional array for easy index extraction in training and test
whole_data = np.array([data_set0,data_set1,data_set2,data_set3,data_set4])
whole_label = np.array([data_label0,data_label1,data_label2,data_label3,data_label4])






accuracy_rate = []

training_set = np.float64(x_data[:])
training_label = x_label[:]




# Training stage
# Augment the feature vector x with an additional constant dimension
training_set = np.insert(training_set, training_set.shape[1], values=1, axis=1)
print('training data plus 1','\ntype:\n',type(training_set[0][0]),'\npart of data:\n',training_set[0:5])

# Scale linearly the attribute values xij into [−1,1] for each dimensional feature as follows:
# method 1
max_i = np.max(training_set,axis=0)
min_i = np.min(training_set,axis=0)
training_set = 2*(training_set-min_i+10**-6)/(max_i-min_i+10**-6)-1
# method 2
# for i in range(training_set.shape[1]):
#     one_column = training_set[:,i]
#     min = np.min(one_column)
#     max = np.max(one_column)
#     training_set[:,i] = 2*(one_column-min+10**-6)/(max-min+10**-6) - 1

print('attribute values into [−1,1]:\n', training_set[0:5])

#  Reset the example vector x according its label y
for i in range(training_set.shape[0]):
    if(training_label[i] == 4):
        training_set[i] = -training_set[i]
print('resit\n',training_set[0:5])

#  Find the weight vector w
one =np.ones([training_set.shape[0], 1])
k = np.matmul(training_set.transpose(), training_set)
w = np.matmul(np.dot(np.linalg.inv(k), training_set.transpose()), one)
print('weight','\nshape:\n', w.shape,',\nvalue:',w)


# Predict the example x
test_set = np.float64(x_data[:])
test_label = x_label[:]
test_set = np.insert(test_set, test_set.shape[1], values=1, axis=1)
for i in range(10):
    one_column = test_set[:,i]
    min = np.min(one_column)
    max = np.max(one_column)
    test_set[:,i] = 2*(one_column-min+10**-6)/(max-min+10**-6) - 1
print('test_set  to [-1,1]','\nshape:',test_set.shape,'\npart value:\n',test_set[0:5])
new_label = np.dot(test_set, w)
print('part new label\n',new_label[0:5])
decision_value = []
for i in range(new_label.shape[0]):
    if(new_label[i] >= 0):
        decision_value.append(2)
    else:
        decision_value.append(4)
print('decision_value\n',decision_value[0:10])
# Test stage : Estimate the accuracy
num = np.asarray(decision_value)-test_label
rate = np.sum(num == 0)/test_label.shape[0]
accuracy_rate.append(rate)
print('the accuracy rate is:', accuracy_rate)
