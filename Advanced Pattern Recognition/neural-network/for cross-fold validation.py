import numpy as np
import random
import math
def read_data():
    # Read file
    random.seed(17)
    f = open('./breast-cancer-wisconsin.data', 'r+')
    dataset = f.readlines()
    f.close()

    # random.shuffle(dataset)

    # Separate string data according to commas
    for i in range(len(dataset)):
        dataset[i] = dataset[i].split(',')

    # Convert string data to numeric data
    data = []
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            data.append(int(dataset[i][j]))
    # Rearrange to original dimension
    new_data = np.asarray(data).reshape(len(dataset), 11)

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

    whole_data = np.array([data_set0,data_set1,data_set2,data_set3,data_set4])
    whole_label = np.array([data_label0,data_label1,data_label2,data_label3,data_label4])

    return whole_data,whole_label

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def forward_propagate(X, theta1, theta2):
    a = np.dot(X, theta1.T)
    z = np.tanh(a)
    yk = np.dot(z, theta2.T)
    return a, z, yk

def cost( h, y):
    m = h.shape[0]
    J = 0
    for i in range(m):  # Iterate through each sample
        J += (np.sum(h[i,:] - y[i,:]))**2
    J = 0.5*J
    return J

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def tanh_gradient(z):
    return 1 - np.dot(z.reshape((4,1)), z.reshape((1,4)))

def scale(training_set):
    # method 1
    max_i = np.max(training_set, axis=0)
    min_i = np.min(training_set, axis=0)
    y = 2 * (training_set - min_i + 10 ** -6) / (max_i - min_i + 10 ** -6) - 1
    return  y

def backprop(params, input_size, hidden_size, num_labels, X,y_onehot,learning_rate):
    m = X.shape[0]
    params_old = params
    theta1 = np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1)))
    theta2 = np.reshape(params[hidden_size * (input_size + 1):], (num_labels, hidden_size))

    a,z,yk  = forward_propagate(X, theta1, theta2)

    J1 = 0
    w1 = np.zeros(theta1.shape)  # (25, 401)
    w2 = np.zeros(theta2.shape)  # (10, 26)

    for i in range(m):  # Iterate through each sample
        J1 += np.power((np.sum(yk[i,:] - y_onehot[i,:])),2)
    J1 = 0.5*J1

    # STEP7：实现反向传播（这里用到的公式请参考原版作业PDF的第5页）
    for t in range(m):
        xt = np.array(X[t,:])
        at = np.array(a[t, :])  # (1, 401)
        zt = np.asarray(z[t, :])  # (1, 25)
        ykt = np.array(yk[t, :]) # (1, 10)
        y_onehott = np.array(y_onehot[t,:])


        dyt = ykt - y_onehott
        dat = np.dot(np.dot(dyt,theta2), tanh_gradient(zt))
        w1 = w1 - learning_rate*(np.dot(dat.reshape((hidden_size,1)),xt.reshape((1,input_size+1))))
        w2 = w2 - learning_rate*(np.dot(dyt.reshape((num_labels,1)),zt.reshape((1,hidden_size))))
    w = np.concatenate((np.ravel(w1), np.ravel(w2)))

    theta1 = np.reshape(w[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1)))
    theta2 = np.reshape(w[hidden_size * (input_size + 1):], (num_labels, hidden_size))

    a,z,yk  = forward_propagate(X, theta1, theta2)

    J2 = 0
    for i in range(m):
        J2 += np.power((np.sum(yk[i,:] - y_onehot[i,:])),2)
    J2 = 0.5*J2
    return J1,J2,params_old,w

# Initialize settings
input_size = 9
hidden_size = 4
num_labels = 2
learning_rate = 0.1

# Randomly initialize the parameter array of the full network parameter size
w_range = (6/(input_size + 1 + num_labels))**0.5
size = hidden_size * (input_size + 1) + num_labels * hidden_size
params = np.random.uniform(-w_range,w_range,size)

weight = []
best_weight_index = []
index = 0
acc1 = []
acc2 = []
acc3 = []
acc4 = []
acc5 = []
acc = [acc1,acc2,acc3,acc4,acc5]
# method 1
for p in range(5):
    str = [0, 1, 2, 3, 4]
    str.remove(p)
    whole_data, whole_label = read_data()
    train_set = np.vstack((whole_data[str[0]], whole_data[str[1]], whole_data[str[2]], whole_data[str[3]]))[:]
    train_label = np.hstack((whole_label[str[0]], whole_label[str[1]], whole_label[str[2]], whole_label[str[3]]))[:]
    test_set = whole_data[p][:]
    test_label = whole_label[p][:]
    X_train = np.insert(train_set, train_set.shape[1] - 1, values=np.ones(train_set.shape[0]), axis=1)
    X_train = scale(X_train)
    X_test = np.insert(test_set, test_set.shape[1] - 1, values=np.ones(test_set.shape[0]), axis=1)
    X_test = scale(X_test)
    y_0 = np.array([1 if label == 2 else 0 for label in train_label])
    y_0 = np.reshape(y_0, (len(train_label), 1))
    y_1 = np.array([1 if label == 4 else 0 for label in train_label])
    y_1 = np.reshape(y_1, (len(train_label), 1))
    y_onehot = np.hstack((y_0, y_1))
    y_test = np.array([1 if label == 2 else 0 for label in test_label])
    y_test = np.reshape(y_test, (len(test_label), 1))

    for i in range(100):
        J1,J2,w,w1 = backprop(params, input_size, hidden_size, num_labels, X_train, y_onehot, learning_rate=0.001)
        params = w1
        weight.append(w)
        theta1 = np.reshape(w[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1)))
        theta2 = np.reshape(w[hidden_size * (input_size + 1):], (num_labels, hidden_size))
        a2, z2, yk2 = forward_propagate(X_test, theta1, theta2)
        y_pred = np.array(np.argmax(yk2, axis=1))
        correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y_test)]
        accuracy = (sum(map(int, correct)) / float(len(correct)))
        acc[p].append(accuracy)
        index +=1

        if accuracy > 0.95:
            best_weight_index.append(index)

for p in range(5):
    print('accuracy',acc[p])
    print('best accuracy',acc[p][np.argmax(acc[p])])