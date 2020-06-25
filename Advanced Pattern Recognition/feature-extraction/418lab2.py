import numpy as np
import matplotlib.pyplot as plt

# Read file
f = open('./iris.data', 'r+')
dataset = f.readlines()
f.close()

# Convert string data to numeric data and create data sets and label
data = np.zeros((len(dataset)-1,4))
label = []
for  i in range(150):
    k = dataset[i][16:-1]
    label.append(k)
    m = float(dataset[i][0:3])
    n= float(dataset[i][4:7])
    p = float(dataset[i][8:11])
    q = float(dataset[i][12:15])
    data[i] = np.asarray([m, n, p, q])
dataset = data

# PCA
X = data
k = 2 #Compressed to k-dimensional data
n_samples = X.shape[0]
n_features = X.shape[1]
mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
# normalization:Standardize the data
norm_X = X - mean
print('norm_X\n',norm_X)
# Compute the covariance matrix
covariance_matrix = np.dot(np.transpose(norm_X), norm_X)
print('covariance_matrix')
print(covariance_matrix)
# Calculate the eigenvectors and eigenvalues
eig_val, eig_vec = np.linalg.eig(covariance_matrix)
print( eig_vec, eig_val)
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
# sort eig_vec based on eig_val from highest to lowest
eig_pairs.sort(reverse=True)
# select the top k eig_vec
feature = np.array([ele[1] for ele in eig_pairs[:k]])
print('new eig_vec\n',feature)
# get new data
data = np.dot(norm_X, np.transpose(feature))
print('new data\n',data)
print('data_pca',data)
# pca drawing data
y1 = data

# plot the data
# Categorize data based on tags
data1 = np.array([y1[i] for i in range(len(y1)) if label[i] == 'Iris-setosa'])
data2 = np.array([y1[i] for i in range(len(y1)) if label[i] == 'Iris-versicolor'])
data3 = np.array([y1[i] for i in range(len(y1)) if label[i] == 'Iris-virginica'])
# Drawing part
plt.scatter(data1[:,0], data1[:,1], alpha=0.6,marker='x', label='Iris-setosa')
plt.scatter(data2[:,0], data2[:,1], alpha=0.6,marker='.',label='Iris-versicolor')
plt.scatter(data3[:,0], data3[:,1], alpha=0.6,marker='*',label='Iris-virginica')
plt.legend()
plt.title('PCA')
plt.show()


#LDA
data_x = dataset
data_y = label
n_sample = data_x.shape[0]
n_feature = data_x.shape[1]
# Generate {Xk} matrix according to the labels of the examples
data1 = np.array([data_x[i] for i in range(len(data_x)) if data_y[i] == 'Iris-setosa'])
data2 = np.array([data_x[i] for i in range(len(data_x)) if data_y[i] == 'Iris-versicolor'])
data3 = np.array([data_x[i] for i in range(len(data_x)) if data_y[i] == 'Iris-virginica'])
print(' Generate Xk matrix\n',data1.shape,data2.shape,data3.shape)
# Compute the class-wise mean and the total mean of the data matrix
u = np.mean(data_x, axis=0, keepdims=True)
u1 = np.mean(data1, axis=0, keepdims=True)
u2 = np.mean(data2, axis=0, keepdims=True)
u3 = np.mean(data3, axis=0, keepdims=True)
print('class-wise mean and the total mean of the data matrix\n')
print('the first type class-wise mean')
print(u1)
print('the second type class-wise mean')
print(u2)
print('the third type class-wise mean')
print(u3)
print('total mean')
print(u)
# Compute the within-class covariance matrix SW
sw = np.dot(np.transpose(data1 - u1), data1 - u1) + np.dot(np.transpose(data2 - u2), data2 - u2) + np.dot(np.transpose(data3 - u3), data3 - u3)
print('sw')
print(sw)
# Compute the between-class covariance matrix SB
sb = data1.shape[0] * np.dot(np.transpose(u1 - u), u1 - u) + data2.shape[0] * np.dot(np.transpose(u2 - u), u2 - u) + data3.shape[0] * np.dot(np.transpose(u3 - u), u3 - u)
print('sb')
print(sb)
# sw^-1 * sb
C = np.linalg.inv(sw) * sb
# D = P * C * P^T
eig_val, eig_vec = np.linalg.eig(C)
print(eig_val, eig_vec)
# Find projection eigvector using method 2
sorted_idx = np.argsort(eig_val)[::-1]
eig_val_new = eig_val[sorted_idx]
eig_vec_new = eig_vec[:, sorted_idx]
print('eig_vec_new')
print(eig_vec_new)
y = np.dot(data_x,np.transpose(eig_vec))
# lda drawing data
y2 = y[:,0:2]
print('new data after LDA\n',data)
print('data_LDA\n',y2)

# plot the data
# Categorize data based on tags
data1 = np.array([y2[i] for i in range(len(y2)) if data_y[i] == 'Iris-setosa'])
data2 = np.array([y2[i] for i in range(len(y2)) if data_y[i] == 'Iris-versicolor'])
data3 = np.array([y2[i] for i in range(len(y2)) if data_y[i] == 'Iris-virginica'])
# Drawing part
plt.scatter(data1[:,0], data1[:,1], alpha=0.6,marker='x', label='Iris-setosa')
plt.scatter(data2[:,0], data2[:,1], alpha=0.6,marker='.',label='Iris-versicolor')
plt.scatter(data3[:,0], data3[:,1], alpha=0.6,marker='*',label='Iris-virginica')
plt.title('LDA')
plt.legend()
plt.show()