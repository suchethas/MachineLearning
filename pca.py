import pickle
import numpy as np
from PIL import Image
import os

fi = open('suchetha/numeric_bb.pkl','rb')
f = pickle.load(fi)
#print(len(f)) # 2
#print(len(f[0]), len(f[1])) # 20 62800 => 20 days window, 62,800 labels
#print(len(f[0][1], f[0][1][0])) # 4 62800 => 4 OHLC values, 62,800 samples
fi.close()

data=np.array(f[0])
labels = np.array(f[1])

for p in range(len(labels)):
    if labels[p]=="buy":
        labels[p]=1
    else:
        labels[p]=-1
        
split_1 = int(0.8 * len(f[1])) #50240
train_data = data[:,:,0:split_1]
test_data = data[:,:,split_1:len(f[1])]
train_labels = labels[0:split_1]
test_labels = labels[split_1:len(f[1])]
#print(len(train_data[0][0])+len(test_data[0][0]))

# print(data[:,:,0:2])

#generating training data
# for i in range(split_1):
#     ii = "/img"+str(i)+".png"
#     if f[1][i]=="sell":
#         save_path = os.path.join('h:/suchetha','stock_bb', 'train','sell')
#         if os.path.isdir(save_path) is False:
#             os.makedirs(save_path)
#         plt.imsave(save_path+ii, train_data[:,:,i].T, cmap="gray")
#     else:
#         save_path = os.path.join('h:/suchetha','stock_bb', 'train','buy')
#         if os.path.isdir(save_path) is False:
#             os.makedirs(save_path)
#         plt.imsave(save_path+ii, data[:,:,i].T, cmap="gray")
# #         img = plt.imshow(data[:,:,i].T, cmap="gray", origin="lower")
# #         plt.show()

#generating test data
# for i in range(len(f[1])-split_1):
#     ii = "/img"+str(i)+".png"
#     if f[1][i]=="sell":
#         save_path = os.path.join('h:/suchetha','stock_bb', 'test','sell')
#         if os.path.isdir(save_path) is False:
#             os.makedirs(save_path)
#         plt.imsave(save_path+ii, test_data[:,:,i].T, cmap="gray")
#     else:
#         save_path = os.path.join('h:/suchetha','stock_bb', 'test','buy')
#         if os.path.isdir(save_path) is False:
#             os.makedirs(save_path)
#         plt.imsave(save_path+ii, test_data[:,:,i].T, cmap="gray")









from sklearn import svm, linear_model, grid_search
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
train_data = train_data.reshape(split_1, -1) # shape = 50240, 80


# def svc_param_selection(X, y, nfolds):
#     Cs = [0.001, 0.01, 0.1, 1, 10]
#     gammas = [0.001, 0.01, 0.1, 1]
#     param_grid = {'C': Cs, 'gamma' : gammas}
#     grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
#     grid_search.fit(X, y)
#     grid_search.best_params_
#     return grid_search.best_params_


Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
grid_search.fit(train_data, train_labels)
print(grid_search.best_params_)


# # Create a classifier: a support vector classifier
# classifier = svm.SVC(C=2, gamma=0.1)
# #classifier = LinearSVC()

# # We learn the digits on the first half of the digits
# classifier.fit(train_data, train_labels)




from sklearn import metrics

# Now predict the value of the digit on the second half:
test_data = test_data.reshape(12560, -1)
predicted = classifier.predict(test_data)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predicted))
