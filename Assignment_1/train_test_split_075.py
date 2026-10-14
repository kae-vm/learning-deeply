import numpy as np

np.random.seed(99)
images = np.load("data/075_images.npy")
labels = np.load("data/075_labels.npy")
n = labels.shape[0]
indices = np.arange(n)
np.random.shuffle(indices)

train_ind = indices[:int(0.8*n)]
valid_ind = indices[int(0.8*n):int(0.9*n)]
test_ind = indices[int(0.9*n):]

X_train = images[train_ind]
X_valid = images[valid_ind]
X_test = images[test_ind]

y_train = labels[train_ind]
y_valid = labels[valid_ind]
y_test = labels[test_ind]

np.save("data/075_images_train.npy", X_train)
np.save("data/075_images_valid.npy", X_valid)
np.save("data/075_images_test.npy", X_test)
np.save("data/075_labels_train.npy", y_train)
np.save("data/075_labels_valid.npy", y_valid)
np.save("data/075_labels_test.npy", y_test)
