import numpy as np

a = np.array([[[1,3], [2,3]], 
                [[1,3], [2,3]],
                [[1,3], [2,3]]])

# new_a = np.array()

# for i in range(len(a)):
#     np.append(new_a[i], a[i].flatten())

# print(a.shape)


# print(len(a.shape))
# print(a)

def flatten_elements(array):
    shape = array.shape
    new_shape = (shape[0], np.product(shape[1:]))
    return array.reshape(new_shape)

print(flatten_elements(a).shape,
        flatten_elements(a))