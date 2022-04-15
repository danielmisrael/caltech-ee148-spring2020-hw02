import os
import numpy as np
import json
from PIL import Image


def get_radial_gradient(size):
    r = Image.radial_gradient(mode='L')
    radial_gradient = np.asarray(r.resize((size, size)))
    while np.sum(radial_gradient) > 0:
        radial_gradient = radial_gradient - np.ones((size, size))
    return radial_gradient

def compute_convolution(I, T, stride=1, padding=0):
    # I and T, stride
    '''
    This function takes an image <I> and a template <T> (both numpy arrays)
    and returns a heatmap where each grid represents the output produced by
    convolution at each location. You can add optional parameters (e.g. stride,
    window_size, padding) to create additional functionality.
    '''
    (n_rows,n_cols) = np.shape(I)

    x = int((n_rows - T.shape[1] + 2 * padding) / stride + 1)
    y = int((n_cols - T.shape[0] + 2 * padding) / stride + 1)
    res = np.zeros((x, y))

    if padding != 0:
        image = np.zeros((n_rows + 2 * padding, n_cols + 2 * padding))
        image[padding:-padding, padding:-padding] = I
    else:
        image = I


    for i in range(0, n_rows, stride):
        for j in range(0, n_cols, stride):
            subimage = image[i: i + T.shape[0], j: j + T.shape[1]]
            if i < x and j < y and subimage.shape == T.shape:
                res[i, j] = (T * subimage).sum()


    return res
    # Cross Correlation
    # kernel = np.flipud(np.fliplr(kernel))
    #
    # # Gather Shapes of Kernel + Image + Padding
    # xKernShape = kernel.shape[0]
    # yKernShape = kernel.shape[1]
    # xImgShape = image.shape[0]
    # yImgShape = image.shape[1]
    #
    # # Shape of Output Convolution
    # xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    # yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    # output = np.zeros((xOutput, yOutput))
    #
    # # Apply Equal Padding to All Sides
    # if padding != 0:
    #     imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
    #     imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    #     # print(imagePadded)
    # else:
    #     imagePadded = image
    #
    # # Iterate through image
    # for y in range(image.shape[1]):
    #     # Exit Convolution
    #     if y > image.shape[1] - yKernShape:
    #         break
    #     # Only Convolve if y has gone down by the specified Strides
    #     if y % strides == 0:
    #         for x in range(image.shape[0]):
    #             # Go to next row once kernel is out of bounds
    #             if x > image.shape[0] - xKernShape:
    #                 break
    #             try:
    #                 # Only Convolve if x has moved by the specified Strides
    #                 if x % strides == 0:
    #                     output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
    #             except:
    #                 break
    #
    # return output

def remove_similar_locations(locations, confidences):
    min_dist = 75
    centroids = [locations[0]]
    conf = [confidences[0]]
    for loc, confidence in zip(locations, confidences):
        is_new_centroid = True
        for c in centroids:
            if abs(loc[0] - c[0]) + abs(loc[1] - c[1]) < min_dist:
                is_new_centroid = False
        if is_new_centroid:
            centroids.append(loc)
            conf.append(confidence)
    return centroids, conf

def get_locations(feature_map):
    top_k = 20
    max_val = np.partition(feature_map.flatten(), -top_k)[-top_k]
    # print("max_val", max_val)
    if max_val <= 100:
        return [], []
    feature_map[feature_map < max_val] = 0
    locs = (feature_map > 0).nonzero()
    # print("locs", locs)
    confidence = feature_map[feature_map >= max_val] / np.max(feature_map)
    return remove_similar_locations(list(zip(locs[0], locs[1])), confidence)

def get_bounding_boxes(locations, confidence):
    if not locations:
        return []
    box_height = 20
    box_width = 20
    bounding_boxes = []
    for i, (x, y) in enumerate(locations):
        tl_row = x - box_height / 2
        tl_col = y - box_width / 2
        br_row = x + box_height / 2
        br_col = y + box_width / 2
        bounding_boxes.append([tl_col, tl_row, br_col, br_row, confidence[i]])
        # print(confidence[i])
    return bounding_boxes

def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''

    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    # box_height = 8
    # box_width = 6
    #
    # num_boxes = np.random.randint(1,5)
    #
    # for i in range(num_boxes):
    #     (n_rows,n_cols,n_channels) = np.shape(I)
    #
    #     tl_row = np.random.randint(n_rows - box_height)
    #     tl_col = np.random.randint(n_cols - box_width)
    #     br_row = tl_row + box_height
    #     br_col = tl_col + box_width
    #
    #     score = np.random.random()
    #
    #     output.append([tl_row,tl_col,br_row,br_col, score])
    #
    # '''
    # END YOUR CODE
    # '''

    locs, confidence = get_locations(heatmap)
    output = get_bounding_boxes(locs, confidence)

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>.
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>.
    The first four entries are four integers specifying a bounding box
    (the row and column index of the top left corner and the row and column
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    # template_height = 8
    # template_width = 6

    # You may use multiple stages and combine the results
    # T = np.random.random((template_height, template_width))

    kernel = get_radial_gradient(10)
    blur_kernel = np.ones((5,5)) * (1 / 25)

    s = np.sum(I, axis=2)
    s[s==0] = 1
    color_map = (I[:, :, 0] / s) ** 2
    color_map = color_map / np.max(color_map) * 255

    feature_map = compute_convolution(color_map, kernel)
    feature_map = feature_map / np.max(feature_map) * 250
    # plt.imshow(feature_map)
    feature_map = compute_convolution(feature_map, blur_kernel)
    # plt.imshow(feature_map)
    # plt.show()

    # heatmap = compute_convolution(I, T)
    output = predict_boxes(feature_map)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    # print(output)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):
    if i % 10 == 0:
        length = len(file_names_train)
        print(f"Finding stop lights in image {i}/{length}")

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set.
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
