from scipy.io import loadmat

def loadData():
    """Gets the data from a .mat file.
        X, y, attributeNames
    """

    mat_data = loadmat('./KidneyData.mat')
    attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
    X = mat_data['X']
    y = mat_data['y']

    return X, y, attributeNames
    