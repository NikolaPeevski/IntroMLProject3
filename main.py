import utils as utils

X, y, attributeNames = utils.loadData()
utils.GMMEM(X, y.squeeze(), attributeNames)