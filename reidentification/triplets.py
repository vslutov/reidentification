from keras import backend as K
import numpy as np

def triplet_loss(y_true, y_pred):

    anchors, positive_items, negative_items = y_pred[0], y_pred[1], y_pred[2]
    return K.mean((anchors - positive_items) ** 2 - (anchors - negative_items) ** 2)

