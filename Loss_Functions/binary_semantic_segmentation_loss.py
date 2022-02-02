from tensorflow.keras import backend as K

### dice loss ###
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
###################

### weighted BCE
def weighted_binary_crossentropy(y_true, y_pred):
    class_weights=[1, 5] # should this add up to 1? Should be relative to the percent a class is represented in a dataset?
    class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
    return K.sum(class_loglosses * K.constant(class_weights))
###############

#### regular BCE
def binary_crossentropy(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred)
#####