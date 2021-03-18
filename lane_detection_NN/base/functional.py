import tensorflow as tf
SMOOTH = 1e-5


# ----------------------------------------------------------------
#   Helpers
# ----------------------------------------------------------------

def _gather_channels(x, indexes):
    """Slice tensor along channels axis by given indexes"""
    if tf.keras.backend.image_data_format() == 'channels_last':
        x = tf.keras.backend.permute_dimensions(x, (3, 0, 1, 2))
        x = tf.keras.backend.gather(x, indexes)
        x = tf.keras.backend.permute_dimensions(x, (1, 2, 3, 0))
    else:
        x = tf.keras.backend.permute_dimensions(x, (1, 0, 2, 3))
        x = tf.keras.backend.gather(x, indexes)
        x = tf.keras.backend.permute_dimensions(x, (1, 0, 2, 3))
    return x


def get_reduce_axes(per_image):
    axes = [1, 2] if tf.keras.backend.image_data_format() == 'channels_last' else [2, 3]
    if not per_image:
        axes.insert(0, 0)
    return axes


def gather_channels(*xs, indexes=None):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes) for x in xs]
    return xs


def round_if_needed(x, threshold):
    if threshold is not None:
        x = tf.keras.backend.greater(x, threshold)
        x = tf.keras.backend.cast(x, tf.keras.backend.floatx())
    return x

def oposite_round_if_needed(x, threshold):
    if threshold is not None:
        x = tf.keras.backend.less_equal(x, threshold)
        x = tf.keras.backend.cast(x, tf.keras.backend.floatx())
    return x

def average(x, per_image=True, class_weights=None):
    if per_image:
        x = tf.keras.backend.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return tf.keras.backend.mean(x)


# ----------------------------------------------------------------
#   Metric Functions
# ----------------------------------------------------------------

def iou_score(gt, pr, class_weights=1., class_indexes=None, smooth=SMOOTH, per_image=True, threshold=None):
    r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
    (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
    similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
    and is defined as the size of the intersection divided by the size of the union of the sample sets:

    .. math:: J(A, B) = \frac{A \cap B}{A \cup B}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round

    Returns:
        IoU/Jaccard score in range [0, 1]

    .. _`Jaccard index`: https://en.wikipedia.org/wiki/Jaccard_index

    """
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    # score calculation
    intersection = tf.keras.backend.sum(gt * pr, axis=axes)
    union = tf.keras.backend.sum(gt + pr, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)
    score = average(score, per_image, class_weights)

    return score


def f_score(gt, pr, beta=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=True, threshold=None):
    r"""The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,
    where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.
    The formula for the F score is:

    .. math:: F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    The formula in terms of *Type I* and *Type II* errors:

    .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}


    where:
        TP - true positive;
        FP - false positive;
        FN - false negative;

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        beta: f-score coefficient
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round

    Returns:
        F-score in range [0, 1]

    """
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    # calculate score
    tp = tf.keras.backend.sum(gt * pr, axis=axes)
    fp = tf.keras.backend.sum(pr, axis=axes) - tp
    fn = tf.keras.backend.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = average(score, per_image, class_weights)

    return score


def precision(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=True, threshold=None):
    r"""Calculate precision between the ground truth (gt) and the prediction (pr).

    .. math:: F_\beta(tp, fp) = \frac{tp} {(tp + fp)}

    where:
         - tp - true positives;
         - fp - false positives;

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: Float value to avoid division by zero.
        per_image: If ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch.
        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.
        name: Optional string, if ``None`` default ``precision`` name is used.

    Returns:
        float: precision score
    """
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    # score calculation
    tp = tf.keras.backend.sum(gt * pr, axis=axes)
    fp = tf.keras.backend.sum(pr, axis=axes) - tp
    
    score = (tp + smooth) / (tp + fp + smooth)
    score = average(score, per_image, class_weights)

    return score


def recall(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=True, threshold=None):
    r"""Calculate recall between the ground truth (gt) and the prediction (pr).

    .. math:: F_\beta(tp, fn) = \frac{tp} {(tp + fn)}

    where:
         - tp - true positives;
         - fp - false positives;

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: Float value to avoid division by zero.
        per_image: If ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch.
        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.
        name: Optional string, if ``None`` default ``precision`` name is used.

    Returns:
        float: recall score
    """
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    tp = tf.keras.backend.sum(gt * pr, axis=axes)
    fn = tf.keras.backend.sum(gt, axis=axes) - tp

    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, per_image, class_weights)

    return score


# ----------------------------------------------------------------
#   Loss Functions
# ----------------------------------------------------------------

def categorical_crossentropy(gt, pr, class_weights=1., class_indexes=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)

    # scale predictions so that the class probas of each sample sum to 1
    axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    pr /= tf.keras.backend.sum(pr, axis=axis, keepdims=True)

    # clip to prevent NaN's and Inf's
    pr = tf.keras.backend.clip(pr, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    # calculate loss
    output = gt * tf.keras.backend.log(pr) * class_weights
    return - tf.keras.backend.mean(output)


def binary_crossentropy(gt, pr):
    return tf.keras.backend.mean(tf.keras.backend.binary_crossentropy(gt, pr))


def categorical_focal_loss(gt, pr, gamma=2.0, alpha=0.25, class_indexes=None):
    r"""Implementation of Focal Loss from the paper in multiclass classification

    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

    """
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)

    # clip to prevent NaN's and Inf's
    pr = tf.keras.backend.clip(pr, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

    # Calculate focal loss
    loss = - gt * (alpha * tf.keras.backend.pow((1 - pr), gamma) * tf.keras.backend.log(pr))

    return tf.keras.backend.mean(loss)


def binary_focal_loss(gt, pr, gamma=2.0, alpha=0.25):
    r"""Implementation of Focal Loss from the paper in binary classification

    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr) - (1 - gt) * alpha * (pr^gamma) * log(1 - pr)

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0

    """
    # clip to prevent NaN's and Inf's
    pr = tf.keras.backend.clip(pr, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

    loss_1 = - gt * (alpha * tf.keras.backend.pow((1 - pr), gamma) * tf.keras.backend.log(pr))
    loss_0 = - (1 - gt) * ((1 - alpha) * tf.keras.backend.pow((pr), gamma) * tf.keras.backend.log(1 - pr))
    loss = tf.keras.backend.mean(loss_0 + loss_1)
    return loss


def focal_tversky_loss(gt, pr, alpha=0.3, beta=0.7, gamma=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=True, threshold=None):
    r"""The Tversky-score (Tversky index) can be seen as a generalization of Dice's coefficient and Tanimoto coefficient (aka Jaccard index),
    where an Tversky-score reaches its best value at 1 and worst score at 0.
    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.

    Args: 
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        alpha, beta: parameter of the Tversky index
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round

    Returns:
        Tversky-score in range [0, 1]

    """
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    # calculate score
    tp = tf.keras.backend.sum(gt * pr, axis=axes)
    fp = tf.keras.backend.sum(pr, axis=axes) - tp
    fn = tf.keras.backend.sum(gt, axis=axes) - tp

    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    
    if per_image:
        score = tf.keras.backend.mean(score, axis=0)

    loss = tf.pow(1 - score, 1/gamma)

    if class_weights is not None:
        loss = loss * class_weights
    loss = tf.keras.backend.mean(loss)

    return loss


import numpy as np
from scipy.ndimage import distance_transform_edt as distance


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res

def calc_dist_map_batch(y_true):
    y_true_numpy = np.array(y_true)
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)

def surface_loss(y_true, y_pred):
    y_true_dist_map = tf.py_func(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return tf.keras.backend.mean(multipled)