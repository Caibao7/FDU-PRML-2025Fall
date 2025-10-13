"""
criterion
"""
import math


def _entropy(counts):
    """Shannon entropy computed from count dictionary."""
    total = float(sum(counts.values()))
    if total == 0.0:
        return 0.0
    entropy = 0.0
    for cnt in counts.values():
        if cnt == 0:
            continue
        p = cnt / total
        entropy -= p * math.log2(p)
    return entropy


def _gini(counts):
    """Gini impurity computed from count dictionary."""
    total = float(sum(counts.values()))
    if total == 0.0:
        return 0.0
    g = 1.0
    for cnt in counts.values():
        p = cnt / total
        g -= p * p
    return g


def _error_rate(counts):
    """Classification error (1 - majority class ratio) computed from count dictionary."""
    total = float(sum(counts.values()))
    if total == 0.0:
        return 0.0
    majority = max(counts.values())
    return 1.0 - majority / total

def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels


def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    info_gain = 0.0
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain if splitting y into      #
    # l_y and r_y                                                             #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    total = float(len(y))
    left_total = float(len(l_y))
    right_total = float(len(r_y))

    parent_entropy = _entropy(all_labels)
    weighted_children = 0.0
    if total > 0.0:
        if left_total > 0.0:
            weighted_children += (left_total / total) * _entropy(left_labels)
        if right_total > 0.0:
            weighted_children += (right_total / total) * _entropy(right_labels)
    info_gain = parent_entropy - weighted_children
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain ratio if splitting y     #
    # into l_y and r_y                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    total = float(len(y))
    if total == 0.0:
        return 0.0

    p_l = len(l_y) / total
    p_r = len(r_y) / total
    split_info = 0.0
    if p_l > 0:
        split_info -= p_l * math.log2(p_l)
    if p_r > 0:
        split_info -= p_r * math.log2(p_r)
    if split_info <= 0.0:
        return 0.0
    info_gain = info_gain / split_info
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return info_gain


def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the gini index value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    total = float(len(y))
    left_total = float(len(l_y))
    right_total = float(len(r_y))

    before = _gini(all_labels)
    after = 0.0
    if total > 0.0:
        if left_total > 0.0:
            after += (left_total / total) * _gini(left_labels)
        if right_total > 0.0:
            after += (right_total / total) * _gini(right_labels)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after


def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the error rate value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    total = float(len(y))
    left_total = float(len(l_y))
    right_total = float(len(r_y))

    before = _error_rate(all_labels)
    after = 0.0
    if total > 0.0:
        if left_total > 0.0:
            after += (left_total / total) * _error_rate(left_labels)
        if right_total > 0.0:
            after += (right_total / total) * _error_rate(right_labels)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after
