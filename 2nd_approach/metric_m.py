def calc_metric(arr):
    summ = 0
    for sample in arr:
        summ += sample[1]
    
    return summ / len(arr)


def calc_best_split(arr):
    best_border = 0
    best_metric = float('inf')
    for border in range(1, len(arr)):
        l = arr[:border]
        r = arr[border:]

        l_metric = calc_metric(l)
        r_metric = calc_metric(r)

        l_int_metric = 1 if l_metric >= 0.5 else 0
        r_int_metric = 1 if r_metric >= 0.5 else 0

        l_error = 0
        for i in l:
            l_error += abs(i[1] - l_int_metric)
        
        r_error = 0
        for i in r:
            r_error += abs(i[1] - r_int_metric)
        
        sum_error = l_error + r_error

        if sum_error < best_metric:
            best_metric = sum_error
            best_border = border
    
    return best_border, best_metric


def calculate_metric_m(feature1: list, feature2: list, targert: list):
    feature1_with_target = [(f, t) for f, t in zip(feature1, targert)]
    feature2_with_target = [(f, t) for f, t in zip(feature2, targert)]

    feature1_with_target = sorted(feature1_with_target, key=lambda x: x[0])
    feature2_with_target = sorted(feature2_with_target, key=lambda x: x[0])

    best_border1, metric1 = calc_best_split(feature1_with_target)
    best_border2, metric2 = calc_best_split(feature2_with_target)

    if metric1 < metric2:
        best_feature = feature1_with_target
        best_border = best_border1
    else:
        best_feature = feature2_with_target
        best_border = best_border2
    
    best_feature_l = best_feature[:best_border]
    best_feature_r = best_feature[best_border:]

    best_border2_1, metric2_1 = calc_best_split(best_feature_l)
    best_border2_2, metric2_2 = calc_best_split(best_feature_r)

    total_metric = 1 - (metric2_1 + metric2_2) / len(feature1)

    return total_metric
