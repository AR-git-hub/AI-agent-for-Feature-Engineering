def getBest(arr):
    arr.sort()
    res = [0, [], [], 10**9]
    for feature in [i[0] for i in arr]:
        lfeatures = [i for i in arr if i[0] < feature]
        rfeatures = [i for i in arr if i[0] >= feature]

        def help(arr):
            mean = sum([i[1] for i in arr]) / len(arr)
            mean = 0 if mean < 0.5 else 1
            std = [abs(mean - i[1]) for i in arr]
            return mean, std

        lMean, lStd = help(lfeatures)
        rMean, rStd = help(rfeatures)
        if res[3] > lStd + rStd:
            res = [feature, lfeatures, rfeatures, lStd + rStd]
    return res

def countMetric(arr):
    cond1, l1, r1, m1 = getBest(arr)
    cond2, l2, r2, m2 = getBest(l1)
    cond2, l3, r3, m3 = getBest(r1)
    return m2 + m3


