def calc_impurity(arr):
    if not arr: return 0
    targets = [x[1] for x in arr]
    pred = 0 if sum(targets) / len(targets) < 0.5 else 1
    return sum(abs(pred - y) for y in targets)


def getBest(arr):
    # Базовый случай: массив пуст или из 1 элемента → дальше не делим
    if not arr or len(arr) <= 1:
        return None, arr, [], calc_impurity(arr)

    best = [None, [], [], float('inf')]
    for f in sorted(set(x[0] for x in arr)):
        l = [x for x in arr if x[0] < f]
        r = [x for x in arr if x[0] >= f]

        # ЛОГИКА ПРОПУСКОВ: пропускаем, если одна из веток пуста
        if not l or not r:
            continue

        imp = calc_impurity(l) + calc_impurity(r)
        if imp < best[3]:
            best = [f, l, r, imp]

    # Если ни один порог не дал валидного разбиения → текущий массив остаётся листом
    if best[3] == float('inf'):
        return None, arr, [], calc_impurity(arr)
    return best


def countMetric(arr):
    # Этаж 1
    _, l1, r1, _ = getBest(arr)
    if not l1 and not r1: l1, r1 = arr, []

    # Этаж 2 (левая и правая ветки)
    _, l2, r2, m2 = getBest(l1)
    _, l3, r3, m3 = getBest(r1)

    # Ровно 4 массива листьев (если ветка не разделилась, правый/левый будет [])
    leaves = [l2, r2, l3, r3]
    return m2 + m3, leaves


# ====================
# Тест
# ====================
res = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
arr = [[i + 1, res[i]] for i in range(len(res))]

metric, leaves = countMetric(arr)
print(f"Метрика: {metric}")
for i, leaf in enumerate(leaves, 1):
    print(f"Массив {i}: {leaf}")