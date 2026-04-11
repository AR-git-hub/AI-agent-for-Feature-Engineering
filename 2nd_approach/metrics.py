def get_impurity(arr):
    if not arr: return 0
    targets = [x[1] for x in arr]
    pred = 0 if sum(targets) / len(targets) < 0.5 else 1
    return sum(abs(pred - y) for y in targets)


def getBest(arr):
    if not arr: return [0, [], [], 0]
    best = [0, [], [], float('inf')]

    for f in sorted(set(x[0] for x in arr)):
        l = [x for x in arr if x[0] < f]
        r = [x for x in arr if x[0] >= f]

        if not l or not r:  # Логика пропусков: пропускаем невалидные разбиения
            continue

        imp = get_impurity(l) + get_impurity(r)
        if imp < best[3]:
            best = [f, l, r, imp]

    if best[3] == float('inf'):  # Если разбить не удалось
        return [0, arr, [], get_impurity(arr)]
    return best


def countMetric(arr):
    _, l1, r1, _ = getBest(arr)
    _, l2, r2, m2 = getBest(l1)
    _, l3, r3, m3 = getBest(r1)

    # Формируем 4 листа нижнего уровня (обрабатываем случаи, когда сплит не удался)
    leaves = [
        l2 if l2 else l1,
        r2 if r2 else [],
        l3 if l3 else r1,
        r3 if r3 else []
    ]
    metric = sum(get_impurity(leaf) for leaf in leaves)
    return metric, leaves


# Тест
res = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
arr = [[i + 1, res[i]] for i in range(12)]

metric, bottom_arrays = countMetric(arr)
print(f"Метрика: {metric}\n")
for i, a in enumerate(bottom_arrays, 1):
    print(f"Массив {i}: {a}")