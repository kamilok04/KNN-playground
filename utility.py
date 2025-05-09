def compute_distance(obj1, obj2, metric='euclidean'):
    val1 = obj1.obj if hasattr(obj1, 'obj') else obj1
    val2 = obj2.obj if hasattr(obj2, 'obj') else obj2

    if metric == 'manhattan':
        return sum(abs(x - y) for x, y in zip(val1, val2))
    elif metric == 'minkowski':
        p = 3  # tweakable
        return sum(abs(x - y) ** p for x, y in zip(val1, val2)) ** (1 / p)
    else:
        return sum((x - y) ** 2 for x, y in zip(val1, val2)) ** 0.5