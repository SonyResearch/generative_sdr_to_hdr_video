def clamp(num, min_value, max_value):
    if isinstance(num, float):
        if num < min_value:
            return min_value
        elif num > max_value:
            return max_value
    else:
        num[num < min_value] = min_value
        num[num > max_value] = max_value
    return num
