def split_80_20(data):
    split_idx = int(0.8 * len(data))
    return data[:split_idx], data[split_idx:] # First 80% as train, last 20% as test

def split_100_point_block(data):
    split_idx = 100
    return data[:split_idx], data[split_idx:]

def split_n_backtracking(data, N):
    split_idx = N * 4 
    return data[:split_idx], data[split_idx:]

def split_20_80(data):
    split_idx = int(0.2 * len(data))
    return data[split_idx:], data[:split_idx]  # First 20% as test, last 80% as train

def split_50_50_80_20(data):
    mid_idx = len(data) // 2
    first_half_train, first_half_test = split_80_20(data[:mid_idx])
    second_half_train, second_half_test = split_80_20(data[mid_idx:])
    return (first_half_train, first_half_test), (second_half_train, second_half_test)


