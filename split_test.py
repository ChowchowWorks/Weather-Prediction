def split_80_20(data):
    split_idx = int(0.8 * len(data))
    return data[:split_idx], data[split_idx:]

def split_100_point_block(data):
    split_idx = 100
    return data[:split_idx], data[split_idx:]

def split_n_backtracking(data, N):
    split_idx = N * 4 
    return data[:split_idx], data[split_idx:]
