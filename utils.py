

def get_correct_n(y_pre, y):
    correct_n = 0
    for i in range(len(y)):
        if y_pre[i] == y[i]:
            correct_n += 1
    return correct_n

