import numpy as np

class SlidingWindow:
    def __init__(self):
        pass
    
    def slidingWindowView(self, array, windowSize):
        tot = []
        window = 0
        arr = []
        index = 0
        while index < len(array):
            # if the window size is less, increase the size, add tot/window to arr
            if (window < windowSize):
                window += 1
                tot.append(array[index])
                index += 1
            else:
                tot.append(array[index])
                tot.pop(0)
                index += 1
            arr.append((np.mean(tot), np.std(tot)))
        return np.array(arr)
    
    def meanStd(self, array, windowSize):
        windows = sliding_window_view(array, windowSize)
        arr = [(np.mean(window), np.std(window)) for window in windows]
        for i in range(windowSize - 1):
            arr.append(arr[-1])
        return arr