import numpy
import time


class Back:
    def __init__(self):
        # Define mat size
        self.n = 1000
        self.m = 20
        # Mat random init
        self.mat = numpy.random.random((self.n, self.m))
        # 0-1 rate
        self.threshold = 0.1
        for i in range(len(self.mat)):
            for j in range(len(self.mat[0])):
                self.mat[i][j] = 1 if self.mat[i][j] < self.threshold else 0
        self.maxNum = 0
        self.maxA = []
        self.maxB = []

    def check(self, j, s):
        for i in range(len(self.mat)):
            if self.mat[i][j]:
                for x in s:
                    if self.mat[i][x]:
                        return False
        return True

    def back(self, level, a, b, none):
        # A, B not empty and len(a) + len(b) > self.maxNum
        # New best, record it
        if len(a) and len(b) and len(a) + len(b) > self.maxNum:
            self.maxNum = len(a) + len(b)
            self.maxA = a
            self.maxB = b
            print(level, a, b, none)
        # No better solution
        if self.maxNum == self.m or level == self.m or len(self.mat[0]) - level + len(a) + len(b) < self.maxNum:
            return

        # Extend three son node if ok
        if self.check(level, b):
            self.back(level + 1, [*a, level], b, none)
        if self.check(level, a):
            self.back(level + 1, a, [*b, level], none)
        if len(self.mat[0]) - level + len(a) + len(b) - 1 < self.maxNum:
            self.back(level + 1, a, b, [*none, level])


if __name__ == '__main__':
    back = Back()
    t = time.time()
    back.back(0, [], [], [])
    print(time.time() - t)
    print(back.maxNum, back.maxA, back.maxB)
