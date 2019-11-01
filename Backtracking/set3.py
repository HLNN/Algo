import numpy
import time
import copy


class Back:
    def __init__(self):
        self.count = 0
        # Define mat size
        self.n = 1000
        self.m = 20
        self.setNum = 3
        # Mat random init
        self.mat = numpy.random.random((self.n, self.m))
        self.threshold = 0.05
        # 0-1 rate
        for i in range(len(self.mat)):
            for j in range(len(self.mat[0])):
                self.mat[i][j] = 1 if self.mat[i][j] < self.threshold else 0
        self.maxNum = 0
        self.maxSet = [[] for _ in range(self.setNum)]

    def check(self, j, s):
        for i in range(len(self.mat)):
            if self.mat[i][j]:
                for x in s:
                    if self.mat[i][x]:
                        return False
        return True

    def back(self, level, sets, none):
        # print(level, sets, none)
        self.count += 1
        # A, B not empty and len(a) + len(b) > self.maxNum
        # New best, record it
        if len(sets) and all(len(s) for s in sets) and sum(len(s) for s in sets) > self.maxNum:
            self.maxNum = sum(len(s) for s in sets)
            self.maxSet = sets
            print(level, sets, none)
        # No better solution
        if self.maxNum == self.m or level == self.m or self.m - level + sum(len(s) for s in sets) < self.maxNum:
            return

        # Extend three sons node if ok
        for i in range(self.setNum):
            flag = True
            for j in range(self.setNum):
                if i != j and not self.check(level, sets[j]):
                    flag = False
                    break
            if flag:
                copySets = copy.deepcopy(sets)
                copySets[i].append(level)
                self.back(level + 1, copySets, none)

        self.back(level + 1, sets, [*none, level])


if __name__ == '__main__':
    back = Back()
    t = time.time()
    back.back(0, [[] for _ in range(back.setNum)], [])
    print(time.time() - t)
    print(back.maxNum, back.maxSet, back.count)
