import sys


def cut(l, p, dp, left, right):
    if right - left == 1:
        return
    minCost = sys.maxsize
    costThisCut = p[right] - p[left]
    for i in range(left + 1, right):
        if dp[left][i] == 0:
            cut(l, p, dp, left, i)
        if dp[i][right] == 0:
            cut(l, p, dp, i, right)
        minCost = min(minCost, costThisCut + dp[left][i] + dp[i][right])
    dp[left][right] = minCost


if __name__ == '__main__':
    l = 10
    p = [1, 3, 6]
    # Add start and end as dummy point
    p = [0, *p, l]
    dp = [[0 for _ in range(len(p))] for _ in range(len(p))]
    cut(l, p, dp, 0, len(p) - 1)
    print(dp[0][len(p)-1])
