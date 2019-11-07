def middle(nums, left, right, lSum, rSum):
    mid = (left + right) // 2
    lSumNew = sum(n['weight'] for n in nums[left:mid])
    rSumNew = sum(n['weight'] for n in nums[mid + 1:right])
    if lSum + lSumNew <= 0.5 and rSum + rSumNew <= 0.5:
        return nums[mid]['value']
    else:
        if lSum + lSumNew > 0.5:
            return middle(nums, left, mid, lSum, rSum + rSumNew)
        else:
            return middle(nums, mid + 1, right, lSum + lSumNew, rSum)


if __name__ == '__main__':
    def getValue(elem):
        return elem['value']

    nums = [{'value': 1, 'weight': 0.1},
            {'value': 4, 'weight': 0.2},
            {'value': 2, 'weight': 0.3},
            {'value': 5, 'weight': 0.4}]
    nums.sort(key=getValue)
    print(nums)

    if sum(num['weight'] for num in nums) == 1:
        print(middle(nums, 0, len(nums), 0, 0))
    else:
        print('input wrong: ', sum(nums))
