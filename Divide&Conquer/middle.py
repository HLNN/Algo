def middle(nums, left, right):
    mid = (left + right) // 2
    if sum(n['weight'] for n in nums[0:mid]) <= 0.5 and sum(n['weight'] for n in nums[mid + 1:len(nums)]) <= 0.5:
        return mid
    else:
        if sum(num['weight'] for num in nums[0:mid]) > 0.5:
            return middle(nums, left, mid)
        else:
            return middle(nums, mid + 1, right)


if __name__ == '__main__':
    def getValue(elem):
        return elem['value']

    nums = [{'value': 1, 'weight': 0.1},
            {'value': 4, 'weight': 0.2},
            {'value': 2, 'weight': 0.3},
            {'value': 5, 'weight': 0.4}]
    nums.sort(key=getValue)

    if sum(num['weight'] for num in nums) == 1:
        print(middle(nums, 0, len(nums)))
    else:
        print('input wrong: ', sum(nums))
