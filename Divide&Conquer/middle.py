def middle(nums, sumLeft, sumRight):
    leftList = []
    rightList = []
    lSum = 0
    rSum = 0

    # Using nums[0] to divide nums
    x = nums[0]
    for i in range(1, len(nums)):
        if nums[i] < x:
            leftList.append(nums[i])
            lSum += nums[i]
        else:
            rightList.append(nums[i])
            rSum += nums[i]

    if sumLeft + lSum <= 0.5 and sumRight + rSum <= 0.5:
        # If both smaller than 0.5, nums[0] is what we find
        return nums[0]
    else:
        # If not, redo with the big side
        if sumLeft + lSum > 0.5:
            return middle(leftList, sumLeft, sumRight + rSum + nums[0])
        else:
            return middle(rightList, sumLeft + lSum + nums[0], sumRight)


if __name__ == '__main__':
    nums = [0.1, 0.2, 0.3, 0.4]
    if sum(nums) == 1:
        print(middle(nums, 0, 0))
    else:
        print('input wrong: ', sum(nums))
