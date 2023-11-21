def nim(nums):
    """
        xor = nums[0] ^ nums[1] ^ nums[2]
        如果xor != 0, 假设这一步能把nums[1]变成nums[1] ^ xor [注1]
        那么下一步就会是nums[0] ^ nums[1] ^ xor ^ nums[2] == nums[0] ^ nums[1] ^ nums[2] ^ xor == xor ^ xor == 0
        下一步就成了P-position, 先手必败

        注1:
            k的最高位的1肯定来源自nums的某个元素, 假设该位置为z
            1. 假设元素a在z的二进制表示不是1, 那么a^k把a的z位置设置为了1, 肯定比原a大, 但不可能把a取成复数个元素
            2. 假设元素a在z的二进制表示是1, 那么a^k把a的z位置异或成了0, 肯定比原a小, 是合法的取法
    """
    assert len(nums) == 3
    xor = nums[0] ^ nums[1] ^ nums[2]
    
    if xor == 0:
        return -1, -1 # P-position 先手必败
    else:
        for i in range(3):
            remain = nums[i] ^ xor # 变换最高位
            if remain < nums[i]: # 如果原最高位为1
                return i, nums[i] - remain # 把当前数变成nums[i] ^ xor
    