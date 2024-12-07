from typing import List


class PrefixSumQueries:
    """
    

    Example 1: Given an integer array nums, an array queries where queries[i] = [x, y] and an integer limit,
    return a boolean array that represents the answer to each query.
    A query is true if the sum of the subarray from x to y is less than limit, or false otherwise.

    For example, given 
    nums = [1, 6, 3, 2, 7, 2],
    queries = [[0, 3], [2, 5], [2, 4]], and limit = 13, the answer is [true, false, true].
    For each query, the subarray sums are [12, 14, 12].

    """
    def __init__(self):
        pass

    def answer_queries(self, nums: List[int], queries: List[List[int]], limit: int) -> List[bool]:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]

        ans = []

        for x, y in queries:
            curr = prefix[y + 1] - prefix[x]
            ans.append(curr < limit)

        return ans

class NumArray:
    """
        nums:[1,6,3,2,7,2]
        prefix_sum:[0,1,7,10,12,19,21]
        
        ranges ([x,y])
        [0,3],[2,5],[2,4]

        ans=[
            12, # prefix_sum[y + 1] - prefix_sum[x] 10 - 0
            14, # 21 - 7 remove all before the range
            12, # 19 - 7
        ]
    """

    def __init__(self, nums: List[int]):
        n = len(nums)
        # use pĺus 1, to manage the range to remove without the reached by range
        # for example [2,5] the subtraction decrease the number before 2
        prefix_sum = [0] * (n + 1)

        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        self.prefix_sum = prefix_sum

    def sum_range(self, left: int, right: int) -> int:
        return self.prefix_sum[right + 1] - self.prefix_sum[left]


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(left,right)