"""
Samples

33. Search in Rotated Sorted Array
"""
from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        """
            if the part in mid is soreted validate normal binary searse else
        """
        lo = 0
        hi = len(nums) - 1

        while lo <= hi:
            mid = (hi + lo) // 2

            if nums[mid] == target:
                return mid

            # is sorted this part
            if nums[lo] <= nums[mid]:
                if nums[lo] <= target and target < nums[mid]:
                    hi = mid - 1
                else:
                    lo = mid + 1
            else:
                if nums[hi] >= target and target > nums[mid]:
                    lo = mid + 1
                else:
                    hi = mid - 1

        return -1
