"""
    Find the left most element in the array
"""
from typing import List


def search_left(arr: List[int], target: int) -> int:
    left = 0
    right = len(arr) - 1

    while left < right:
        mid = (left + right) // 2

        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left

def findLeftOptimized(nums: List[int], target: int) -> int:
    low = 0
    hi = len(nums) - 1

    while low <= hi:
        mid = (low + hi) // 2

        if nums[mid] == target:
            if mid == 0 or nums[mid - 1] < target:
                return mid
            hi = mid - 1
        elif nums[mid] < target:
            low = mid + 1
        else:
            hi = mid - 1

    return -1


def findRightOptimized(self, nums: List[int], target: int) -> int:
    lo = 0
    hi = len(nums) - 1
    
    while lo <= hi:
        mid = (lo + hi) // 2

        if nums[mid] == target:
            if mid == len(nums) - 1 or nums[mid + 1] > target:
                return mid
            lo = mid + 1
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    
    return -1