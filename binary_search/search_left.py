"""
852. Peak Index in a Mountain Array

You are given an integer mountain array arr of length n where the values increase to a peak element and then decrease.

Return the index of the peak element.

Your task is to solve it in O(log(n)) time complexity.

Input: arr = [0,10,5,2]
returns 1

https://leetcode.com/problems/peak-index-in-a-mountain-array/
"""


from typing import List


class MountainPeak:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        left = 0
        right = len(arr) - 1

        while left < right:
            mid = (left + right) // 2

            if arr[mid] > arr[mid + 1]:
                right = mid
            else:
                left = mid + 1
        
        return left

"""
https://leetcode.com/problems/guess-number-higher-or-lower/
374. Guess Number Higher or Lower
We are playing the Guess Game. The game is as follows:

I pick a number from 1 to n. You have to guess which number I picked.

Every time you guess wrong, I will tell you whether the number I picked is higher or lower than your guess.

You call a pre-defined API int guess(int num), which returns three possible results:

-1: Your guess is higher than the number I picked (i.e. num > pick).
1: Your guess is lower than the number I picked (i.e. num < pick).
0: your guess is equal to the number I picked (i.e. num == pick).
Return the number that I picked.
"""

class GuessNumber:
    def guess(self, num):
            if num == 6:
                return 0
            elif num < 6:
                return -1
            else:
                return 1
    def guessNumber(self, n: int) -> int:
        # The guess API is already defined for you.
        lo = 0
        hi = n

        while lo < hi:
            mid = (lo + hi) // 2
            num = self.guess(mid)
            if num == -1 or num == 0:
                hi = mid
            else:
                lo = mid + 1 # the last mid + 1 is the answer
                
        return lo
                