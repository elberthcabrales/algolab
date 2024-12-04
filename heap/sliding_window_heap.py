import heapq
from typing import List

class Solution:
    def slidingWindowWithHeap(self, nums: List[List[int]]) -> List[int]:
        """
        Finds the smallest range covering at least one number from each of the sorted lists.
        
        Args:
            nums (List[List[int]]): A list of k sorted lists.
        
        Returns:
            List[int]: The smallest range [low, high] that covers at least one number from each list.
        """
        # Initialize variables
        ans = [-float('inf'), float('inf')]  # Result range [low, high]
        heap = []  # Min-heap to track the smallest element
        max_value = -float('inf')  # Tracks the max value in the current window
        
        # Push the first element of each list into the heap
        for row_index in range(len(nums)):
            heap.append((nums[row_index][0], row_index, 0))  # (value, row index, column index)
            max_value = max(max_value, nums[row_index][0])
        
        # Heapify to build the min-heap
        heapq.heapify(heap)
        
        # Process the heap until one of the lists is exhausted
        while True:
            # Extract the smallest value from the heap
            min_value, row_index, col_index = heapq.heappop(heap)
            
            # Update the result range if it's smaller than the current range
            if max_value - min_value < ans[1] - ans[0]:
                ans = [min_value, max_value]
            
            # Move to the next element in the current list
            col_index += 1
            
            # If we've reached the end of one list, stop
            if col_index == len(nums[row_index]):
                break
            
            # Push the next element from the current list into the heap
            next_value = nums[row_index][col_index]
            heapq.heappush(heap, (next_value, row_index, col_index))
            
            # Update the max value in the current window
            max_value = max(max_value, next_value)
        
        return ans
