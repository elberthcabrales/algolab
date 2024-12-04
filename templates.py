from collections import defaultdict, deque
import heapq
from math import inf
from typing import List, Optional

class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
        

class templates:
    """
        Two points template
        Where to use it?
        125. Valid Palindrome
        344. Reverse String
        392. Is Subsequence
        15. 3Sum
    """
    def two_pointers(self, arr):
        n = len(arr)
        left, right = 0, n - 1
        while left < right:
            # do something
            if arr[left] != arr[right]:
                return False
            left += 1
            right -= 1

        return True


    """
        Two points template with two parameters
    """
    def two_pointers_two_params(self, arr1, arr2):
        i = 0
        j = 0
        result = []

        # Merge elements from both arrays
        while i < len(arr1) and j < len(arr2):
            if arr1[i] < arr2[j]:
                result.append(arr1[i])
                i += 1
            else:
                result.append(arr2[j])
                j += 1

        # Add remaining elements from arr1
        while i < len(arr1):
            result.append(arr1[i])
            i += 1

        # Add remaining elements from arr2
        while j < len(arr2):
            result.append(arr2[j])
            j += 1

        return result

    """
        Sliding window template
        Where to use it?
        3. Longest Substring Without Repeating Characters
    """
    def sliding_window(self, arr):
        left = ans = curr = 0

        for right in range(len(arr)):
            # do logic here to add arr[right] to curr

            while WINDOW_CONDITION_BROKEN:
                # remove arr[left] from curr
                left += 1

            # update ans
        
        return ans

    """
        Build a prefix sum array
        Where to use it?
        303. Range Sum Query - Immutable
    """
    def answer_queries(self, nums: List[int], queries: List[List[int]], limit: int) -> List[bool]:
        n = len(nums)
        # use pĺus 1, to manage the range to remove without the reached by range
        # for example [2,5] the subtraction decrease the number before 2
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]

        ans = []

        for x, y in queries:
            curr = prefix[y + 1] - prefix[x]
            ans.append(curr < limit)

        return ans

    """
        Efficient string building
    """
    def efficient_string_building(self, arr: List[str]) -> str:
        ans = []
        for c in arr:
            ans.append(c)
        
        return "".join(ans)

    """
        Linked list: fast and slow pointer
    """
    def linked_list(self, head):
        fast = slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        return slow

    """
        Reversing a linked list
        where to use it?
        206. Reverse Linked List
    """
    def reverse_linked_list(self, head):
        prev = None
        curr = head
        while curr:
            next_temp = curr.next
            curr.next = prev
            prev = curr
            curr = next_temp

        return prev
    """
        Find number of subarrays that fit an exact criteria
        Where to use it?
        560. Subarray Sum Equals K
    """
    def subarray_sum_equals_k(self, nums, k):
        count = 0
        sum = 0
        sum_map = {0: 1}
        for num in nums:
            sum += num
            count += sum_map.get(sum - k, 0)
            sum_map[sum] = sum_map.get(sum, 0) + 1

        return count
    """
        Monotonic increasing stack
    """
    def monotonic_increasing_stack(self, arr):
        stack = []
        for i in range(len(arr)):
            while stack and arr[i] < stack[-1]:
                stack.pop()
            stack.append(arr[i])

        return stack
    """
        Binary tree: DFS (recursive)
        1448. Count Good Nodes in Binary Tree

    """
    def binary_tree_dfs_recursive(self, root):
        if not root:
            return
        
        ans = 0

        # do logic
        self.binary_tree_dfs_recursive(root.left)
        self.binary_tree_dfs_recursive(root.right)
        return ans
    """
        Binary tree: DFS (iterative)
    """
    def binary_tree_dfs_iterative(self, root):
        stack = [root]
        while stack:
            node = stack.pop()
            # do logic
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
    """
        Binary tree: BFS
        where to use it?
        199. Binary Tree Right Side View
        1302. Deepest Leaves Sum
        102. Binary Tree Level Order Traversal
    """
    def binary_tree_bfs(self, root):
        queue = deque([root])
        ans = 0

        while queue:
            current_length = len(queue)

            for _ in range(current_length):
                node = queue.popleft()
                # do logic
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            # do logic for current level

        return ans

    """
        Graph: DFS (recursive)
        200. Number of Islands
        547. Number of Provinces
        
    """
    def graph_dfs_recursive(self, graph):
        seen = {0}

        def dfs(node):
            ans = 0
            # do some logic
            for neighbor in graph[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    ans += dfs(neighbor)
            
            return ans

        return dfs(0)

    """
        Graph: DFS (iterative)
        841. Keys and Rooms
        
    """

    def graph_dfs_iterative(self,graph):
        stack = [START_NODE]
        seen = {START_NODE}
        ans = 0

        while stack:
            node = stack.pop()
            # do some logic
            for neighbor in graph[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        
        return ans
    """
        Graph: BFS
        994. Rotting Oranges
        1136. Parallel Courses
    """
    def graph_bfs(self, graph):
        queue = deque([START_NODE])
        seen = {START_NODE}
        ans = 0

        while queue:
            current_length = len(queue)
            # do some logic for current level
            for _ in range(current_length):
                node = queue.popleft()
                # do some logic
                for neighbor in graph[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)

        return ans

    """
        graph topological sort
        1557. Minimum Number of Vertices to Reach All Nodes
        207. Course Schedule
        210. Course Schedule II
    """
    def topological_sort(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        indegree = [0] * numCourses
        graph = defaultdict(list)
        queue = deque()

        for a, b in prerequisites:
            graph[b].append(a)
            indegree[a] += 1
        
        for i in range(numCourses):
            if indegree[i] == 0:
                queue.append(i)
        

        courses = []
        
        while queue:
            node = queue.pop()
            courses.append(node)

            for next_node in graph[node]:
                indegree[next_node] -= 1
                if indegree[next_node] == 0:
                    queue.append(next_node)
        
        return courses if len(courses) == numCourses else []

    """
        clone graph
        133. Clone Graph
    """
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        """
            use the node as the reference and create a new object as value using class Node
            traverse the node param is an instance of Node and has neighbors
            
            if finds any node add neighbor
        """
        if not node:
            return node

        visited = {} #create map and the key is node
        visited[node] = Node(node.val, [])
        queue = deque([node])

        while queue:
            current_node = queue.popleft()

            for neighbor in current_node.neighbors:
                if neighbor not in visited:
                    visited[neighbor] = Node(neighbor.val, [])
                    queue.append(neighbor)
                visited[current_node].neighbors.append(visited[neighbor])

        
        return visited[node]

    """
        Find top k elements with heap
    """
    def find_top_k_elements(self, arr, k):
        heap = []
        for num in arr:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)

        return heap

    """
        Binary search in sorted array
        704. Binary Search
        278. First Bad Version
    """

    def binary_search(self, arr,  target):
        left = 0
        right = len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                # do something
                return
            if arr[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        
        # left is the insertion point
        return left
    """
        Binary search in 2D matrix
        74. Search a 2D Matrix
        240. Search a 2D Matrix II

        approach:
            1. iterate over the minimum of rows and columns
            2. for each iteration, do a binary search in the row and
            3. implement a binary search with params matrix, target, start, vertical
                if is vertical, iterate over the row
                if is horizontal, iterate over the column
                    if do not find the target, return False
            4. if binary search return True, return True else continue with the next iteration
            
    """
    def binary_search(self, matrix: List[List[int]], start: int, target: int, vertical: bool) -> bool:
        lo = start
        hi = len(matrix) - 1 if vertical else len(matrix[0]) - 1
        if vertical:
            while(lo <= hi):
                mid = (lo + hi) // 2

                if matrix[mid][start] == target:
                    return True
                elif matrix[mid][start] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return False
        else:
            while(lo <= hi):
                mid = (lo + hi) // 2

                if matrix[start][mid] == target:
                    return True
                elif matrix[start][mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            
            return False

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        len_row = len(matrix)
        len_col = len(matrix[0])

        for i in range(min(len_row, len_col)):
            col_found = self.binary_search(matrix, i, target, True)
            row_found = self.binary_search(matrix, i, target, False)

            if col_found or row_found:
                return True

        return False
    
    """
        Binary search: duplicate elements, left-most insertion point
        278. First Bad Version
        852. Peak Index in a Mountain Array (NO DUPLICATES)
    """
    def binary_search_with_repeated_left_most(arr, target):
        left = 0
        right = len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] >= target:
                right = mid
            else:
                left = mid + 1 # the last mid + 1 is the answer

        return left
    """
        Binary search: duplicate elements, right-most insertion point
    """
    def binary_search_with_repeated_right_most(arr, target):
        left = 0
        right = len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] > target:
                right = mid
            else:
                left = mid + 1 # the last mid + 1 is the answer

        return left

    # ------------------- PENDING -------------------
    """
        Binary search: for greedy problems 
        1011. Capacity To Ship Packages Within D Days
    """
    def binary_search_greedy(self, arr):
        def check(x):
            # this function is implemented depending on the problem
            return BOOLEAN

        left = MINIMUM_POSSIBLE_ANSWER
        right = MAXIMUM_POSSIBLE_ANSWER
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        
        return left

    # ------------------- PENDING -------------------
    """
        Binary search: for greedy problems
        IF YOU FOR A MAXIMUM
        410. Split Array Largest Sum
        
    """
    def binary_search_greedy_max(self, arr):
        def check(x):
            # this function is implemented depending on the problem
            return BOOLEAN

        left = MINIMUM_POSSIBLE_ANSWER
        right = MAXIMUM_POSSIBLE_ANSWER
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                left = mid + 1
            else:
                right = mid - 1
        
        return right



    """
        Backtracking
        22. Generate Parentheses
        46. Permutations
        47. Permutations II
        78. Subsets
        131. Palindrome Partitioning
    """
    def backtrack(curr, OTHER_ARGUMENTS...):
        if (BASE_CASE):
            # modify the answer
            return
        
        ans = 0
        for (ITERATE_OVER_INPUT):
            # modify the current state
            ans += backtrack(curr, OTHER_ARGUMENTS...)
            # undo the modification of the current state
        
        return ans

    """
        Dynamic programming: top-down memoization
        198. House Robber
        70. Climbing Stairs
        91. Decode Ways
    """

    def dp_top_down(self, arr):
        def dp(STATE):
            if BASE_CASE:
                return 0
            
            if STATE in memo:
                return memo[STATE]
            
            ans = RECURRENCE_RELATION(STATE)
            memo[STATE] = ans
            return ans

        memo = {}
        return dp(STATE_FOR_WHOLE_INPUT)

    
    """
        Dynamic programming: bottom-up tabulation
        198. House Robber
        70. Climbing Stairs
        91. Decode Ways
        907. Sum of Subarray Minimums (tiene monotonic stack)
    """
    def dp_bottom_up(self, arr):
        dp = [0] * (len(arr) + 1)
        for i in range(1, len(arr) + 1):
            dp[i] = RECURRENCE_RELATION(dp[i - 1])

        return dp[-1]

    """
        Dynamic programming: bottom-up tabulation with 2D array
        5. Longest Palindromic Substring
        64. Minimum Path Sum
        1143. Longest Common Subsequence
    """
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        len_col = len(text1) + 1
        len_row = len(text2) + 1
        dp = [[0] * len_col for _ in range(len_row)]


        for row in range(1, len_row):
            for col in range(1, len_col):
                if text1[col - 1] == text2[row - 1]:
                    dp[row][col] = dp[row - 1][col - 1] + 1
                else:
                    dp[row][col] = max(dp[row][col - 1], dp[row - 1][col])

        return dp[len(text2)][len(text1)]

    # ------------------- PENDING -------------------
    """
        Build a trie
        208. Implement Trie (Prefix Tree)
        14. Longest Common Prefix
    """
    class TrieNode:
        def __init__(self):
            # you can store data at nodes if you wish
            self.data = None
            self.children = {}

        def tri_builder(words):
            root = TrieNode()
            for word in words:
                curr = root
                for c in word:
                    if c not in curr.children:
                        curr.children[c] = TrieNode()
                    curr = curr.children[c]
                # at this point, you have a full word at curr
                # you can perform more logic here to give curr an attribute if you want
            
            return root

    # ------------------- PENDING -------------------
    """
        Dijkstra's algorithm
        3341. Find Minimum Time to Finish All Jobs
        1928. Minimum Cost to Reach Destination in Time
    """

    def dijkstra(graph, start):
        distances = [inf] * n
        distances[start] = 0
        heap = [(0, start)]

        while heap:
            curr_dist, node = heapq.heappop(heap)
            if curr_dist > distances[node]:
                continue
            
            for nei, weight in graph[node]:
                dist = curr_dist + weight
                if dist < distances[nei]:
                    distances[nei] = dist
                    heapq.heappush(heap, (dist, nei))

    # ------------------- PENDING -------------------
    """	    
        Floyd-Warshall's algorithm
        1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance

        This algorithm is used to find the **shortest paths between all pairs of vertices** in a graph.

        difference  
            --Dijkstra's Algorithm--
                Finds the shortest path from a single source to all other vertices.
            --Floyd-Warshall's Algorithm
                Finds the shortest paths between all pairs of vertices.

    """
    def findTheCity(n, edges, distanceThreshold):
        # Inicializar la matriz de distancias
        dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0

        # Cargar las distancias iniciales desde las aristas
        for u, v, w in edges:
            dist[u][v] = w
            dist[v][u] = w

        # Algoritmo de Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        # Encontrar la ciudad con el menor número de vecinos alcanzables
        min_neighbors = n
        best_city = -1
        for i in range(n):
            count = sum(1 for j in range(n) if dist[i][j] <= distanceThreshold)
            if count < min_neighbors or (count == min_neighbors and i > best_city):
                min_neighbors = count
                best_city = i

        return best_city

"""
    Greedy algorithm
    134. Gas Station
    455. Assign Cookies

"""