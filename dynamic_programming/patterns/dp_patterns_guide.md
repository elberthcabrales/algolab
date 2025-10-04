# Dynamic Programming Patterns - Complete Guide

## Table of Contents
1. [Knapsack Pattern](#knapsack-pattern)
2. [Longest Common Subsequence (LCS) Pattern](#lcs-pattern)
3. [Grid/Path Pattern](#grid-path-pattern)
4. [Recursive Number Pattern](#recursive-number-pattern)
5. [Stock Trading Pattern](#stock-trading-pattern)
6. [Interval Scheduling Pattern](#interval-scheduling-pattern)

---

## 1. Knapsack Pattern

**When to use:** Choose items with weights/costs to maximize/minimize value within constraints.

### 0/1 Knapsack (Each item once)

#### Top-Down Template
```python
def knapsack_td(weights, values, capacity):
    n = len(weights)
    memo = {}
    
    def dp(i, remaining_capacity):
        # Base cases
        if i < 0 or remaining_capacity <= 0:
            return 0
            
        if (i, remaining_capacity) in memo:
            return memo[(i, remaining_capacity)]
        
        # Don't take item i
        skip = dp(i - 1, remaining_capacity)
        
        # Take item i (if it fits)
        take = 0
        if weights[i] <= remaining_capacity:
            take = values[i] + dp(i - 1, remaining_capacity - weights[i])
        
        memo[(i, remaining_capacity)] = max(take, skip)
        return memo[(i, remaining_capacity)]
    
    return dp(n - 1, capacity)
```

#### Bottom-Up Template
```python
def knapsack_bu(weights, values, capacity):
    n = len(weights)
    # dp[i][w] = max value using first i items with capacity w
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i-1
            dp[i][w] = dp[i-1][w]
            
            # Take item i-1 (if it fits)
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], values[i-1] + dp[i-1][w - weights[i-1]])
    
    return dp[n][capacity]
```

### Unbounded Knapsack (Unlimited items)

#### Top-Down Template
```python
def unbounded_knapsack_td(weights, values, capacity):
    memo = {}
    
    def dp(i, remaining_capacity):
        if i < 0 or remaining_capacity <= 0:
            return 0
            
        if (i, remaining_capacity) in memo:
            return memo[(i, remaining_capacity)]
        
        # Don't take item i
        skip = dp(i - 1, remaining_capacity)
        
        # Take item i (stay at same index for unlimited use)
        take = 0
        if weights[i] <= remaining_capacity:
            take = values[i] + dp(i, remaining_capacity - weights[i])
        
        memo[(i, remaining_capacity)] = max(take, skip)
        return memo[(i, remaining_capacity)]
    
    return dp(len(weights) - 1, capacity)
```

#### Bottom-Up Template
```python
def unbounded_knapsack_bu(weights, values, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i-1
            dp[i][w] = dp[i-1][w]
            
            # Take item i-1 (can use same item again)
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], values[i-1] + dp[i][w - weights[i-1]])
    
    return dp[n][capacity]
```

### Common Variations:
- **Coin Change** (minimum coins)
- **Partition Equal Subset Sum** (boolean knapsack)
- **Target Sum** (assign +/- to reach target)
- **Ones and Zeroes** (2D knapsack with two constraints)

---

## 2. Longest Common Subsequence (LCS) Pattern

**When to use:** Compare two sequences to find optimal alignment, transformation, or common elements.

### LCS Template

#### Top-Down Template
```python
def lcs_td(text1, text2):
    memo = {}
    
    def dp(i, j):
        # Base cases
        if i < 0 or j < 0:
            return 0
            
        if (i, j) in memo:
            return memo[(i, j)]
        
        if text1[i] == text2[j]:
            # Characters match - include both
            memo[(i, j)] = 1 + dp(i - 1, j - 1)
        else:
            # Characters don't match - try skipping either
            memo[(i, j)] = max(dp(i - 1, j), dp(i, j - 1))
        
        return memo[(i, j)]
    
    return dp(len(text1) - 1, len(text2) - 1)
```

#### Bottom-Up Template
```python
def lcs_bu(text1, text2):
    m, n = len(text1), len(text2)
    # dp[i][j] = LCS length of text1[0:i] and text2[0:j]
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

### Edit Distance Template

#### Top-Down Template
```python
def edit_distance_td(word1, word2):
    memo = {}
    
    def dp(i, j):
        # Base cases
        if i == 0: return j  # Insert j characters
        if j == 0: return i  # Delete i characters
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        if word1[i-1] == word2[j-1]:
            # Characters match
            memo[(i, j)] = dp(i-1, j-1)
        else:
            # Try all three operations
            memo[(i, j)] = 1 + min(
                dp(i-1, j),    # Delete
                dp(i, j-1),    # Insert
                dp(i-1, j-1)   # Replace
            )
        
        return memo[(i, j)]
    
    return dp(len(word1), len(word2))
```

#### Bottom-Up Template
```python
def edit_distance_bu(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]
```

### Common Variations:
- **Longest Common Substring** (must be contiguous)
- **Shortest Common Supersequence**
- **Delete Operation for Two Strings**
- **Minimum ASCII Delete Sum**
- **Interleaving String**

---

## 3. Grid/Path Pattern

**When to use:** Navigate through a 2D grid with constraints to find optimal paths.

### Unique Paths Template

#### Top-Down Template
```python
def unique_paths_td(m, n):
    memo = {}
    
    def dp(i, j):
        # Base cases
        if i == 0 and j == 0:
            return 1
        if i < 0 or j < 0:
            return 0
            
        if (i, j) in memo:
            return memo[(i, j)]
        
        # Can come from top or left
        memo[(i, j)] = dp(i-1, j) + dp(i, j-1)
        return memo[(i, j)]
    
    return dp(m-1, n-1)
```

#### Bottom-Up Template
```python
def unique_paths_bu(m, n):
    # dp[i][j] = number of paths to reach (i, j)
    dp = [[0 for _ in range(n)] for _ in range(m)]
    
    # Initialize first row and column
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]
```

### Minimum Path Sum Template

#### Top-Down Template
```python
def min_path_sum_td(grid):
    rows, cols = len(grid), len(grid[0])
    memo = {}
    
    def dp(i, j):
        # Base case
        if i == rows-1 and j == cols-1:
            return grid[i][j]
        
        # Out of bounds
        if i >= rows or j >= cols:
            return float('inf')
            
        if (i, j) in memo:
            return memo[(i, j)]
        
        # Choose minimum path
        memo[(i, j)] = grid[i][j] + min(dp(i+1, j), dp(i, j+1))
        return memo[(i, j)]
    
    return dp(0, 0)
```

#### Bottom-Up Template
```python
def min_path_sum_bu(grid):
    rows, cols = len(grid), len(grid[0])
    dp = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Initialize starting point
    dp[0][0] = grid[0][0]
    
    # Fill first row
    for j in range(1, cols):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    # Fill first column
    for i in range(1, rows):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    
    # Fill rest of the grid
    for i in range(1, rows):
        for j in range(1, cols):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
    
    return dp[rows-1][cols-1]
```

---

## 4. Recursive Number Pattern

**When to use:** Build numbers following specific rules or count ways to reach a target.

### Climbing Stairs Template

#### Top-Down Template
```python
def climb_stairs_td(n):
    memo = {}
    
    def dp(i):
        # Base cases
        if i == 0 or i == 1:
            return 1
        if i < 0:
            return 0
            
        if i in memo:
            return memo[i]
        
        # Can come from 1 or 2 steps back
        memo[i] = dp(i-1) + dp(i-2)
        return memo[i]
    
    return dp(n)
```

#### Bottom-Up Template
```python
def climb_stairs_bu(n):
    if n <= 1:
        return 1
    
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
```

### House Robber Template

#### Top-Down Template
```python
def rob_td(nums):
    memo = {}
    
    def dp(i):
        if i < 0:
            return 0
            
        if i in memo:
            return memo[i]
        
        # Either rob this house or skip it
        memo[i] = max(nums[i] + dp(i-2), dp(i-1))
        return memo[i]
    
    return dp(len(nums) - 1)
```

#### Bottom-Up Template
```python
def rob_bu(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    for i in range(2, n):
        dp[i] = max(nums[i] + dp[i-2], dp[i-1])
    
    return dp[n-1]
```

---

## 5. Stock Trading Pattern

**When to use:** Optimize buying/selling decisions with transaction constraints.

### Stock Trading Template

#### Top-Down Template
```python
def max_profit_td(prices, fee=0, cooldown=False):
    n = len(prices)
    memo = {}
    
    def dp(day, holding):
        if day >= n:
            return 0
            
        if (day, holding) in memo:
            return memo[(day, holding)]
        
        if holding:
            # Can sell or hold
            sell = prices[day] - fee + dp(day + (2 if cooldown else 1), False)
            hold = dp(day + 1, True)
            memo[(day, holding)] = max(sell, hold)
        else:
            # Can buy or skip
            buy = dp(day + 1, True) - prices[day]
            skip = dp(day + 1, False)
            memo[(day, holding)] = max(buy, skip)
        
        return memo[(day, holding)]
    
    return dp(0, False)
```

#### Bottom-Up Template
```python
def max_profit_bu(prices, fee=0):
    n = len(prices)
    # dp[i][0] = max profit on day i not holding stock
    # dp[i][1] = max profit on day i holding stock
    dp = [[0, 0] for _ in range(n + 1)]
    
    for day in range(n - 1, -1, -1):
        # Not holding stock
        buy = dp[day + 1][1] - prices[day] - fee
        skip = dp[day + 1][0]
        dp[day][0] = max(buy, skip)
        
        # Holding stock
        sell = prices[day] + dp[day + 1][0]
        hold = dp[day + 1][1]
        dp[day][1] = max(sell, hold)
    
    return dp[0][0]
```

---

## 6. Interval Scheduling Pattern

**When to use:** Schedule non-overlapping intervals to maximize profit/count.

### Job Scheduling Template

#### Template with Binary Search
```python
import bisect

def job_scheduling(start_times, end_times, profits):
    n = len(start_times)
    # Combine and sort by end time
    jobs = list(zip(start_times, end_times, profits))
    jobs.sort(key=lambda x: x[1])
    
    end_times_sorted = [job[1] for job in jobs]
    
    # dp[i] = max profit using first i jobs
    dp = [0] * (n + 1)
    
    for i, (start, end, profit) in enumerate(jobs):
        # Find latest job that doesn't overlap
        idx = bisect.bisect_right(end_times_sorted, start) - 1
        
        # Take current job or skip it
        dp[i + 1] = max(dp[i], profit + dp[idx + 1])
    
    return dp[n]
```

---

## Pattern Recognition Guide

### How to Identify Patterns:

1. **Knapsack**: Items with weights/costs, capacity constraints
   - Keywords: "subset", "capacity", "weight", "choose items"

2. **LCS**: Two sequences, find similarity/difference
   - Keywords: "two strings", "common", "edit", "transform"

3. **Grid/Path**: 2D movement, path counting/optimization
   - Keywords: "grid", "path", "robot", "unique ways"

4. **Recursive Numbers**: Building up numbers, counting ways
   - Keywords: "stairs", "fibonacci", "ways to reach", "house robber"

5. **Stock Trading**: Time series with buy/sell decisions
   - Keywords: "buy", "sell", "stock", "transaction", "cooldown"

6. **Interval Scheduling**: Time intervals, scheduling optimization
   - Keywords: "intervals", "schedule", "non-overlapping", "events"

### State Design Tips:

1. **Identify what changes**: Position, remaining capacity, current state
2. **Define states clearly**: What information is needed to make decisions?
3. **Handle base cases**: What are the simplest scenarios?
4. **Check constraints**: Bounds, validity of states
5. **Optimize space**: Can you reduce dimensions or use rolling arrays?

### Implementation Strategy:

1. **Start with Top-Down**: Easier to think recursively
2. **Add Memoization**: Cache repeated subproblems
3. **Convert to Bottom-Up**: Better space/time complexity
4. **Optimize Space**: Use rolling arrays when possible