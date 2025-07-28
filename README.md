# LeetCode Pattern Templates & Cheatsheet

## 🎯 How to Use This Guide
1. **Study the pattern** before solving problems
2. **Copy the template** and modify for specific problems
3. **Practice recognition** - identify which pattern to use
4. **Time complexity** - always analyze and optimize

---

## 1. Two Pointers Pattern

### When to Use:
- Array/string with sorted property
- Finding pairs with specific sum
- Palindrome checks
- Container problems

### Template:
```python
def two_pointers_template(arr):
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []
```

### Key Problems:
- [1] Two Sum, [15] 3Sum, [11] Container With Most Water
- [125] Valid Palindrome, [167] Two Sum II

---

## 2. Sliding Window Pattern

### When to Use:
- Subarray/substring problems
- Find optimal window size
- Character frequency problems

### Template:
```python
def sliding_window_template(s):
    left = 0
    window_map = {}
    result = 0
    
    for right in range(len(s)):
        # Expand window
        char = s[right]
        window_map[char] = window_map.get(char, 0) + 1
        
        # Contract window when condition violated
        while window_condition_violated:
            left_char = s[left]
            window_map[left_char] -= 1
            if window_map[left_char] == 0:
                del window_map[left_char]
            left += 1
        
        # Update result
        result = max(result, right - left + 1)
    
    return result
```

### Key Problems:
- [3] Longest Substring Without Repeating Characters
- [76] Minimum Window Substring, [424] Longest Repeating Character Replacement

---

## 3. Fast & Slow Pointers (Floyd's Cycle Detection)

### When to Use:
- Linked list cycle detection
- Finding middle element
- Palindrome linked list

### Template:
```python
def floyd_cycle_detection(head):
    if not head or not head.next:
        return False
    
    slow = fast = head
    
    # Phase 1: Detect cycle
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            break
    else:
        return False  # No cycle
    
    # Phase 2: Find cycle start (if needed)
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow  # Cycle start node
```

### Key Problems:
- [141] Linked List Cycle, [142] Linked List Cycle II
- [876] Middle of Linked List, [234] Palindrome Linked List

---

## 4. Binary Search Pattern

### When to Use:
- Sorted arrays
- Search space can be reduced by half
- Find minimum/maximum satisfying condition

### Template:
```python
def binary_search_template(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# For finding first/last occurrence
def binary_search_boundary(arr, target, find_first=True):
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            if find_first:
                right = mid - 1  # Continue searching left
            else:
                left = mid + 1   # Continue searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

### Key Problems:
- [704] Binary Search, [33] Search in Rotated Sorted Array
- [153] Find Minimum in Rotated Sorted Array

---

## 5. Tree Traversal Patterns

### DFS (Depth-First Search)
```python
# Recursive DFS
def dfs_recursive(root):
    if not root:
        return
    
    # Process current node
    process(root.val)
    
    # Recurse on children
    dfs_recursive(root.left)
    dfs_recursive(root.right)

# Iterative DFS (using stack)
def dfs_iterative(root):
    if not root:
        return
    
    stack = [root]
    
    while stack:
        node = stack.pop()
        process(node.val)
        
        # Add children (right first for left-to-right processing)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
```

### BFS (Breadth-First Search)
```python
from collections import deque

def bfs_level_order(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

### Key Problems:
- [102] Binary Tree Level Order Traversal
- [104] Maximum Depth of Binary Tree, [226] Invert Binary Tree

---

## 6. Dynamic Programming Patterns

### 1D DP Template
```python
def dp_1d_template(arr):
    n = len(arr)
    dp = [0] * n
    
    # Base case
    dp[0] = arr[0]
    
    # Fill DP table
    for i in range(1, n):
        dp[i] = max(dp[i-1], arr[i])  # Example: max ending here
    
    return dp[n-1]
```

### 2D DP Template
```python
def dp_2d_template(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    
    # Base cases
    dp[0][0] = grid[0][0]
    
    # Fill first row and column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    # Fill rest of table
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    
    return dp[m-1][n-1]
```

### Key Problems:
- [70] Climbing Stairs, [198] House Robber
- [62] Unique Paths, [64] Minimum Path Sum

---

## 7. Backtracking Pattern

### Template:
```python
def backtrack_template():
    result = []
    
    def backtrack(current_path, remaining_choices):
        # Base case
        if is_valid_solution(current_path):
            result.append(current_path[:])  # Make a copy
            return
        
        # Try each choice
        for choice in remaining_choices:
            # Make choice
            current_path.append(choice)
            
            # Recurse with updated state
            backtrack(current_path, get_next_choices(choice))
            
            # Backtrack (undo choice)
            current_path.pop()
    
    backtrack([], initial_choices)
    return result
```

### Key Problems:
- [78] Subsets, [46] Permutations
- [39] Combination Sum, [79] Word Search

---

## 8. Graph Traversal Patterns

### DFS for Graphs
```python
def dfs_graph(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    process(start)
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_graph(graph, neighbor, visited)
    
    return visited
```

### BFS for Graphs
```python
from collections import deque

def bfs_graph(graph, start):
    visited = {start}
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        process(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited
```

### Key Problems:
- [200] Number of Islands, [133] Clone Graph
- [207] Course Schedule, [417] Pacific Atlantic Water Flow

---

## 9. Heap/Priority Queue Pattern

### Template:
```python
import heapq

# Min heap (default in Python)
def heap_template():
    heap = []
    
    # Add elements
    heapq.heappush(heap, value)
    
    # Remove minimum
    min_val = heapq.heappop(heap)
    
    # For max heap, negate values
    heapq.heappush(heap, -value)  # Add
    max_val = -heapq.heappop(heap)  # Remove
    
    return heap

# K largest elements
def find_k_largest(nums, k):
    return heapq.nlargest(k, nums)

# K smallest elements  
def find_k_smallest(nums, k):
    return heapq.nsmallest(k, nums)
```

### Key Problems:
- [215] Kth Largest Element in Array
- [295] Find Median from Data Stream, [23] Merge k Sorted Lists

---

## 10. Monotonic Stack Pattern

### When to Use:
- Next/previous greater/smaller element
- Histogram area problems
- Temperature problems

### Template:
```python
def monotonic_stack_template(arr):
    stack = []  # Stores indices
    result = [-1] * len(arr)
    
    for i in range(len(arr)):
        # While stack not empty and current element breaks monotonic property
        while stack and arr[i] > arr[stack[-1]]:  # For next greater element
            index = stack.pop()
            result[index] = arr[i]
        
        stack.append(i)
    
    return result
```

### Key Problems:
- [739] Daily Temperatures, [496] Next Greater Element I
- [84] Largest Rectangle in Histogram

---

## 🧠 Pattern Recognition Flowchart

```
Problem Analysis:
├── Array/String Input?
│   ├── Two pointers needed? → Two Pointers Pattern
│   ├── Subarray/substring? → Sliding Window Pattern
│   └── Sorted array search? → Binary Search Pattern
├── Tree/Graph Input?
│   ├── Level-by-level? → BFS Pattern
│   ├── Path finding? → DFS Pattern
│   └── Shortest path? → BFS/Dijkstra Pattern
├── Generate all possibilities?
│   └── Backtracking Pattern
├── Optimization problem?
│   └── Dynamic Programming Pattern
└── Need min/max quickly?
    └── Heap Pattern
```

---

## 📝 Quick Reference Complexities

| Pattern | Time Complexity | Space Complexity |
|---------|----------------|------------------|
| Two Pointers | O(n) | O(1) |
| Sliding Window | O(n) | O(k) |
| Binary Search | O(log n) | O(1) |
| DFS/BFS | O(V + E) | O(V) |
| Dynamic Programming | O(n²) typical | O(n²) typical |
| Backtracking | O(2ⁿ) worst | O(n) |
| Heap Operations | O(log n) | O(n) |

---

## 💡 Pro Tips for Pattern Recognition

1. **Read problem twice** - understand what's being asked
2. **Identify input/output** - guides pattern selection
3. **Look for keywords**: "subarray", "substring", "path", "optimize"
4. **Consider constraints** - helps choose optimal approach
5. **Start with brute force** - then optimize using patterns

Remember: **Practice makes pattern recognition automatic!**
