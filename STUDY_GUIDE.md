# Guía de Estudio: Algoritmos y Patrones - AlgoLab

## 📋 Índice
- [Arrays](#arrays)
- [Strings](#strings)
- [Two Pointers](#two-pointers)
- [Sliding Window](#sliding-window)
- [Linked Lists](#linked-lists)
- [Stacks y Queues](#stacks-y-queues)
- [Heaps](#heaps)
- [Trees](#trees)
- [Trie](#trie)
- [Graphs](#graphs)
- [Dynamic Programming](#dynamic-programming)
- [Backtracking](#backtracking)
- [Greedy](#greedy)
- [Binary Search](#binary-search)
- [Bit Manipulation](#bit-manipulation)
- [Matrix](#matrix)
- [Sorting](#sorting)
- [Math](#math)
- [Prefix Sum](#prefix-sum)
- [Counting](#counting)

---

## Arrays

**Archivo**: [arrays/templates.ipynb](arrays/templates.ipynb)

### Básico
- [ ] Median of Two Sorted Arrays - Merge de arrays ordenados
- [ ] Group Anagrams - Hash con Counter
- [ ] Check if Array is Sorted and Rotated

### Intermedio
- [ ] Maximum Subarray - **Kadane's Algorithm**
- [ ] Majority Element - **Boyer-Moore Voting Algorithm**
- [ ] Most Frequent Even Element - Counter con ties
- [ ] Group Shifted Strings - Pattern matching con mod 26
- [ ] Merge Intervals - [arrays/merged_intervals.ipynb](arrays/merged_intervals.ipynb)

### Avanzado
- [ ] Array utility functions - [arrays/stuff_functions.ipynb](arrays/stuff_functions.ipynb)

---

## Strings

**Archivo**: [strings/template.ipynb](strings/template.ipynb)

### Básico a Intermedio
- [ ] String manipulation patterns
- [ ] Character shifting (similar a Group Shifted Strings)
- [ ] Anagram detection

---

## Two Pointers

**Archivo**: [two-points/templates.ipynb](two-points/templates.ipynb)

### Básico
- [ ] Two Pointers - Mismo array (palindrome check)
- [ ] Two Pointers - Dos arrays (merge sorted arrays)
- [ ] Two Sum II - Input Array Is Sorted

### Intermedio
- [ ] Container With Most Water
- [ ] 3Sum - Two pointers con loop externo
- [ ] 4Sum - Dos loops externos + two pointers
- [ ] Trapping Rain Water - Two pointers con max tracking
- [ ] Valid Triangle Number

### Avanzado
- [ ] Next Permutation - Manipulación de arrays in-place

---

## Sliding Window

**Archivo**: [slide-window/template.ipynb](slide-window/template.ipynb)

### Básico
- [ ] Repeated DNA Sequences - Window fijo de tamaño 10
- [ ] Permutation in String - Window fijo con Counter

### Intermedio
- [ ] Minimum Size Subarray Sum - Window variable
- [ ] Longest Substring Without Repeating Characters - Window variable con hash
- [ ] Subarray Product Less Than K - Window variable con producto
- [ ] Count the Number of Good Subarrays - Window con conteo de pares

### Avanzado
- [ ] Minimum Window Substring - Window variable con múltiples condiciones
- [ ] Maximum Frequency Score of a Subarray - Window con modular arithmetic
- [ ] Sliding Window Maximum - Window con Monotonic Deque

---

## Linked Lists

**Archivo**: [linked-list/templates.ipynb](linked-list/templates.ipynb)

### Básico
- [ ] Fast & Slow Pointers - Encontrar el medio
- [ ] Reverse Linked List - Iterativo
- [ ] Remove Duplicates from Sorted List II

### Intermedio
- [ ] Remove Nth Node From End - Fast & slow con gap
- [ ] Delete Middle Node - Fast & slow
- [ ] Odd Even Linked List - Reordenamiento
- [ ] Swap Nodes in Pairs - Manipulación de punteros
- [ ] Palindrome Linked List - Reverse + comparación

### Avanzado
- [ ] Reorder List - Middle + reverse + merge
- [ ] Maximum Twin Sum - Middle + reverse
- [ ] Reverse Nodes in K-Group - Reverse en bloques

---

## Stacks y Queues

**Archivo**: [stack-queue/templates.ipynb](stack-queue/templates.ipynb)

### Básico a Avanzado
- [ ] Stack patterns básicos
- [ ] Queue patterns básicos
- [ ] Monotonic stack/queue patterns
- [ ] Deque patterns

---

## Heaps

**Archivos**:
- [heap/template.ipynb](heap/template.ipynb)
- [heap/ktop.ipynb](heap/ktop.ipynb)
- [heap/smallest_range.ipynb](heap/smallest_range.ipynb)

### Básico
- [ ] Heap implementation básico
- [ ] K-th element patterns

### Intermedio
- [ ] Top K elements
- [ ] Smallest Range patterns

### Avanzado
- [ ] Merge K sorted structures

---

## Trees

**Archivos**:
- [tree/templates.ipynb](tree/templates.ipynb)
- [tree/creation.ipynb](tree/creation.ipynb)
- [tree/segment_tree.ipynb](tree/segment_tree.ipynb)
- [tree/ntree.ipynb](tree/ntree.ipynb)

### Básico
- [ ] Tree traversals (inorder, preorder, postorder)
- [ ] Tree creation patterns
- [ ] N-ary tree patterns

### Intermedio
- [ ] Binary Search Tree operations
- [ ] Tree manipulation

### Avanzado
- [ ] Segment Tree - Range queries y updates
- [ ] Advanced tree algorithms

---

## Trie

**Archivos**:
- [trie/template.ipynb](trie/template.ipynb)
- [trie/withdfs.ipynb](trie/withdfs.ipynb)

### Básico
- [ ] Trie implementation
- [ ] Insert, search, startsWith

### Intermedio
- [ ] Trie con DFS
- [ ] Word search patterns

### Avanzado
- [ ] Trie optimizations

---

## Graphs

**Archivos**:
- [graph/bfs.ipynb](graph/bfs.ipynb)
- [graph/dfs.ipynb](graph/dfs.ipynb)
- [graph/dsu.ipynb](graph/dsu.ipynb) - Disjoint Set Union
- [graph/topological_sort.ipynb](graph/topological_sort.ipynb)
- [graph/distance.ipynb](graph/distance.ipynb)
- [graph/clone_graph.ipynb](graph/clone_graph.ipynb)
- [graph/wild_card.ipynb](graph/wild_card.ipynb)

### Básico
- [ ] BFS (Breadth-First Search)
- [ ] DFS (Depth-First Search)
- [ ] Clone Graph
- [ ] Graph traversal básico

### Intermedio
- [ ] Topological Sort
- [ ] Distance algorithms
- [ ] Disjoint Set Union (DSU/Union-Find)
- [ ] Connected components
- [ ] Cycle detection

### Avanzado
- [ ] Wildcard matching patterns
- [ ] Advanced graph algorithms

---

## Dynamic Programming

**Archivos**:
- [dynamic_programming/templates.ipynb](dynamic_programming/templates.ipynb)
- [dynamic_programming/patterns.ipynb](dynamic_programming/patterns.ipynb)
- [dynamic_programming/patterns/longest_common.ipynb](dynamic_programming/patterns/longest_common.ipynb)
- [dynamic_programming/patterns/knapsack.ipynb](dynamic_programming/patterns/knapsack.ipynb)
- [dynamic_programming/patterns/recursive_numbers.ipynb](dynamic_programming/patterns/recursive_numbers.ipynb)

### Básico
- [ ] House Robber - DP 1D básico
- [ ] Delete and Earn - Variante de House Robber
- [ ] Fibonacci Number - Top-down y bottom-up
- [ ] Pascal's Triangle
- [ ] Maximum Subarray (Kadane's)

### Intermedio
- [ ] Unique Paths - DP 2D grid
- [ ] Unique Paths II - DP con obstáculos
- [ ] Best Time to Buy and Sell Stock II
- [ ] Longest Increasing Subsequence (LIS)
- [ ] Longest Common Subsequence (LCS)
- [ ] Longest Mountain in Array
- [ ] Count Pairs (Binary Search + DP)

### Avanzado
- [ ] Check Path with Equal 0s and 1s - DP con balance
- [ ] Knapsack patterns
- [ ] Recursive number patterns
- [ ] DP optimization techniques

---

## Backtracking

**Archivo**: [backtracking/templates.ipynb](backtracking/templates.ipynb)

### Básico a Avanzado
- [ ] Subset generation
- [ ] Permutations
- [ ] Combinations
- [ ] Backtracking con poda
- [ ] Backtracking optimizado

---

## Greedy

**Archivos**:
- [greedy/templates.ipynb](greedy/templates.ipynb)
- [greedy/streak.ipynb](greedy/streak.ipynb)

### Básico a Intermedio
- [ ] Greedy choice patterns
- [ ] Interval scheduling
- [ ] Activity selection
- [ ] Streak patterns

---

## Binary Search

**Archivos**:
- [binary_search/template.ipynb](binary_search/template.ipynb)
- [binary_search/rotated_sorted.ipynb](binary_search/rotated_sorted.ipynb)

### Básico
- [ ] Binary Search básico
- [ ] Binary Search en array

### Intermedio
- [ ] Search in Rotated Sorted Array
- [ ] Find Minimum in Rotated Sorted Array
- [ ] Binary Search variations

### Avanzado
- [ ] Binary Search en respuesta
- [ ] Binary Search en matrices

---

## Bit Manipulation

**Archivo**: [bit-manipulation/max_cost_toll_in_trip.ipynb](bit-manipulation/max_cost_toll_in_trip.ipynb)

### Intermedio a Avanzado
- [ ] Bit operations básicas
- [ ] Max Cost Toll in Trip
- [ ] Bit masks
- [ ] XOR patterns

---

## Matrix

**Archivos**:
- [matrix/template.ipynb](matrix/template.ipynb)
- [matrix/patterns.ipynb](matrix/patterns.ipynb)

### Básico a Intermedio
- [ ] Matrix traversal
- [ ] Matrix manipulation
- [ ] Matrix patterns comunes
- [ ] DFS/BFS en matrices

---

## Sorting

**Archivos**:
- [sort/array.ipynb](sort/array.ipynb)
- [sort/sorted-set.ipynb](sort/sorted-set.ipynb)

### Básico
- [ ] Sorting algorithms básicos
- [ ] Array sorting patterns

### Intermedio
- [ ] Sorted Set patterns
- [ ] Custom sorting
- [ ] Sorting with constraints

---

## Math

**Archivo**: [math/lab.ipynb](math/lab.ipynb)

### Básico a Avanzado
- [ ] Mathematical algorithms
- [ ] Number theory
- [ ] Combinatorics
- [ ] Mathematical patterns

---

## Prefix Sum

**Archivo**: [prefix_sum/templates.ipynb](prefix_sum/templates.ipynb)

### Básico
- [ ] 1D Prefix Sum
- [ ] Subarray sum queries

### Intermedio
- [ ] 2D Prefix Sum
- [ ] Range sum queries
- [ ] Prefix sum optimizations

---

## Counting

**Archivo**: [counting/templates.ipynb](counting/templates.ipynb)

### Básico a Intermedio
- [ ] Frequency counting
- [ ] Counter patterns
- [ ] Counting optimizations

---

## Recursion

**Archivo**: [recursion/templates.ipynb](recursion/templates.ipynb)

### Básico a Avanzado
- [ ] Recursive patterns básicos
- [ ] Recursion con memorización
- [ ] Tail recursion
- [ ] Recursion optimization

---

## 📚 Labs y Utilidades

**Archivos de práctica y herramientas**:
- [labs/practica.ipynb](labs/practica.ipynb) - Ejercicios de práctica
- [labs/collections.ipynb](labs/collections.ipynb) - Python collections
- [labs/bisect.ipynb](labs/bisect.ipynb) - Módulo bisect
- [labs/sortedcontainers.ipynb](labs/sortedcontainers.ipynb) - SortedContainers library
- [labs/suffs.ipynb](labs/suffs.ipynb) - Suffix patterns

---

## 🎯 Plan de Estudio Recomendado (14 semanas)

### Semana 1: Fundamentos de Arrays y Strings
- Arrays básico e intermedio
- Strings básico
- Counting patterns
- **Objetivo**: Dominar manipulación básica de arrays

### Semana 2: Two Pointers y Sliding Window
- Two Pointers (todos los niveles)
- Sliding Window básico e intermedio
- **Objetivo**: Reconocer cuándo usar cada patrón

### Semana 3: Linked Lists
- Linked Lists (todos los niveles)
- Fast & Slow pointers
- Reverse patterns
- **Objetivo**: Dominar manipulación de punteros

### Semana 4: Stacks, Queues y Prefix Sum
- Stack/Queue patterns
- Monotonic stack/queue
- Prefix Sum 1D y 2D
- **Objetivo**: Entender estructuras auxiliares

### Semana 5: Trees Básico
- Tree traversals
- Tree creation
- BST operations
- N-ary trees
- **Objetivo**: Dominar recorridos de árboles

### Semana 6: Trees Avanzado y Heaps
- Segment Tree
- Heaps básico
- Top K patterns
- **Objetivo**: Estructuras de datos avanzadas

### Semana 7: Graphs - Traversal
- BFS
- DFS
- Clone Graph
- **Objetivo**: Dominar recorridos de grafos

### Semana 8: Graphs - Algoritmos Especiales
- DSU (Union-Find)
- Topological Sort
- Distance algorithms
- **Objetivo**: Algoritmos especializados de grafos

### Semana 9: Dynamic Programming - 1D
- House Robber
- Fibonacci
- LIS (Longest Increasing Subsequence)
- **Objetivo**: DP en una dimensión

### Semana 10: Dynamic Programming - 2D
- Unique Paths
- LCS (Longest Common Subsequence)
- Grid DP
- **Objetivo**: DP en dos dimensiones

### Semana 11: DP Avanzado y Patterns
- Knapsack patterns
- DP optimization
- Recursive numbers
- **Objetivo**: Patrones avanzados de DP

### Semana 12: Binary Search y Backtracking
- Binary Search (todos los niveles)
- Rotated arrays
- Backtracking patterns
- **Objetivo**: Búsqueda y generación

### Semana 13: Algoritmos Especializados
- Trie y Trie con DFS
- Bit Manipulation
- Greedy patterns
- Matrix patterns
- **Objetivo**: Técnicas especializadas

### Semana 14: Revisión y Práctica Integrada
- Labs y práctica
- Problemas que combinan múltiples patrones
- Mock interviews
- **Objetivo**: Integrar todo el conocimiento

---

## 💡 Algoritmos Clave por Importancia

### Must-Know (Críticos para entrevistas)
1. **Kadane's Algorithm** - Maximum Subarray
2. **Boyer-Moore Voting** - Majority Element
3. **Two Pointers** - Multiple variations
4. **Sliding Window** - Fixed y variable
5. **Fast & Slow Pointers** - Linked Lists
6. **BFS/DFS** - Graphs y Trees
7. **Binary Search** - Todas las variantes
8. **DP básico** - 1D y 2D
9. **Union-Find (DSU)** - Graph connectivity
10. **Topological Sort** - DAG ordering

### Important (Frecuentes)
11. **Segment Tree** - Range queries
12. **Trie** - String search
13. **Heap patterns** - Top K
14. **LCS/LIS** - Subsequence problems
15. **Backtracking** - Generation problems
16. **Prefix Sum** - Range sums
17. **Monotonic Stack/Queue** - Next greater/smaller
18. **Greedy algorithms** - Optimization
19. **Knapsack** - Resource allocation
20. **Matrix traversal** - 2D problems

### Advanced (Para destacar)
21. **Bit Manipulation** - XOR, masks
22. **Advanced DP** - Optimization techniques
23. **Graph algorithms** - Distance, paths
24. **String algorithms** - Pattern matching
25. **Mathematical algorithms** - Number theory

---

## 📊 Complejidades a Memorizar

### Arrays
- Two Pointers: O(n) tiempo, O(1) espacio
- Sliding Window: O(n) tiempo, O(k) espacio
- Kadane's: O(n) tiempo, O(1) espacio
- Boyer-Moore: O(n) tiempo, O(1) espacio

### Linked Lists
- Reverse: O(n) tiempo, O(1) espacio
- Fast & Slow: O(n) tiempo, O(1) espacio

### Trees
- Traversal: O(n) tiempo, O(h) espacio
- Segment Tree: O(n) build, O(log n) query/update

### Graphs
- BFS/DFS: O(V + E) tiempo, O(V) espacio
- Union-Find: O(α(n)) por operación
- Topological Sort: O(V + E)

### DP
- 1D DP: O(n) tiempo, O(n) o O(1) espacio
- 2D DP: O(n*m) tiempo, O(n*m) espacio
- LIS: O(n²) o O(n log n)
- LCS: O(n*m)

### Search
- Binary Search: O(log n) tiempo, O(1) espacio
- Trie: O(m) insert/search donde m = longitud palabra

---

## 🔍 Cómo Identificar el Patrón

### Two Pointers
- Array/string ordenado
- Buscar pares/tripletes
- Comparar elementos desde extremos

### Sliding Window
- Subarray/substring continuo
- "Máximo/mínimo en ventana de tamaño k"
- Optimización de fuerza bruta O(n*k)

### Fast & Slow Pointers
- Linked List
- Detectar ciclos
- Encontrar medio

### BFS
- Shortest path en grafo no ponderado
- Level order traversal
- States exploration

### DFS
- Exploración exhaustiva
- Backtracking
- Connected components

### DP
- Optimal substructure
- Overlapping subproblems
- "Máximo/mínimo/número de formas"

### Binary Search
- Array ordenado
- Búsqueda en espacio de respuestas
- O(log n) requirement

### Greedy
- Optimal choice en cada paso
- No need to reconsider
- Proof de correctness

---

## 📝 Tips para Entrevistas

1. **Clarifica el problema**: Pregunta edge cases
2. **Piensa en voz alta**: Comunica tu razonamiento
3. **Empieza simple**: Brute force → Optimiza
4. **Dibuja ejemplos**: Visualiza el problema
5. **Analiza complejidad**: Tiempo y espacio
6. **Testea tu código**: Casos normales y edge cases
7. **Conoce tus estructuras**: Python collections, bisect, sortedcontainers

---

## 🛠️ Python Tools Importantes

- **collections**: Counter, defaultdict, deque
- **heapq**: Min heap operations
- **bisect**: Binary search en listas ordenadas
- **sortedcontainers**: SortedList, SortedDict, SortedSet
- **functools**: cache, lru_cache para memoization

**Ver**: [labs/](labs/) para ejemplos de cada herramienta

---

**Última actualización**: Noviembre 2024
**Objetivo**: Senior Software Engineer - Preparación completa
