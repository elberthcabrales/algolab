import unittest
from prefix_sum.core import PrefixSum
class TestQueriesInPrefixSum(unittest.TestCase):


    def test_range_of_sums(self):
        nums = [1, 6, 3, 2, 7, 2]
        na = PrefixSum(nums)
        ranges = [[0, 3], [2, 5], [2, 4]]
        expected = [12, 14, 12]
        result = [na.sum_range(x, y) for x, y in ranges]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
