from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = prices[0]
        track = None
        diff = None
        flag = True

        day = 0  # start from 1 since day starts from 1 not 0
        # but since we start iter from prices[1:] then index = 0 is ok
        final_index = 0

        for i, curr_price in enumerate(prices[1:]):
            if curr_price < min_price:  # price = 1, min_price = 2
                # means new min price and we do not even need
                # to check diff since 1 - 2 < 0

                min_price = curr_price
                flag = False

            else:  # price = 10, min_price = 2
                # diff = curr_price - min_price
                if track is None:
                    track = curr_price
                    final_index = i
                    diff = curr_price - min_price

                else:
                    if curr_price > track:
                        track = curr_price
                        final_index = i

                        diff = curr_price - min_price
            day += 1

        if not flag:
            return 0
        return diff


import time

import matplotlib.pyplot as plt
import numpy as np


def fib(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


def measure_time(func, args_list):
    times = []
    for args in args_list:
        start = time.time()
        func(args)
        end = time.time()
        times.append(end - start)
    return times


args_list = list(range(1, 31))  # Adjust this based on the complexity of your function
times = measure_time(fib, args_list)

plt.plot(args_list, times, label="Execution Time")
plt.xlabel("Input Size")
plt.ylabel("Time (seconds)")
plt.legend()
plt.show()
