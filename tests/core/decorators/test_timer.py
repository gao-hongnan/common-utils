import time
import unittest

from common_utils.core.decorators.timer import timer


# TODO: Do we want to refactor unit tests to use pytest?
class TestTimer(unittest.TestCase):
    def test_timer(self):
        timer_instance = timer(
            display_table=False,
            unit="seconds",
            decimal_places=4,
            store_times=True,
            log=False,
        )

        # Decorate a dummy function that sleeps for a known time
        @timer_instance
        def dummy_function(sleep_time):
            time.sleep(sleep_time)

        # Execute the decorated function
        sleep_time = 0.1
        dummy_function(sleep_time)

        # Check the execution_times
        self.assertIn(
            "dummy_function",
            timer_instance.execution_times,
            "Function name not found in execution_times",
        )
        self.assertGreaterEqual(
            timer_instance.execution_times["dummy_function"][0],
            sleep_time,
            "Recorded time less than expected",
        )


if __name__ == "__main__":
    unittest.main()
