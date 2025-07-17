import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        from utils.util import setup_logger
        logger = setup_logger(log_file="/home/my_proj/logs/app1.log")
        logger.debug("This is a debug message.")
        logger.info("This is an info message.")
        logger.warning("This is a warning.")
        logger.error("This is an error.")
        logger.critical("This is critical!")

if __name__ == '__main__':
    unittest.main()
