import logging
import os
import sys
import os.path as osp
import os
import logging
import sys


def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Stream handler to output log messages to stdout
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Decide log file name based on training or testing
        base_filename = "train_log" if if_train else "test_log"
        file_extension = ".txt"
        log_filename = os.path.join(save_dir, f"{base_filename}{file_extension}")

        # Check if the file exists and adjust filename if necessary
        counter = 1
        while os.path.exists(log_filename):
            log_filename = os.path.join(save_dir, f"{base_filename}_{counter}{file_extension}")
            counter += 1

        # Create a file handler for logging
        fh = logging.FileHandler(log_filename, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
