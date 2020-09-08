import os
import shutil
import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def check_directory(dir_name):
    dir_name = os.path.abspath(dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def copy_file(source_path, target_path):
    path = os.path.join(target_path, os.path.basename(source_path))
    shutil.copyfile(source_path, path)
