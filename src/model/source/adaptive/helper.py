from contextlib import contextmanager, AbstractContextManager
from fcntl import flock, LOCK_EX, LOCK_UN
import inspect
import io
from itertools import zip_longest
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np



@contextmanager
def open_files(filepaths, mode='r'):
    files = []
    try:
        files = [Path(filepath).open(mode) for filepath in filepaths]
        yield files
    finally:
        [f.close() for f in files]


def yield_lines_in_parallel(filepaths, strip=True, strict=True, n_lines=float('inf')):
    assert type(filepaths) == list
    with open_files(filepaths) as files:
        for i, parallel_lines in enumerate(zip_longest(*files)):
            if i >= n_lines:
                break
            if None in parallel_lines:
                assert not strict, f'Files don\'t have the same number of lines: {filepaths}, use strict=False'
            if strip:
                parallel_lines = [l.rstrip('\n') if l is not None else None for l in parallel_lines]
            yield parallel_lines


class FilesWrapper:
    '''Write to multiple open files at the same time'''
    def __init__(self, files, strict=True):
        self.files = files
        self.strict = strict  # Whether to raise an exception when a line is None

    def write(self, lines):
        assert len(lines) == len(self.files)
        for line, f in zip(lines, self.files):
            if line is None:
                assert not self.strict
                continue
            f.write(line.rstrip('\n') + '\n')


@contextmanager
def write_lines_in_parallel(filepaths, strict=True):
    with open_files(filepaths, 'w') as files:
        yield FilesWrapper(files, strict=strict)


def write_lines(lines, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open('w') as f:
        for line in lines:
            f.write(line + '\n')


def yield_lines(filepath, n_lines=float('inf'), prop=1):
    if prop < 1:
        assert n_lines == float('inf')
        n_lines = int(prop * count_lines(filepath))
    with open(filepath, 'r') as f:
        for i, l in enumerate(f):
            if i >= n_lines:
                break
            yield l.rstrip('\n')


def read_lines(filepath, n_lines=float('inf'), prop=1):
    return list(yield_lines(filepath, n_lines, prop))


def count_lines(filepath):
    n_lines = 0
    with Path(filepath).open() as f:
        for l in f:
            n_lines += 1
    return n_lines


@contextmanager
def open_with_lock(filepath, mode):
    with open(filepath, mode) as f:
        flock(f, LOCK_EX)
        yield f
        flock(f, LOCK_UN)


def get_lockfile_path(path):
    path = Path(path)
    if path.is_dir():
        return path / '.lockfile'
    if path.is_file():
        return path.parent / f'.{path.name}.lockfile'


@contextmanager
def lock_directory(dir_path):
    # TODO: Locking a directory should lock all files in that directory
    # Right now if we lock foo/, someone else can lock foo/bar.txt
    # TODO: Nested with lock_directory() should not be blocking
    assert Path(dir_path).exists(), f'Directory does not exists: {dir_path}'
    lockfile_path = get_lockfile_path(dir_path)
    with open_with_lock(lockfile_path, 'w'):
        yield


def safe_division(a, b):
    if b == 0:
        return 0
    return a / b


def harmonic_mean(values, coefs=None):
    if 0 in values:
        return 0
    values = np.array(values)
    if coefs is None:
        coefs = np.ones(values.shape)
    values = np.array(values)
    coefs = np.array(coefs)
    return np.sum(coefs) / np.dot(coefs, 1 / values)


@contextmanager
def mute(mute_stdout=True, mute_stderr=True):
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    if mute_stdout:
        sys.stdout = io.StringIO()
    if mute_stderr:
        sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr


@contextmanager
def log_stdout(filepath, mute_stdout=False):
    '''Context manager to write both to stdout and to a file'''
    class MultipleStreamsWriter:
        def __init__(self, streams):
            self.streams = streams

        def write(self, message):
            for stream in self.streams:
                stream.write(message)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    save_stdout = sys.stdout
    log_file = open(filepath, 'w')
    if mute_stdout:
        sys.stdout = MultipleStreamsWriter([log_file])  # Write to file only
    else:
        sys.stdout = MultipleStreamsWriter([save_stdout, log_file])  # Write to both stdout and file
    try:
        yield
    finally:
        sys.stdout = save_stdout
        log_file.close()


def add_dicts(*dicts):
    return {k: v for dic in dicts for k, v in dic.items()}


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_allowed_kwargs(func, *args, **kwargs):
    expected_args = inspect.getargspec(func).args
    allowed_kwargs = expected_args[len(args):]
    return {k: v for k, v in kwargs.items() if k in allowed_kwargs}