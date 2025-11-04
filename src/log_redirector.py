"""
Log redirector utility to capture both stdout and stderr to a file while maintaining console output.
"""

import sys
from contextlib import contextmanager
from datetime import datetime


class Tee:
    """Tee class that writes to multiple file-like objects simultaneously."""

    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()

    def fileno(self):
        # Return the file descriptor of the first file (usually console)
        # This is needed for some operations that expect a real file descriptor
        return self.files[0].fileno()

    def close(self):
        for file in self.files:
            file.close()


@contextmanager
def redirect_output_to_file(log_file_path, append=True):
    """
    Context manager to redirect both stdout and stderr to a file while maintaining console output.

    Args:
        log_file_path (str): Path to the log file
        append (bool): If True, append to existing file. If False, overwrite.
    """
    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Open log file
    mode = "a" if append else "w"

    try:
        with open(log_file_path, mode) as log_file:
            # Add timestamp header
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"\n=== Log started at {timestamp} ===\n")
            log_file.flush()

            # Create Tee objects for both stdout and stderr
            sys.stdout = Tee(original_stdout, log_file)
            sys.stderr = Tee(original_stderr, log_file)

            yield log_file

    finally:
        # Restore original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def setup_permanent_logging(log_file_path, append=True):
    """
    Set up permanent logging for the entire script execution.
    Call this once at the beginning of your script.

    Args:
        log_file_path (str): Path to the log file
        append (bool): If True, append to existing file. If False, overwrite.

    Returns:
        file object: The log file handle (keep this reference to avoid garbage collection)
    """
    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Open log file
    mode = "a" if append else "w"
    log_file = open(log_file_path, mode)

    # Add timestamp header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"\n=== Log started at {timestamp} ===\n")
    log_file.flush()

    # Create Tee objects for both stdout and stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    return log_file


def restore_original_streams():
    """Restore original stdout and stderr streams."""
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == "__main__":
    # Example usage
    print("Testing log redirector...")

    # Method 1: Using context manager
    print("\n=== Testing context manager ===")
    with redirect_output_to_file("test_log.txt"):
        print("This goes to both console and log file")
        print("So does this print statement")
        print("And errors too:", file=sys.stderr)

        # Simulate an error
        try:
            1 / 0
        except ZeroDivisionError as e:
            print(f"Caught error: {e}", file=sys.stderr)

    print("Back to normal console output")

    # Method 2: Permanent setup
    print("\n=== Testing permanent setup ===")
    log_handle = setup_permanent_logging("permanent_log.txt")

    print("This is now permanently logged")
    print("All subsequent output goes to both console and file")

    # Clean up
    log_handle.close()
    restore_original_streams()

    print("Logging disabled - console only again")
