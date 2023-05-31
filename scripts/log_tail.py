"""
Script to follow tail of latest log file on Windows
using gitbash.
"""
import subprocess
import dstnx

def run_tail(filename):
    try:
        tail_process = subprocess.Popen(['tail', '-f', filename], stdout=subprocess.PIPE)
        for line in iter(tail_process.stdout.readline, b''):
            try:
                print(line.decode("latin-1").rstrip())  # Process the output as needed
            except UnicodeDecodeError:
                print(line)
    except FileNotFoundError:
        print(f"File '{filename}' not found.")


def get_latest_file(files) -> str:
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return latest_file


if __name__ == "__main__":
    file = get_latest_file(dstnx.fp.LOG_DST.glob("*"))
    if file:
        print(f"Following {file}")
        run_tail(file)