import sys
import os

if __name__ == "__main__":
    sys.exit(
        "Error: You are trying to run 'localm' directly with the system Python, "
        "but it must be run from its virtual environment.\n"
        "Please use '.venv\\Scripts\\localm' or activate the venv first."
    )
