import os
import sys

if __name__ == "__main__":
    launcher = r".venv\Scripts\localm" if os.name == "nt" else ".venv/bin/localm"
    sys.exit(
        "Error: localm is not started by running this file. "
        f"Use '{launcher}' from the install folder, or activate its virtual "
        "environment and run 'localm'.")
