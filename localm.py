import sys

if __name__ == "__main__":
    _cli = (".venv\\Scripts\\localm" if sys.platform == "win32"
            else ".venv/bin/localm")
    sys.exit(
        "Error: this Python is not localm's own environment, and localm must be "
        "run from it.\n"
        f"Use '{_cli}' in the install folder, or activate that venv first.")
