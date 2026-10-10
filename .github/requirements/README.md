# CI requirement sets

The workflows install Python packages only with `pip install --require-hashes`.

- Most jobs build the set at run time: `uv export --locked --no-emit-project --extra ...`
  writes the hash-locked list for the extras the job needs from the committed `uv.lock`,
  and `pip install --require-hashes --no-deps -r` installs it. There is nothing to
  regenerate; Dependabot's `uv` updates keep `uv.lock` current.
- Two sets are committed because `uv.lock` does not hold them:
  - `torch-cpu.txt`: the optional-stacks job needs the CPU build of torch, while `uv.lock`
    resolves the CUDA wheels, so that job exports with `--prune torch` and installs this
    file first. The file leaves out `setuptools`, which comes from `uv.lock`. Dependabot
    ignores `torch`, so it changes only by hand.
  - `mutmut.txt`: mutmut and its dependencies, resolved against the versions `uv.lock`
    gives the dev extra so the shared packages agree. The mutation jobs install it after
    the dev set. Adding mutmut to `uv.lock` would make every test file count as affected
    by the change, which the per-PR gate refuses.

Regenerate from the repository root (delete `mutmut-constraints.txt` afterwards):

```bash
printf 'torch==2.13.0+cpu\n' | uv pip compile - --no-config --no-emit-package setuptools --generate-hashes --emit-index-url \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match -o .github/requirements/torch-cpu.txt

uv export --quiet --locked --no-emit-project --extra dev --no-hashes --no-header -o mutmut-constraints.txt
printf 'mutmut==3.7.0\n' | uv pip compile - --no-config -c mutmut-constraints.txt --generate-hashes \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 -o .github/requirements/mutmut.txt
```

`scripts/check_ci_requirements.py` runs the two commands above into a temporary
directory and compares the result with the committed files, and looks up every pinned
version in the OSV database. The `requirements currency` workflow runs it weekly and
on pushes to master that touch the sets; a red run means regenerate the set (or move
the pin past the advisory) with the commands above. Dependabot does not manage these
files: `torch-cpu.txt` must stay on the CPU index build and `mutmut.txt` must agree
with `uv.lock`, which an independent bump of a shared package would break.
