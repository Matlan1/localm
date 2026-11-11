# CI requirement sets

The workflows install their dependencies only with `pip install --require-hashes`, except
`macos-smoke.yml`, which runs the user-facing `setup.sh` install on purpose. Build tools
are hashed too: every job that installs the project first installs `editable.txt`, then
installs its set and the project's editable install (`pip install --no-deps
--no-build-isolation -e .`) without build isolation, so pip builds with those locked
tools instead of fetching an unpinned `hatchling`, `editables` or `setuptools`. The same
`setuptools` builds the few dependencies that ship only as a source archive (Distance,
docopt, proxy_tools).

- Most jobs build the set at run time: `uv export --locked --no-emit-project --extra ...`
  writes the hash-locked list for the extras the job needs from the committed `uv.lock`,
  and `pip install --require-hashes --no-deps -r` installs it. There is nothing to
  regenerate; Dependabot's `uv` updates keep `uv.lock` current.
- Four sets are committed because `uv.lock` does not hold them:
  - `torch-cpu.txt`: the lint and optional-stacks jobs need the CPU builds of torch and torchvision, while `uv.lock`
    resolves the CUDA wheels, so those jobs export with `--prune torch` and install this
    file first. The file leaves out `setuptools`, which comes from `uv.lock`. Dependabot
    ignores both, so they change only by hand, and `torchvision` moves to the release
    paired with the torch pin.
  - `mutmut.txt`: mutmut and its dependencies, resolved against the versions `uv.lock`
    gives the dev extra so the shared packages agree. The mutation jobs install it after
    the dev set. Adding mutmut to `uv.lock` would make every test file count as affected
    by the change, which the per-PR gate refuses.
  - `build.txt`: `build`, the `hatchling` build backend and `twine`, the toolchain that
    produces and checks the sdist and wheel. `publish-pypi.yml` and `release-attest.yml`
    both install it with `--require-hashes` and build with `python -m build
    --no-isolation`, so both builds run the same locked packages. In those two
    workflows the backend named in `pyproject.toml` is never resolved at build time.
  - `editable.txt`: the `hatchling` backend (the same version as `build.txt`),
    `editables` and `setuptools` (the version `uv.lock` holds), resolved against the
    versions `uv.lock` gives the dev extra so the shared packages agree. The jobs that
    install the project install it before anything else.

Regenerate from the repository root (to move a pin, edit its version in the command; delete `mutmut-constraints.txt` and `editable-constraints.txt` afterwards):

```bash
printf 'torch==2.13.0+cpu\ntorchvision==0.28.0+cpu\n' | uv pip compile - --no-config --no-emit-package setuptools --generate-hashes --emit-index-url \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match -o .github/requirements/torch-cpu.txt

uv export --quiet --locked --no-emit-project --extra dev --no-hashes --no-header -o mutmut-constraints.txt
printf 'mutmut==3.7.0\n' | uv pip compile - --no-config -c mutmut-constraints.txt --generate-hashes \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 -o .github/requirements/mutmut.txt

printf 'build==1.6.1\nhatchling==1.32.4\ntwine==7.0.0\n' | uv pip compile - --no-config --generate-hashes \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 -o .github/requirements/build.txt

uv export --quiet --locked --no-emit-project --extra dev --no-hashes --no-header -o editable-constraints.txt
printf 'editables==0.5\nhatchling==1.32.4\nsetuptools==84.2.1\n' | uv pip compile - --no-config -c editable-constraints.txt --generate-hashes \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 -o .github/requirements/editable.txt
```

`scripts/check_ci_requirements.py` runs the commands above into a temporary
directory and compares the result with the committed files, and looks up every pinned
version in the OSV database. The `requirements currency` workflow runs it weekly and
on pushes to master that touch the sets; a red run means regenerate the set (or move
the pin past the advisory) with the commands above. Dependabot does not manage these
files: `torch-cpu.txt` must stay on the CPU index builds, and `mutmut.txt` and `editable.txt` must agree
with `uv.lock`, which an independent bump of a shared package would break.
