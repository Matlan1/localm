# SPDX-License-Identifier: AGPL-3.0-or-later
"""A fake ``subprocess.Popen`` that localm's background threads cannot reach."""
import subprocess
import threading


def ignore_background_threads(fake):
    """Return a ``subprocess.Popen`` replacement that sends a call to the real
    ``subprocess.Popen`` when it comes from a thread that already existed when
    this was built (other than the calling thread) or from a thread whose name
    starts with ``localm-``, and every other call to *fake*.

    Build it inside the test, before ``subprocess.Popen`` is patched: the real
    class and the set of existing threads are captured at that moment. A GPU or
    VRAM probe an earlier test started ("localm-vram-probe", and the
    "localm-gpu-probe" thread it starts) that spawns a process while the patch
    is active therefore neither reaches *fake* nor shows up in what it records,
    while the calling thread and the threads the test starts do reach it.
    See test_a_background_probe_never_reaches_the_fake.
    """
    real = subprocess.Popen
    caller = threading.current_thread()
    earlier = {t for t in threading.enumerate() if t is not caller}

    def popen(*args, **kwargs):
        current = threading.current_thread()
        if current in earlier or current.name.startswith("localm-"):
            return real(*args, **kwargs)
        return fake(*args, **kwargs)

    return popen
