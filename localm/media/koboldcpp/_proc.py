# SPDX-License-Identifier: AGPL-3.0-or-later
"""Starting and killing the KoboldCpp process so it never outlives localm.

Windows: the process is assigned to a job object created with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``; only localm holds the job handle, so the
process (and anything it started) is killed when localm exits for any reason,
and :func:`kill` terminates the whole job. POSIX: the process leads its own
session, and a small watcher process kills that session when localm's PID goes
away. Only the process localm started is ever signalled.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Optional

_WATCHER = r"""
import os, signal, sys, time
parent, child = int(sys.argv[1]), int(sys.argv[2])
def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
while alive(child):
    if os.getppid() != parent:
        try:
            os.killpg(child, signal.SIGKILL)
        except OSError:
            try:
                os.kill(child, signal.SIGKILL)
            except OSError:
                pass
        break
    time.sleep(1.0)
"""


class ManagedProcess:
    """A started KoboldCpp process and what is needed to kill exactly it."""

    def __init__(self, proc: subprocess.Popen, job=None, watcher=None) -> None:
        self.proc = proc
        self._job = job
        self._watcher = watcher

    @property
    def pid(self) -> int:
        return self.proc.pid

    def poll(self) -> Optional[int]:
        return self.proc.poll()


def _win_job_for(proc: subprocess.Popen):
    """A kill-on-close job object holding *proc*, or None when it cannot be
    created (the process then still dies through :func:`kill`, but not when
    localm is killed outright)."""
    import ctypes
    from ctypes import wintypes

    class _BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC),
                    ("IoInfo", _IO),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    if ok and kernel32.AssignProcessToJobObject(job, int(proc._handle)):
        return job
    kernel32.CloseHandle(job)
    return None


def start(argv: list, *, cwd: str, env: dict) -> ManagedProcess:
    """Start *argv* with stdout and stderr merged into one text pipe, tied to
    localm's lifetime as described in the module docstring."""
    kwargs = dict(cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                  bufsize=1)
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        proc = subprocess.Popen(argv, **kwargs)
        job = None
        try:
            job = _win_job_for(proc)
        except (OSError, AttributeError, ValueError):
            job = None
        if job is None:
            from localm.debuglog import logger
            logger.warning("koboldcpp: could not tie pid %s to a kill-on-close job; it is "
                           "stopped with localm's normal shutdown only", proc.pid)
        return ManagedProcess(proc, job=job)
    kwargs["start_new_session"] = True
    proc = subprocess.Popen(argv, **kwargs)
    watcher = None
    try:
        watcher = subprocess.Popen(
            [sys.executable, "-I", "-c", _WATCHER, str(os.getpid()), str(proc.pid)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=True)
    except OSError as e:
        from localm.debuglog import logger
        logger.warning("koboldcpp: could not start the exit watcher for pid %s (%s); it is "
                       "stopped with localm's normal shutdown only", proc.pid, e)
    return ManagedProcess(proc, watcher=watcher)


def kill(mp: ManagedProcess, *, grace: float = 5.0) -> None:
    """Stop the process localm started and everything it started, then reap it.
    Never raises and never blocks on the output pipe: a reader of ``proc.stdout``
    ends at end of file once every process holding the pipe has exited. See
    test_stop_kills_what_the_server_started."""
    proc = mp.proc
    if sys.platform == "win32":
        if mp._job is not None:
            try:
                import ctypes
                from ctypes import wintypes
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
                kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                kernel32.TerminateJobObject(mp._job, 1)
                kernel32.CloseHandle(mp._job)
            except OSError:
                pass
            mp._job = None
        if proc.poll() is None:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, timeout=15)
            except (OSError, subprocess.SubprocessError):
                pass
    else:
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = None
        if pgid is not None and (pgid <= 1 or pgid == os.getpgrp()):
            pgid = None
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if proc.poll() is not None:
                break
            try:
                if pgid is not None:
                    os.killpg(pgid, sig)
                else:
                    proc.send_signal(sig)
            except OSError:
                pass
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                continue
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    if mp._watcher is not None:
        try:
            mp._watcher.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                mp._watcher.kill()
            except OSError:
                pass
        mp._watcher = None
