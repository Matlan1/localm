# SPDX-License-Identifier: AGPL-3.0-or-later
"""nvidia_preflight() must not return a silent blank for a field nvidia-smi can
answer explicitly.

A 5-GPU report showed an empty driver version and CUDA capability while the
explicit --query-gpu calls for the GPU name and compute capability worked, so
the banner regex was the only thing that failed. The driver version now comes
from --query-gpu=driver_version, the CUDA version falls back to
``nvidia-smi -q`` when the banner lacks it, every GPU's memory is listed, and a
field that still cannot be parsed logs the raw output at debug.
"""

import logging

import localm.setup_llama as sl
from localm import bugreport

_Q_OUTPUT = """
==============NVSMI LOG==============

Timestamp                                 : Thu Sep 25 14:08:00 2026
Driver Version                            : 595.44.02
CUDA Version                              : 13.2

Attached GPUs                             : 5
"""

_MEMORY = """0, NVIDIA GeForce RTX 4080, 16376, 14030
1, NVIDIA GeForce RTX 3090, 24576, 22430
2, NVIDIA GeForce RTX 3090, 24576, 22430
3, NVIDIA RTX A4000, 16376, 14746
4, NVIDIA GeForce GTX 1660, 6144, 2765
"""


def _fake(banner, *, driver_query="595.44.02\n", q=_Q_OUTPUT, memory=_MEMORY):
    calls = []

    def fake_smi(*args):
        calls.append(args)
        joined = " ".join(args)
        if "--query-gpu=driver_version" in joined:
            return driver_query
        if "--query-gpu=index,name,memory.total,memory.free" in joined:
            return memory
        if "--query-gpu=name" in joined:
            return "NVIDIA GeForce RTX 4080\nNVIDIA GeForce RTX 3090\n"
        if "compute_cap" in joined:
            return "8.9\n8.6\n"
        if args == ("-q",):
            return q
        return banner
    return fake_smi, calls


def test_driver_version_comes_from_the_explicit_query(monkeypatch):
    # Neither the banner nor -q carries a driver version: only the query can.
    fake, _ = _fake("| NVIDIA-SMI |\n",
                    q="CUDA Version                              : 13.2\n")
    monkeypatch.setattr(sl, "_nvidia_smi", fake)
    info = sl.nvidia_preflight()
    assert info.driver_version == "595.44.02"


def test_cuda_version_falls_back_to_the_long_query(monkeypatch):
    fake, calls = _fake("| NVIDIA-SMI 595.44.02 |\n")
    monkeypatch.setattr(sl, "_nvidia_smi", fake)
    info = sl.nvidia_preflight()
    assert info.cuda_capability == "13.2"
    assert ("-q",) in calls


def test_a_complete_banner_needs_no_long_query(monkeypatch):
    fake, calls = _fake("| NVIDIA-SMI 552.22   Driver Version: 552.22   "
                        "CUDA Version: 12.4  |\n")
    monkeypatch.setattr(sl, "_nvidia_smi", fake)
    info = sl.nvidia_preflight()
    assert info.cuda_capability == "12.4"
    assert ("-q",) not in calls


def test_a_rejected_driver_query_falls_back_to_the_banner(monkeypatch):
    fake, _ = _fake("| NVIDIA-SMI 470.182.03   Driver Version: 470.182.03   "
                    "CUDA Version: 11.4  |\n",
                    driver_query='Field "driver_version" is not a valid field to query.\n')
    monkeypatch.setattr(sl, "_nvidia_smi", fake)
    info = sl.nvidia_preflight()
    assert info.driver_version == "470.182.03"
    assert info.cuda_capability == "11.4"


def test_every_gpu_is_listed_with_its_memory(monkeypatch):
    fake, _ = _fake("| NVIDIA-SMI |\n")
    monkeypatch.setattr(sl, "_nvidia_smi", fake)
    info = sl.nvidia_preflight()
    assert [g["index"] for g in info.gpus] == [0, 1, 2, 3, 4]
    assert info.gpus[4] == {"index": 4, "name": "NVIDIA GeForce GTX 1660",
                            "total_mib": 6144, "free_mib": 2765}


def test_an_unparsed_field_logs_the_raw_output(monkeypatch, caplog):
    fake, _ = _fake("| SOMETHING UNEXPECTED 595 |\n", driver_query="",
                    q="no useful lines\n")
    monkeypatch.setattr(sl, "_nvidia_smi", fake)
    with caplog.at_level(logging.DEBUG, logger="localm"):
        info = sl.nvidia_preflight()
    assert info.present
    assert info.driver_version == "" and info.cuda_capability == ""
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "could not parse driver version, CUDA version" in logged
    assert "SOMETHING UNEXPECTED 595" in logged
    assert "no useful lines" in logged


def test_the_report_lists_nvidia_gpus_and_names_missing_fields(monkeypatch):
    from localm import hwdetect
    det = hwdetect.Detection(vendors=["nvidia"], recommended="cuda",
                             source="test", gpu_names="nvidia")
    monkeypatch.setattr(hwdetect, "detect", lambda: det)
    fake, _ = _fake("| NVIDIA-SMI |\n", driver_query="", q="")
    monkeypatch.setattr(sl, "_nvidia_smi", fake)
    diag = bugreport.collect_diagnostics({})
    assert diag["nvidia_driver"] == "not detected"
    assert diag["nvidia_cuda_capability"] == "not detected"
    assert diag["nvidia_gpus"][4] == "4: NVIDIA GeForce GTX 1660 (2.7 of 6.0 GB free)"
    assert len(diag["nvidia_gpus"]) == 5
