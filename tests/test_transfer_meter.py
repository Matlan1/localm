# SPDX-License-Identifier: AGPL-3.0-or-later
"""The byte counters behind download progress and speed: a rate averaged over a
window of the network byte count, and a progress bar for huggingface_hub that
counts into them."""

import inspect
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localm.model_manager import _transfer as tr


class TestRateWindow:
    def test_rate_is_the_byte_delta_over_the_window(self):
        w = tr.RateWindow(window=6.0)
        w.add(0, now=0.0)
        w.add(1000, now=1.0)
        w.add(3000, now=2.0)
        assert w.rate(now=2.0) == pytest.approx(1500.0)

    def test_rate_is_unknown_before_two_samples(self):
        w = tr.RateWindow()
        assert w.rate(now=1.0) is None
        w.add(10, now=1.0)
        assert w.rate(now=1.0) is None

    def test_a_stalled_transfer_decays_towards_zero_without_new_samples(self):
        w = tr.RateWindow(window=6.0)
        w.add(0, now=0.0)
        w.add(6000, now=1.0)
        fast = w.rate(now=1.0)
        later = w.rate(now=60.0)
        assert fast == pytest.approx(6000.0)
        assert later < fast / 10

    def test_old_samples_leave_the_window(self):
        w = tr.RateWindow(window=6.0)
        w.add(0, now=0.0)
        w.add(60000, now=1.0)
        for i in range(2, 30):
            w.add(60000 + (i - 1) * 100, now=float(i))
        assert w.rate(now=29.0) == pytest.approx(100.0, rel=0.2)


class TestTransferMeter:
    def test_counts_network_and_written_bytes_separately(self):
        m = tr.TransferMeter()
        m.add_transferred(100)
        m.add_written(40)
        assert (m.transferred, m.written) == (100, 40)

    def test_taking_bytes_back_never_goes_below_zero(self):
        m = tr.TransferMeter()
        m.add_transferred(10)
        m.add_transferred(-50)
        assert m.transferred == 0

    def test_reset_forgets_everything(self):
        m = tr.TransferMeter()
        m.add_transferred(10)
        m.reset()
        assert m.transferred == 0 and m.rate() is None and m.seconds_idle() is None

    def test_rate_is_unknown_until_a_byte_arrives(self):
        assert tr.TransferMeter().rate() is None


class TestMeterBar:
    @pytest.fixture(autouse=True)
    def _fresh(self):
        tr.METER.reset()
        yield
        tr.METER.reset()

    def test_a_file_bar_counts_written_bytes_on_update_and_network_bytes_on_update_transfer(self):
        bar = tr.MeterBar(desc="model.gguf", total=100, unit="B", initial=20)
        bar.update(30)
        bar.update_transfer(25)
        assert bar.n == 50
        assert (tr.METER.written, tr.METER.transferred) == (30, 25)

    def test_the_initial_offset_is_not_counted_as_a_download(self):
        tr.MeterBar(desc="model.gguf", unit="B", initial=500)
        assert (tr.METER.written, tr.METER.transferred) == (0, 0)

    def test_a_snapshot_transfer_bar_counts_update_as_network_bytes(self):
        bar = tr.MeterBar(desc="Downloading bytes", total=0, unit="B")
        bar.update(70)
        assert (tr.METER.written, tr.METER.transferred) == (0, 70)

    def test_a_bar_that_counts_files_is_ignored(self):
        bar = tr.MeterBar(desc="Fetching 3 files", total=3)
        bar.update(1)
        bar.update_transfer(1)
        assert (tr.METER.written, tr.METER.transferred) == (0, 0)

    def test_it_works_as_a_context_manager_and_swallows_display_calls(self):
        with tr.MeterBar(desc="x", unit="B") as bar:
            bar.set_postfix_str("1 MB/s", refresh=False)
            bar.set_transfer_postfix_str("1 MB/s", refresh=False)
            bar.set_description("done")
            bar.set_description_str("Download complete")
            bar.refresh()
        bar.close()


class _Blob(BaseHTTPRequestHandler):
    payload = b"x" * 3_000_000

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *a):
        pass


class TestAgainstTheRealHubClient:
    """The contract with the installed huggingface_hub, not a stand-in for it."""

    def test_the_bar_has_every_tqdm_method_the_hub_client_calls(self):
        import pathlib
        import re

        import huggingface_hub
        from tqdm import tqdm
        root = pathlib.Path(huggingface_hub.__file__).parent
        source = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in root.rglob("*.py"))
        tqdm_methods = {name for name, value in vars(tqdm).items()
                        if callable(value) and not name.startswith("_")}
        called = {name for name in tqdm_methods
                  if re.search(rf"\w*(?:bar|progress)\.{name}\(", source)}
        assert "set_description_str" in called and "update" in called
        missing = sorted(name for name in called if not callable(getattr(tr.MeterBar, name, None)))
        assert missing == []

    def test_http_get_feeds_the_meter_through_the_bar(self, tmp_path):
        from huggingface_hub.file_download import http_get
        tr.METER.reset()
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Blob)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/blob"
            with open(tmp_path / "out.bin", "wb") as f:
                http_get(url, f, expected_size=len(_Blob.payload),
                         tqdm_class=tr.meter_tqdm_class())
        finally:
            server.shutdown()
            server.server_close()
        assert tr.METER.written == len(_Blob.payload)
        assert tr.METER.transferred == len(_Blob.payload)
        tr.METER.reset()

    def test_the_snapshot_network_bar_is_still_told_apart_by_its_description(self):
        import huggingface_hub._snapshot_download as sd
        src = inspect.getsource(sd)
        assert 'desc="Downloading bytes"' in src, (
            "snapshot_download no longer names its network bar 'Downloading bytes'; "
            "MeterBar would count its network bytes as written bytes")
        assert "bar.update(inc)" in inspect.getsource(
            __import__("huggingface_hub.utils._xet_progress_reporting",
                       fromlist=["x"]))


class TestFormatting:
    def test_rate(self):
        assert tr.format_rate(None) == "--"
        assert tr.format_rate(500) == "500 B/s"
        assert tr.format_rate(22_000_000) == "22.0 MB/s"

    def test_duration(self):
        assert tr.format_duration(None) == "--"
        assert tr.format_duration(42) == "42s"
        assert tr.format_duration(185) == "3m 05s"
        assert tr.format_duration(3720) == "1h 02m"

    def test_eta_needs_a_total_and_a_positive_rate(self):
        assert tr.eta_seconds(100, 1100, 100.0) == 10
        assert tr.eta_seconds(100, 0, 100.0) is None
        assert tr.eta_seconds(100, 1100, 0.0) is None
        assert tr.eta_seconds(100, 1100, None) is None
        assert tr.eta_seconds(2000, 1100, 100.0) is None

    def test_describe(self):
        text = tr.describe(167_100_000, 491_400_000, 22_000_000.0, "m.gguf")
        assert text == "m.gguf: 34% (167.1 MB / 491.4 MB) at 22.0 MB/s, 15s left"
        assert tr.describe(5_000_000, 0, None) == "5.0 MB"
