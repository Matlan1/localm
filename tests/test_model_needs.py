# SPDX-License-Identifier: AGPL-3.0-or-later
"""The reviewed model-needs list: the shipped file is well formed and every
malformed or unconfirmed document is refused."""

from __future__ import annotations

import copy
import json

import pytest

from localm.model_manager import model_needs
from localm.model_manager.missing_packages import SHIPPED_MODULES
from localm.model_manager.model_needs import ModelNeedsError, Need

ROW = {"scope": "custom_code", "part": "model", "module": "fla", "project": "fla-core",
       "min_version": "0.6.0", "platforms": ["linux-x64", "win-x64"], "status": "confirmed",
       "evidence": ["pip install line in code of moonshotai/Kimi-Linear-48B-A3B-Instruct"],
       "repos": 2, "of": 2}
DOC = {"version": 1, "collected": "2026-11-04", "transformers": "5.20.2",
       "types": {"kimi_linear": [ROW]}}


def doc_with(**changes) -> dict:
    doc = copy.deepcopy(DOC)
    doc["types"]["kimi_linear"][0].update(changes)
    return doc


class TestShippedFile:
    def test_loads_and_every_entry_is_confirmed(self):
        needs = model_needs.load()
        assert needs
        for model_type, rows in needs.items():
            assert rows, model_type
            for n in rows:
                assert isinstance(n, Need)
                assert n.evidence

    def test_no_entry_names_a_module_localm_ships(self):
        for model_type, rows in model_needs.load().items():
            for n in rows:
                assert n.module not in SHIPPED_MODULES, (model_type, n.module)

    def test_on_demand_modules_use_the_project_localm_already_reviewed(self):
        from localm.model_manager.model_packages import ON_DEMAND, REFUSED
        for model_type, rows in model_needs.load().items():
            for n in rows:
                assert n.module not in REFUSED, (model_type, n.module)
                if n.module in ON_DEMAND:
                    assert n.project == ON_DEMAND[n.module].project.lower().replace("_", "-")

    def test_file_is_valid_utf8_json_without_dashes(self):
        raw = model_needs.DATA_FILE.read_bytes().decode("utf-8")
        assert chr(0x2014) not in raw and chr(0x2013) not in raw
        json.loads(raw)

    def test_needs_for_reads_the_shipped_list(self):
        first = next(iter(model_needs.load()))
        assert model_needs.needs_for(first) == model_needs.load()[first]
        assert model_needs.needs_for("no-such-model-type") == ()
        assert model_needs.needs_for(first, scope="no-such-scope") == ()


class TestParse:
    def test_valid_document(self):
        needs = model_needs.parse(DOC)
        assert needs["kimi_linear"] == (Need(
            "custom_code", "model", "fla", "fla-core", "0.6.0", ("linux-x64", "win-x64"),
            tuple(ROW["evidence"]), 2, 2),)

    def test_source_only_package_has_no_platforms(self):
        assert model_needs.parse(doc_with(platforms=[]))["kimi_linear"][0].platforms == ()

    def test_native_row_needs_no_counts(self):
        row = {k: v for k, v in ROW.items() if k not in ("repos", "of")} | {"scope": "native"}
        doc = copy.deepcopy(DOC)
        doc["types"]["kimi_linear"] = [row]
        assert model_needs.parse(doc)["kimi_linear"][0].repos is None

    @pytest.mark.parametrize("status", ["unconfirmed", "likely", "", None, True])
    def test_unconfirmed_entry_is_refused(self, status):
        with pytest.raises(ModelNeedsError, match="only confirmed|keys must be"):
            model_needs.parse(doc_with(status=status))

    def test_entry_without_status_is_refused(self):
        doc = copy.deepcopy(DOC)
        del doc["types"]["kimi_linear"][0]["status"]
        with pytest.raises(ModelNeedsError):
            model_needs.parse(doc)

    @pytest.mark.parametrize("field,value", [
        ("scope", "everywhere"), ("part", "weights"), ("module", "not a module"),
        ("module", "a.b"), ("module", ""), ("project", "Fla_Core"), ("project", "fla core"),
        ("project", "git+https://example.com/x"), ("project", "../x"), ("project", ""),
        ("project", "-x"), ("project", "x-"), ("min_version", "latest"), ("min_version", ">=1"),
        ("min_version", ""), ("min_version", "1.0; import os"),
        ("platforms", ["linux-arm64"]), ("platforms", ["linux-x64", "linux-x64"]),
        ("platforms", "linux-x64"), ("evidence", []), ("evidence", [""]), ("evidence", "x"),
        ("repos", 0), ("repos", True), ("repos", 3), ("of", 0), ("repos", "2"),
    ])
    def test_bad_field_is_refused(self, field, value):
        with pytest.raises(ModelNeedsError):
            model_needs.parse(doc_with(**{field: value}))

    def test_extra_or_missing_entry_key_is_refused(self):
        with pytest.raises(ModelNeedsError):
            model_needs.parse(doc_with(url="https://example.com"))
        doc = copy.deepcopy(DOC)
        del doc["types"]["kimi_linear"][0]["project"]
        with pytest.raises(ModelNeedsError):
            model_needs.parse(doc)

    def test_repos_without_of_is_refused(self):
        doc = copy.deepcopy(DOC)
        del doc["types"]["kimi_linear"][0]["of"]
        with pytest.raises(ModelNeedsError, match="go together"):
            model_needs.parse(doc)

    def test_duplicate_package_for_one_type_is_refused(self):
        doc = copy.deepcopy(DOC)
        doc["types"]["kimi_linear"].append(dict(ROW))
        with pytest.raises(ModelNeedsError, match="twice"):
            model_needs.parse(doc)

    @pytest.mark.parametrize("change", [
        lambda d: d.update(version=2), lambda d: d.update(version=True),
        lambda d: d.update(collected="yesterday"), lambda d: d.update(transformers=""),
        lambda d: d.update(types=[]), lambda d: d.update(extra=1),
        lambda d: d.pop("types"), lambda d: d["types"].update({"x": []}),
        lambda d: d["types"].update({"x": {}}), lambda d: d["types"].update({"": [ROW]}),
        lambda d: d["types"].update({"x": ["fla"]}),
    ])
    def test_bad_document_is_refused(self, change):
        doc = copy.deepcopy(DOC)
        change(doc)
        with pytest.raises(ModelNeedsError):
            model_needs.parse(doc)

    @pytest.mark.parametrize("doc", [None, [], "text", 1])
    def test_non_object_document_is_refused(self, doc):
        with pytest.raises(ModelNeedsError):
            model_needs.parse(doc)


class TestLoad:
    def test_malformed_files_are_refused(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(ModelNeedsError, match="cannot be read"):
            model_needs.load(bad)
        with pytest.raises(ModelNeedsError, match="cannot be read"):
            model_needs.load(tmp_path / "missing.json")

    def test_unconfirmed_entry_in_a_file_is_refused(self, tmp_path):
        path = tmp_path / "needs.json"
        path.write_text(json.dumps(doc_with(status="unconfirmed")), encoding="utf-8")
        with pytest.raises(ModelNeedsError, match="only confirmed"):
            model_needs.load(path)

    def test_valid_file_round_trips(self, tmp_path):
        path = tmp_path / "needs.json"
        path.write_text(json.dumps(DOC), encoding="utf-8")
        assert model_needs.load(path)["kimi_linear"][0].project == "fla-core"
