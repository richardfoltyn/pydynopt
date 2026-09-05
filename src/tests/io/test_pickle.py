import importlib.util
import pickle

import pytest

from pydynopt.io.pickle import dump, get_cached_object, load


def test_atomic_dump_preserves_existing_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "cache"
    dump(path, {"value": "old"}, compress=False)

    def fail_dump(*args, **kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(pickle, "dump", fail_dump)

    with pytest.raises(RuntimeError, match="serialization failed"):
        dump(path, {"value": "new"}, compress=False, atomic=True)

    assert load(path) == {"value": "old"}
    assert list(tmp_path.glob(".cache.*.tmp")) == []


@pytest.mark.parametrize(
    "suffix",
    [
        ".gz",
        ".xz",
        "",
        pytest.param(
            ".lz4",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("lz4") is None,
                reason="lz4 is not installed",
            ),
        ),
        pytest.param(
            ".zst",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("pyzstd") is None,
                reason="pyzstd is not installed",
            ),
        ),
        pytest.param(
            ".zstd",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("pyzstd") is None,
                reason="pyzstd is not installed",
            ),
        ),
    ],
)
def test_atomic_dump_roundtrip(tmp_path, suffix):
    path = tmp_path / f"cache{suffix}"
    obj = {"value": [1, 2, 3]}

    dump(path, obj, compress=bool(suffix), atomic=True)

    assert load(path) == obj


def test_corrupt_cache_is_recomputed(tmp_path, caplog):
    path = tmp_path / "cache.xz"
    path.write_bytes(b"not a pickle")
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return {"value": "new"}

    assert get_cached_object(compute, cache_file=path) == {"value": "new"}
    assert calls == 1
    assert load(path) == {"value": "new"}
    assert "corrupt" in caplog.text
