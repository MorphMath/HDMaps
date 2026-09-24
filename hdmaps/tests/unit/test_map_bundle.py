import numpy as np
import pytest
import scipy.sparse as sp

from hdmaps.mappings import DirStorage, MapBundle, PackedStorage

DATA = [1, 2, 3]
DIAG = np.eye(3, dtype=bool)


class CountingMap:
    def __init__(self, dtype: type[np.generic] = np.float64):
        self.dtype = dtype
        self.calls = 0

    def __call__(self, a, b):
        self.calls += 1
        return sp.csr_matrix(np.eye(2, dtype=self.dtype) * (a + b))


STORAGES = {
    "memory": lambda tmp_path: None,
    "dir": lambda tmp_path: DirStorage(str(tmp_path / "maps")),
    "packed": lambda tmp_path: PackedStorage(str(tmp_path / "maps.npz")),
}


@pytest.fixture(params=STORAGES.values(), ids=list(STORAGES))
def make_storage(request, tmp_path):
    return lambda: request.param(tmp_path)


def test_maps_match_compute(make_storage):
    bundle = MapBundle(CountingMap(), DATA, storage=make_storage())
    for i in range(3):
        for j in range(3):
            assert bundle[i, j][0, 0] == DATA[i] + DATA[j]


def test_from_maps_matches_constructor(make_storage):
    maps = {(i, j): CountingMap()(a, b) for i, a in enumerate(DATA) for j, b in enumerate(DATA)}
    bundle = MapBundle.from_maps(maps, DATA, storage=make_storage())
    for (i, j), m in maps.items():
        np.testing.assert_array_equal(bundle[i, j].toarray(), m.toarray())


def test_masked_pair_raises(make_storage):
    bundle = MapBundle(CountingMap(), DATA, storage=make_storage(), mask=DIAG)
    with pytest.raises(LookupError):
        bundle[0, 1]


def test_mask_shape_mismatch_raises():
    with pytest.raises(ValueError):
        MapBundle(CountingMap(), DATA, mask=np.eye(2, dtype=bool))


@pytest.mark.parametrize("storage", ["dir", "packed"])
def test_reload_does_not_recompute(storage, tmp_path):
    MapBundle(CountingMap(), DATA, storage=STORAGES[storage](tmp_path))
    f = CountingMap()
    MapBundle(f, DATA, storage=STORAGES[storage](tmp_path))
    assert f.calls == 0


def test_packed_prunes_stale_maps(tmp_path):
    path = str(tmp_path / "maps.npz")
    MapBundle(CountingMap(), DATA, storage=PackedStorage(path))
    MapBundle(CountingMap(), DATA, storage=PackedStorage(path), mask=DIAG)
    assert len(np.load(path)["keys"]) == 3


def test_packed_appends_npz_suffix(tmp_path):
    MapBundle(CountingMap(), DATA, storage=PackedStorage(str(tmp_path / "maps")))
    f = CountingMap()
    MapBundle(f, DATA, storage=PackedStorage(str(tmp_path / "maps")))
    assert f.calls == 0


def test_packed_preserves_dtypes(tmp_path):
    path = str(tmp_path / "maps.npz")
    maps = {(0, 0): CountingMap(np.float32)(1, 1), (1, 1): CountingMap(np.float64)(2, 2)}
    MapBundle.from_maps(maps, DATA[:2], storage=PackedStorage(path), mask=np.eye(2, dtype=bool))
    bundle = MapBundle(CountingMap(), DATA[:2], storage=PackedStorage(path), mask=np.eye(2, dtype=bool))
    assert bundle[0, 0].dtype == np.float32
    assert bundle[1, 1].dtype == np.float64


def test_packed_empty_mask_keeps_file(tmp_path):
    path = str(tmp_path / "maps.npz")
    MapBundle(CountingMap(), DATA, storage=PackedStorage(path))
    MapBundle(CountingMap(), DATA, storage=PackedStorage(path), mask=np.zeros((3, 3), dtype=bool))
    assert len(np.load(path)["keys"]) == 9
