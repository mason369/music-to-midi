"""Hold the actual Windows metadata handle at the manifest replacement boundary."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.projects import store as storage
from src.projects.service import ProjectService


@pytest.mark.skipif(os.name != "nt", reason="Windows metadata handle sharing semantics")
@pytest.mark.parametrize("operation", ["service", "manifest_path"])
def test_metadata_probe_cannot_block_manifest_replacement(tmp_path, monkeypatch, operation):
    import ctypes
    import ntpath
    from ctypes import wintypes

    service = ProjectService(tmp_path)
    project_id = service.create("sharing")["storage_id"]
    store = service.store(project_id)
    document = store.load()
    original_stat = Path.stat
    original_final_path = ntpath._getfinalpathname
    original_access = storage.state_access
    main_thread = threading.get_ident()
    probe_ready = threading.Event()
    release_handle = threading.Event()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    )
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL

    def hold_metadata_handle(path, result):
        if threading.get_ident() == main_thread or str(path) != str(store.path):
            return result
        # CPython 3.11 win32_xstat_impl uses FILE_READ_ATTRIBUTES and share=0.
        # Keep precisely that kernel handle alive until the write is attempted.
        handle = kernel.CreateFileW(str(path), 0x80, 0, None, 3, 0x02000000, None)
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            probe_ready.set()
            assert release_handle.wait(10), "writer did not release metadata probe"
        finally:
            assert kernel.CloseHandle(handle)
        return result

    def observed_stat(path, *args, **kwargs):
        return hold_metadata_handle(path, original_stat(path, *args, **kwargs))

    def observed_final_path(path):
        return hold_metadata_handle(path, original_final_path(path))

    @contextmanager
    def observed_access(root):
        if threading.get_ident() != main_thread:
            probe_ready.set()
        with original_access(root):
            yield

    monkeypatch.setattr(Path, "stat", observed_stat)
    monkeypatch.setattr(ntpath, "_getfinalpathname", observed_final_path)
    monkeypatch.setattr(storage, "state_access", observed_access)

    def probe():
        if operation == "service":
            return service.store(project_id)
        result = storage.ProjectStore(store.path)
        probe_ready.set()
        return result

    with ThreadPoolExecutor(max_workers=1) as pool:
        with original_access(store.root):
            future = pool.submit(probe)
            try:
                assert probe_ready.wait(10), "probe did not reach metadata/lock boundary"
                document.name = "saved"
                storage.atomic_json(store.path, document.model_dump(mode="json"))
            finally:
                release_handle.set()
        assert future.result(timeout=10).root == store.root
    assert store.load().name == "saved"
