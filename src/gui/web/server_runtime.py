"""Shared server-runtime configuration for the browser interfaces."""

from __future__ import annotations


UVICORN_WEBSOCKET_PROTOCOL = "websockets-sansio"


def configure_uvicorn_websocket_protocol() -> str:
    """Require Uvicorn's maintained Sans-I/O WebSocket implementation.

    Gradio creates ``uvicorn.Config`` internally with ``ws="auto"``. Uvicorn
    otherwise prefers its legacy ``websockets`` adapter whenever that package
    is installed, which imports APIs deprecated by current websockets releases.
    Updating the registered ``auto`` target keeps Gradio on Uvicorn's supported
    Sans-I/O implementation. Missing support is a hard startup error.
    """

    from uvicorn.config import WS_PROTOCOLS

    target = WS_PROTOCOLS.get(UVICORN_WEBSOCKET_PROTOCOL)
    if target is None:
        raise RuntimeError(
            "The installed Uvicorn runtime does not provide the required "
            f"{UVICORN_WEBSOCKET_PROTOCOL!r} WebSocket protocol"
        )
    WS_PROTOCOLS["auto"] = target
    return UVICORN_WEBSOCKET_PROTOCOL


__all__ = ["UVICORN_WEBSOCKET_PROTOCOL", "configure_uvicorn_websocket_protocol"]
