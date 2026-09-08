"""Publish verified project assets through Gradio's content-addressed cache."""

from pathlib import Path


def cache_project_file(path: str | Path) -> str:
    from gradio.context import Context, LocalContext
    from gradio.processing_utils import save_file_to_cache

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    blocks = LocalContext.blocks.get(None) or Context.root_block
    if blocks is None:
        raise RuntimeError("项目文件只能在当前 Gradio 应用中发布")
    # Gradio rejects Windows device namespaces in public file URLs. Keep the
    # verified original in the project and serve its cached, ordinary path.
    cached = save_file_to_cache(source, blocks.GRADIO_CACHE)
    blocks.temp_files.add(cached)
    return cached
