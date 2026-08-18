# 独立 Web 前端

此目录是纯静态浏览器客户端。它不导入 Python 推理代码，也不依赖 Gradio 事件队列；所有操作通过 `src.web_api` 提供的版本化 REST + SSE 契约完成。

推荐从仓库根目录运行：

```powershell
.\run_web.ps1
```

也可以分别启动两个进程：

```powershell
.\venv\Scripts\python.exe -m src.web_api --host 127.0.0.1 --port 8765
.\venv\Scripts\python.exe -m http.server 5173 --bind 127.0.0.1 --directory web
```

浏览器打开 `http://127.0.0.1:5173`。后端 OpenAPI 文档位于 `http://127.0.0.1:8765/docs`。

生产部署时应把 `web/` 交给静态站点服务器，并通过 `MUSIC_TO_MIDI_ALLOWED_ORIGINS` 显式配置允许的前端 Origin。后端默认只接受本机 5173/8765 Origin，不开放通配 CORS。
