#!/usr/bin/env python
"""
app.py - name the character in cosplay photos, in your browser.

    pip install -r requirements.txt
    python app.py

Downloads the model from Hugging Face the first time (about 180 MB, kept in
Hugging Face's cache), then opens http://localhost:5000: drop photos on the
page and each one gets the model's guesses. Nothing leaves your computer
except that one download.

    python app.py --port 8000 --no-browser
    python app.py --model local-dir:path/to/a/downloaded/model
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import timm
import torch
from PIL import Image, ImageOps

MODEL = "hf-hub:GwenRB/cosplay-characters"
PAGE = Path(__file__).with_name("index.html")
MAX_BYTES = 40 * 1024 * 1024
TOP = 5


class Identifier:
    def __init__(self, name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The weights are half precision; create_model loads them into float32.
        self.model = timm.create_model(name, pretrained=True).eval().to(self.device)
        # The model's config squashes the whole photo to a square, as in training.
        self.transform = timm.data.create_transform(**timm.data.resolve_data_config(model=self.model))
        self.names = self.model.pretrained_cfg["label_names"]
        self.lock = threading.Lock()   # one photo at a time on the GPU

    def guess(self, data: bytes) -> list[dict]:
        with Image.open(io.BytesIO(data)) as im:
            x = self.transform(ImageOps.exif_transpose(im).convert("RGB"))[None].to(self.device)
        with self.lock, torch.no_grad(), torch.autocast("cuda", enabled=self.device.type == "cuda"):
            probs = self.model(x).float().softmax(1)[0].cpu()
        p, i = probs.topk(min(TOP, len(self.names)))
        return [{"name": self.names[k], "prob": round(v, 4)} for v, k in zip(p.tolist(), i.tolist())]


def handler(ident: Identifier, page: bytes):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code: int, body: bytes, kind: str):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, code: int, obj):
            self.send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path.split("?")[0] == "/":
                return self.send(200, page, "text/html; charset=utf-8")
            if self.path == "/api/info":
                return self.json(200, {"characters": len(ident.names),
                                       "device": ident.device.type})
            self.send(404, b"Not found", "text/plain")

        def do_POST(self):
            # The page sends one photo per request, as the raw body.
            if self.path != "/api/identify":
                return self.send(404, b"Not found", "text/plain")
            size = int(self.headers.get("Content-Length") or 0)
            if not 0 < size <= MAX_BYTES:
                return self.json(413, {"error": f"photos up to {MAX_BYTES // 2**20} MB"})
            data = self.rfile.read(size)
            try:
                return self.json(200, {"guesses": ident.guess(data)})
            except Exception as e:   # not a picture PIL can read, mostly
                return self.json(400, {"error": f"couldn't read this picture ({e})"})

        def log_message(self, *args):
            pass

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL, help=f"timm model name (default {MODEL})")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--no-browser", action="store_true", help="Don't open the page")
    args = p.parse_args()

    print(f"Loading {args.model} (the first time downloads it)…")
    ident = Identifier(args.model)
    # 127.0.0.1 only: the page is for this computer
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(ident, PAGE.read_bytes()))
    url = f"http://localhost:{args.port}"
    print(f"{len(ident.names)} characters, on {ident.device.type}. Open {url} (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
