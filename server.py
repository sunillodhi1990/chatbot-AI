"""Local server for the voice chat.

- Serves index.html
- /api/free : proxies the free Pollinations text API (the browser can't call it directly,
              it demands a Turnstile captcha token when an Origin header is present)
- /api/tts  : natural female neural voice via Microsoft Edge's free TTS (pip install --user edge-tts)
- /api/image: pictures for stories / the image creator (free Pollinations image API, no key)
- /api/chat : the user's own API key for ChatGPT, Claude, Groq, OpenRouter, DeepSeek or Mistral
              (Claude uses the official SDK: pip install --user anthropic)

Run:  python server.py   then open http://localhost:8000
"""
import asyncio
import http.server
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import anthropic
import edge_tts

UPSTREAM = "https://text.pollinations.ai/openai"
PORT = 8000
# Female neural voices per UI language
VOICES = {"hi-IN": "hi-IN-SwaraNeural", "en-IN": "en-IN-NeerjaNeural", "en-US": "en-US-AvaNeural"}
# If the reply is written in another Indian script, use a female voice that can read it
SCRIPT_VOICES = [
    (r"[ఀ-౿]", "te-IN-ShrutiNeural"),   # Telugu
    (r"[஀-௿]", "ta-IN-PallaviNeural"),  # Tamil
    (r"[ಀ-೿]", "kn-IN-SapnaNeural"),    # Kannada
    (r"[ഀ-ൿ]", "ml-IN-SobhanaNeural"),  # Malayalam
    (r"[ঀ-৿]", "bn-IN-TanishaaNeural"), # Bengali
    (r"[઀-૿]", "gu-IN-DhwaniNeural"),   # Gujarati
    (r"[਀-੿]", "pa-IN-OjasNeural"),     # Punjabi (Gurmukhi)
]
# Voice styles the page can ask for (pitch/rate tweaks on any voice)
STYLES = {
    "normal": ("+3Hz", "+4%"),
    "cartoon": ("+30Hz", "+8%"),
    "deep": ("-15Hz", "-2%"),
    "slow": ("+0Hz", "-22%"),
}
VOICE_NAME = re.compile(r"^[a-z]{2,3}-[A-Z]{2}-[A-Za-z]+Neural$")
FREE_MAX_WAIT = 75  # seconds to keep retrying the rate-limited free API before giving up

# OpenAI-compatible chat endpoints (all take "Authorization: Bearer <key>")
OPENAI_COMPATIBLE = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
}
# Claude models that take output_config.effort and the server-side refusal fallback
CLAUDE_5X = re.compile(r"^claude-(opus|sonnet|haiku|fable)-5")
CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class ChatError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def chat_openai_compatible(provider, key, model, system, messages):
    out = [{"role": "system", "content": system}]
    for m in messages:
        if m.get("image"):
            img = m["image"]
            url = f"data:{img['mimeType']};base64,{img['data']}"
            content = [{"type": "text", "text": m["text"]}, {"type": "image_url", "image_url": {"url": url}}]
        else:
            content = m["text"]
        out.append({"role": m["role"], "content": content})
    req = urllib.request.Request(
        OPENAI_COMPATIBLE[provider], method="POST",
        data=json.dumps({"model": model, "messages": out}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 "User-Agent": "voice-chat/1.0", "X-Title": "AI Chatbot"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            j = json.loads(body)
            err = j.get("error")
            # providers differ: {"error": {"message"}}, {"error": "..."}, or a top-level "message"/"detail"
            msg = (err.get("message") if isinstance(err, dict) else err) or j.get("message") or j.get("detail")
            if e.code == 401 and msg and msg.lower() in ("unauthorized", "missing authentication header"):
                msg = "Invalid API key."
        except Exception:
            msg = body.decode(errors="replace")[:300]
        raise ChatError(e.code, msg or f"HTTP {e.code}")
    except urllib.error.URLError as e:
        raise ChatError(502, f"Network error: {e.reason}")
    return data["choices"][0]["message"]["content"]


def chat_claude(key, model, system, messages):
    out = []
    for m in messages:
        blocks = []
        if m.get("image"):
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": m["image"]["mimeType"], "data": m["image"]["data"]}})
        blocks.append({"type": "text", "text": m["text"]})
        out.append({"role": m["role"], "content": blocks})
    client = anthropic.Anthropic(api_key=key, max_retries=2)
    params = dict(model=model, max_tokens=16000, system=system, messages=out)
    if CLAUDE_5X.match(model):
        params["output_config"] = {"effort": "low"}  # short conversational replies: keep latency down
    try:
        if model in CLAUDE_FALLBACK_MODELS:
            # if a safety classifier declines, the API re-runs the request on a suitable fallback model
            response = client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params)
        else:
            response = client.messages.create(**params)
    except anthropic.AuthenticationError:
        raise ChatError(401, "Invalid Claude API key.")
    except anthropic.PermissionDeniedError as e:
        raise ChatError(403, e.message)
    except anthropic.NotFoundError:
        raise ChatError(404, f"Claude model not found: {model}")
    except anthropic.RateLimitError:
        raise ChatError(429, "Claude rate limit reached. Please wait a moment and try again.")
    except anthropic.APIStatusError as e:
        raise ChatError(e.status_code, e.message)
    except anthropic.APIConnectionError:
        raise ChatError(502, "Couldn't reach the Claude API. Check your internet connection.")
    if response.stop_reason == "refusal":
        return "Sorry, I can't help with that one."
    return "".join(b.text for b in response.content if b.type == "text")


async def synthesize(text, voice, style="normal"):
    pitch, rate = STYLES.get(style, STYLES["normal"])
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return bytes(audio)


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # never serve a stale copy of the page after an update
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_body(self, status, data, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/image?"):
            return self.image()
        super().do_GET()

    def image(self):
        # /api/image?prompt=...&seed=...  -> picture from the free Pollinations image API
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        prompt = (q.get("prompt", [""])[0] or "a friendly robot")[:400]
        seed = re.sub(r"\D", "", q.get("seed", ["1"])[0])[:9] or "1"
        url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt)
               + f"?width=768&height=512&nologo=true&seed={seed}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "voice-chat/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data, ctype = r.read(), r.headers.get("Content-Type", "image/jpeg")
            if not ctype.startswith("image/"):
                raise RuntimeError("no image returned")
            self.send_body(200, data, ctype)
        except Exception as e:
            self.send_body(502, json.dumps({"error": str(e)}).encode(), "application/json")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/api/free":
            self.proxy_free(body)
        elif self.path == "/api/tts":
            self.tts(body)
        elif self.path == "/api/chat":
            self.chat(body)
        else:
            self.send_error(404)

    def proxy_free(self, body):
        # The free tier allows ~1 reply per 30s per IP: on a rate-limit, wait and retry
        # silently so the user just sees a longer "thinking" instead of an error.
        deadline = time.time() + FREE_MAX_WAIT
        while True:
            req = urllib.request.Request(
                UPSTREAM, data=body, method="POST",
                headers={"Content-Type": "application/json", "User-Agent": "voice-chat/1.0"},
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    status, data = r.status, r.read()
            except urllib.error.HTTPError as e:
                status, data = e.code, e.read()
            except Exception as e:
                status, data = 502, json.dumps({"error": str(e)}).encode()
            limited = status == 429 or b"limit reached" in data.lower()
            if (not limited and status < 500) or time.time() >= deadline:
                break
            m = re.search(rb"retry after (\d+)\s*s", data, re.I)
            time.sleep(min(max(int(m.group(1)) if m else 3, 2), 10))
        self.send_body(status, data, "application/json")

    def chat(self, body):
        try:
            req = json.loads(body)
            provider, key, model = req["provider"], req["key"].strip(), req["model"].strip()
            if provider == "anthropic":
                text = chat_claude(key, model, req["system"], req["messages"])
            elif provider in OPENAI_COMPATIBLE:
                text = chat_openai_compatible(provider, key, model, req["system"], req["messages"])
            else:
                raise ChatError(400, f"Unknown provider: {provider}")
            self.send_body(200, json.dumps({"text": text}).encode(), "application/json")
        except ChatError as e:
            self.send_body(e.status, json.dumps({"error": {"message": str(e)}}).encode(), "application/json")
        except Exception as e:
            self.send_body(500, json.dumps({"error": {"message": str(e)}}).encode(), "application/json")

    def tts(self, body):
        try:
            req = json.loads(body)
            text = req["text"][:3000]
            style = req.get("style", "normal")
            chosen = req.get("voice") or ""
            if not VOICE_NAME.match(chosen):
                chosen = VOICES.get(req.get("lang"), VOICES["hi-IN"])
            # a reply in another Indian script needs a voice that can read it
            voice = next((v for pat, v in SCRIPT_VOICES if re.search(pat, text)), chosen)
            try:
                audio = asyncio.run(synthesize(text, voice, style))
            except edge_tts.exceptions.NoAudioReceived:
                # e.g. an English-only voice given Devanagari text: Swara reads both
                audio = asyncio.run(synthesize(text, VOICES["hi-IN"], style))
            if not audio:
                raise RuntimeError("no audio returned")
            self.send_body(200, audio, "audio/mpeg")
        except Exception as e:
            self.send_body(502, json.dumps({"error": str(e)}).encode(), "application/json")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # On Windows, address reuse lets a second copy bind the same port and silently share requests
    # with an old copy; refuse instead so only one (up-to-date) server answers.
    http.server.ThreadingHTTPServer.allow_reuse_address = False
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        raise SystemExit(f"Port {PORT} is already in use - another server.py is probably running. "
                         "Close it first (Ctrl+C in its window), then start this one again.")
    print(f"Serving on http://localhost:{PORT}")
    server.serve_forever()
