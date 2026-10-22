"""Minimal OpenAI-compatible chat client (stdlib only) for DeepSeek.

Works with any server that speaks ``POST {base_url}/chat/completions``:
  * Ollama (local):  http://localhost:11434/v1   model e.g. deepseek-r1:8b
  * Groq (cloud):    https://api.groq.com/openai/v1   model e.g. deepseek-r1-distill-llama-70b
  * DeepSeek (cloud): https://api.deepseek.com/v1   model e.g. deepseek-chat

Configuration comes from CLI flags in main.py or the environment:
  F1_AI_BASE_URL, F1_AI_MODEL, F1_AI_API_KEY
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request

from ai_launcher import KEEP_ALIVE, ollama_root

DEFAULT_BASE_URL = 'http://localhost:11434/v1'
DEFAULT_MODEL = 'deepseek-r1:8b'
# Context window asked of Ollama. Its own default (2048 on many versions) is smaller than the
# system prompt plus a full race's data, and Ollama silently cuts the start of a prompt that
# doesn't fit: the model then really "doesn't have that information". 4096 fits a full race
# with room for the answer without the extra graphics memory a bigger window would take.
DEFAULT_NUM_CTX = 4096

# The radio line, complete, inside a streamed {"voice": "...", ...} reply.
_VOICE_DONE = re.compile(r'"voice"\s*:\s*"((?:[^"\\]|\\.)*)"', re.IGNORECASE)
# Keys a model may use for its answer instead of "voice" (small models drift from the format).
_ANSWER_KEYS = ('voice', 'radio', 'answer', 'response', 'reply', 'message', 'text', 'advice')

_THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r'<think>.*', re.DOTALL | re.IGNORECASE)


class AIUnavailable(Exception):
    """The model could not be reached or returned nothing usable."""


class AITimeout(AIUnavailable):
    """The model took longer than the timeout. Retrying would only double the wait."""


def _is_timeout(error) -> bool:
    if isinstance(error, (TimeoutError, socket.timeout)):
        return True
    return isinstance(error, urllib.error.URLError) and isinstance(
        error.reason, (TimeoutError, socket.timeout)
    )


def strip_reasoning(text: str) -> str:
    """Drop DeepSeek-R1 ``<think>`` reasoning so only the final answer remains."""
    text = _THINK_BLOCK.sub('', text or '')
    text = _THINK_OPEN.sub('', text)  # unterminated block = ran out of tokens mid-thought
    return text.strip()


def ollama_options(
    num_ctx: int | None = DEFAULT_NUM_CTX, gpu_layers: int | None = None, threads: int | None = None
) -> dict:
    """Model options that decide how Ollama loads the model.

    Every request must send the same ones (the launcher's preload and keep-warm pings too):
    a request with different load options makes Ollama reload the whole model first.
    ``gpu_layers`` 0 keeps the model off the graphics card so the game keeps all of it;
    ``threads`` caps the CPU cores Ollama uses for whatever runs on the CPU.
    """
    options = {}
    if num_ctx:
        options['num_ctx'] = int(num_ctx)
    if gpu_layers is not None:
        options['num_gpu'] = int(gpu_layers)
    if threads:
        options['num_thread'] = int(threads)
    return options


def voice_from_partial(text: str) -> str | None:
    """The finished "voice" string from a JSON reply that may still be arriving, else None."""
    match = _VOICE_DONE.search(text or '')
    if not match:
        return None
    try:
        return json.loads(f'"{match.group(1)}"')
    except ValueError:
        return match.group(1)


def _json_object(text: str):
    """The JSON value in ``text`` (also when wrapped in prose or a code fence), else None."""
    try:
        return json.loads(text)
    except ValueError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except ValueError:
            pass
    return None


def radio_line(value) -> str | None:
    """The spoken answer in a parsed JSON reply: "voice", or another answer-like text field."""
    if isinstance(value, str):
        return value.strip() or None
    if not isinstance(value, dict):
        return None
    by_key = {str(key).lower(): item for key, item in value.items()}
    for key in _ANSWER_KEYS:
        item = by_key.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    for item in value.values():
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def voice_from_reply(text: str) -> str | None:
    """The radio line in a finished reply: JSON {"voice": ...}, another text field, or plain text."""
    text = strip_reasoning(text)
    if not text:
        return None
    parsed = _json_object(text)
    if parsed is not None:
        return radio_line(parsed)
    if text.lstrip().startswith(('{', '[')):
        return voice_from_partial(text)  # JSON cut off: keep the voice line if it was finished
    return text


def _error_text(error) -> str:
    """Readable reason for a failed request, including Ollama's own message ("model not found")."""
    if isinstance(error, urllib.error.HTTPError):
        try:
            detail = json.loads(error.read().decode('utf-8', 'replace')).get('error')
        except (ValueError, AttributeError, OSError):
            detail = None
        return f'HTTP {error.code}: {detail or error.reason}'
    return str(error)


class AIClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 15.0,
        *,
        load_options: dict | None = None,
    ):
        self.base_url = (base_url or os.environ.get('F1_AI_BASE_URL') or DEFAULT_BASE_URL).rstrip('/')
        self.model = model or os.environ.get('F1_AI_MODEL') or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get('F1_AI_API_KEY') or 'ollama'
        self.timeout = timeout
        # Local Ollama gets the native API for quick radio answers: it can switch
        # DeepSeek R1's long "thinking" off, which is most of the wait.
        self.ollama_root = ollama_root(self.base_url)
        self.load_options = ollama_options() if load_options is None else dict(load_options)
        self.last_error = ''  # why the last attempt failed, for the log and --ai-test
        self.last_reply = ''  # the model's raw text from the last streamed answer

    supports_fast = True

    def _post_native(
        self, system: str, user: str, temperature: float, max_tokens: int, json_mode: bool, think: bool | None
    ) -> str:
        body = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            'stream': False,
            'keep_alive': KEEP_ALIVE,
            'options': {**self.load_options, 'temperature': temperature, 'num_predict': max_tokens},
        }
        if think is not None:
            body['think'] = think
        if json_mode:
            body['format'] = 'json'
        request = urllib.request.Request(
            f'{self.ollama_root}/api/chat',
            data=json.dumps(body).encode('utf-8'),
            method='POST',
            headers={'Content-Type': 'application/json'},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode('utf-8'))
            message = payload['message']
            content = message.get('content') or ''
            thinking = message.get('thinking') or ''
            # Some Ollama versions put a thinking model's whole JSON reply in "thinking".
            self.last_reply = content or thinking
            if json_mode and not content.strip() and voice_from_partial(thinking):
                return thinking
            return content
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            if _is_timeout(error):
                raise AITimeout(str(error)) from error
            raise AIUnavailable(_error_text(error)) from error

    def _fast(self, system, user, temperature, max_tokens, json_mode) -> str | None:
        """Native Ollama call with thinking off; None means use the OpenAI-compatible path."""
        if not self.ollama_root:
            return None
        try:
            return self._post_native(system, user, temperature, max_tokens, json_mode, think=False)
        except AITimeout:
            raise
        except AIUnavailable as error:
            self.last_error = str(error)
        try:  # e.g. an older Ollama or a model that rejects the think switch
            return self._post_native(system, user, temperature, max_tokens, json_mode, think=None)
        except AITimeout:
            raise
        except AIUnavailable as error:
            self.last_error = str(error)
            return None

    def _stream_voice(
        self, system: str, user: str, temperature: float, max_tokens: int, think: bool | None
    ) -> str:
        """Stream a JSON radio reply and stop reading as soon as its "voice" line is complete.

        The spoken line comes first in the reply, so this returns before the model writes the
        rest (the on-screen note), and closing the stream makes Ollama stop generating, which
        frees the graphics card for the game sooner.
        """
        body = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            'stream': True,
            'format': 'json',
            'keep_alive': KEEP_ALIVE,
            'options': {**self.load_options, 'temperature': temperature, 'num_predict': max_tokens},
        }
        if think is not None:
            body['think'] = think
        request = urllib.request.Request(
            f'{self.ollama_root}/api/chat',
            data=json.dumps(body).encode('utf-8'),
            method='POST',
            headers={'Content-Type': 'application/json'},
        )
        deadline = time.monotonic() + self.timeout
        content = thinking = ''
        done_reason = None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw in response:
                    if not raw.strip():
                        continue
                    chunk = json.loads(raw.decode('utf-8'))
                    if chunk.get('error'):
                        raise AIUnavailable(str(chunk['error']))
                    message = chunk.get('message') or {}
                    content += message.get('content') or ''
                    thinking += message.get('thinking') or ''
                    voice = voice_from_partial(content)
                    if voice is not None:
                        self.last_reply = content
                        return voice
                    if chunk.get('done'):
                        done_reason = chunk.get('done_reason')
                        break
                    if time.monotonic() > deadline:
                        raise AITimeout(f'no answer within {self.timeout:g} s')
        except AIUnavailable:
            raise
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            AttributeError,
        ) as error:
            if _is_timeout(error):
                raise AITimeout(str(error)) from error
            raise AIUnavailable(_error_text(error)) from error
        self.last_reply = content or thinking
        # Some Ollama versions file a thinking model's whole JSON reply under "thinking".
        voice = voice_from_reply(content) or voice_from_partial(thinking)
        if voice:
            return voice
        if not strip_reasoning(content):
            if thinking or '<think>' in content.lower():
                raise AIUnavailable(f'the model only thought and gave no answer ({max_tokens} token budget)')
            raise AIUnavailable('the model sent an empty reply')
        if done_reason == 'length':
            raise AIUnavailable(f'the reply was cut off at {max_tokens} tokens: {content[:120]!r}')
        raise AIUnavailable(f'the reply had no radio line: {content[:120]!r}')

    def radio_voice(
        self, system: str, user: str, *, temperature: float = 0.3, max_tokens: int = 220
    ) -> str | None:
        """Fastest path for a spoken answer on a local Ollama: streamed, thinking off.

        Returns None when the server isn't a local Ollama (the caller uses chat_json()) or the
        reply was unusable; ``last_error`` then says why.
        """
        if not self.ollama_root:
            return None
        try:
            return self._stream_voice(system, user, temperature, max_tokens, think=False)
        except AITimeout:
            raise
        except AIUnavailable as error:
            self.last_error = str(error)
        try:  # an older Ollama or a model that rejects the think switch
            return self._stream_voice(system, user, temperature, max_tokens, think=None)
        except AITimeout:
            raise
        except AIUnavailable as error:
            self.last_error = str(error)
            return None

    def _post(self, system: str, user: str, temperature: float, max_tokens: int, json_mode: bool) -> str:
        body = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': False,
        }
        if json_mode:
            body['response_format'] = {'type': 'json_object'}
        request = urllib.request.Request(
            f'{self.base_url}/chat/completions',
            data=json.dumps(body).encode('utf-8'),
            method='POST',
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode('utf-8'))
            self.last_reply = payload['choices'][0]['message']['content'] or ''
            return self.last_reply
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            if _is_timeout(error):
                raise AITimeout(str(error)) from error
            raise AIUnavailable(_error_text(error)) from error

    def chat(
        self, system: str, user: str, *, temperature: float = 0.3, max_tokens: int = 600, fast: bool = False
    ) -> str:
        """Return the model's final answer text, or raise AIUnavailable.

        ``max_tokens`` is generous on purpose: R1-style models spend tokens on
        reasoning before the answer, and a tight limit would cut the answer off.
        """
        content = self._fast(system, user, temperature, max_tokens, False) if fast else None
        if content is None:
            content = self._post(system, user, temperature, max_tokens, json_mode=False)
        answer = strip_reasoning(content)
        if not answer:
            raise AIUnavailable('empty response')
        return answer

    def chat_json(
        self, system: str, user: str, *, temperature: float = 0.3, max_tokens: int = 700, fast: bool = False
    ) -> dict:
        """Chat with the server's JSON-object mode, returning the parsed object.

        Prefer this over scraping free text out of ``chat()`` whenever the prompt asks for
        a fixed shape (e.g. {"advice": "...", "voice": "..."}): forcing JSON mode is far more
        reliable than trusting the model to follow a textual format instruction, and it is
        what Box-Box's analyzer.py does for its strategy and radio replies.

        Not every OpenAI-compatible server honours ``response_format``; servers that reject
        the field raise AIUnavailable from the first attempt, so this retries once without it
        and parses the (``<think>``-stripped) reply as JSON by hand. Still raises AIUnavailable
        if no valid JSON object can be found either way.
        """
        content = self._fast(system, user, temperature, max_tokens, True) if fast else None
        if content is None:
            try:
                content = self._post(system, user, temperature, max_tokens, json_mode=True)
            except AITimeout:
                raise  # a slow model won't be faster the second time
            except AIUnavailable:
                content = self._post(system, user, temperature, max_tokens, json_mode=False)
        text = strip_reasoning(content)
        if not text:
            raise AIUnavailable('empty response (the model may have used its whole budget thinking)')
        parsed = _json_object(text)  # also finds JSON the model wrapped in prose
        if isinstance(parsed, dict):
            return parsed
        raise AIUnavailable(f'response was not valid JSON: {text[:120]!r}')
