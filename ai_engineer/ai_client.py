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
import urllib.error
import urllib.request

from ai_launcher import KEEP_ALIVE, ollama_root

DEFAULT_BASE_URL = 'http://localhost:11434/v1'
DEFAULT_MODEL = 'deepseek-r1:8b'

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


class AIClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 15.0,
    ):
        self.base_url = (base_url or os.environ.get('F1_AI_BASE_URL') or DEFAULT_BASE_URL).rstrip('/')
        self.model = model or os.environ.get('F1_AI_MODEL') or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get('F1_AI_API_KEY') or 'ollama'
        self.timeout = timeout
        # Local Ollama gets the native API for quick radio answers: it can switch
        # DeepSeek R1's long "thinking" off, which is most of the wait.
        self.ollama_root = ollama_root(self.base_url)

    supports_fast = True

    def _post_native(
        self, system: str, user: str, temperature: float, max_tokens: int, json_mode: bool, think: bool | None
    ) -> str:
        body = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            'stream': False,
            'keep_alive': KEEP_ALIVE,
            'options': {'temperature': temperature, 'num_predict': max_tokens},
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
            return payload['message']['content']
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
            raise AIUnavailable(str(error)) from error

    def _fast(self, system, user, temperature, max_tokens, json_mode) -> str | None:
        """Native Ollama call with thinking off; None means use the OpenAI-compatible path."""
        if not self.ollama_root:
            return None
        try:
            return self._post_native(system, user, temperature, max_tokens, json_mode, think=False)
        except AITimeout:
            raise
        except AIUnavailable:
            pass
        try:  # e.g. an older Ollama or a model that rejects the think switch
            return self._post_native(system, user, temperature, max_tokens, json_mode, think=None)
        except AITimeout:
            raise
        except AIUnavailable:
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
            return payload['choices'][0]['message']['content']
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
            raise AIUnavailable(str(error)) from error

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
            raise AIUnavailable('empty response')
        try:
            return json.loads(text)
        except ValueError:
            pass
        match = re.search(r'\{.*\}', text, re.DOTALL)  # model wrapped the JSON in prose
        if match:
            try:
                return json.loads(match.group(0))
            except ValueError:
                pass
        raise AIUnavailable('response was not valid JSON')
