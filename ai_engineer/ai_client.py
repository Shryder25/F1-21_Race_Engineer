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
import urllib.error
import urllib.request

DEFAULT_BASE_URL = 'http://localhost:11434/v1'
DEFAULT_MODEL = 'deepseek-r1:8b'

_THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r'<think>.*', re.DOTALL | re.IGNORECASE)


class AIUnavailable(Exception):
    """The model could not be reached or returned nothing usable."""


def strip_reasoning(text: str) -> str:
    """Drop DeepSeek-R1 ``<think>`` reasoning so only the final answer remains."""
    text = _THINK_BLOCK.sub('', text or '')
    text = _THINK_OPEN.sub('', text)  # unterminated block = ran out of tokens mid-thought
    return text.strip()


class AIClient:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 15.0):
        self.base_url = (base_url or os.environ.get('F1_AI_BASE_URL') or DEFAULT_BASE_URL).rstrip('/')
        self.model = model or os.environ.get('F1_AI_MODEL') or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get('F1_AI_API_KEY') or 'ollama'
        self.timeout = timeout

    def _post(self, system: str, user: str, temperature: float, max_tokens: int,
             json_mode: bool) -> str:
        body = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': user}],
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': False,
        }
        if json_mode:
            body['response_format'] = {'type': 'json_object'}
        request = urllib.request.Request(
            f'{self.base_url}/chat/completions', data=json.dumps(body).encode('utf-8'), method='POST',
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {self.api_key}'})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode('utf-8'))
            return payload['choices'][0]['message']['content']
        except (urllib.error.URLError, TimeoutError, OSError, ValueError,
                KeyError, IndexError, TypeError) as error:
            raise AIUnavailable(str(error)) from error

    def chat(self, system: str, user: str, *, temperature: float = 0.3,
             max_tokens: int = 600) -> str:
        """Return the model's final answer text, or raise AIUnavailable.

        ``max_tokens`` is generous on purpose: R1-style models spend tokens on
        reasoning before the answer, and a tight limit would cut the answer off.
        """
        content = self._post(system, user, temperature, max_tokens, json_mode=False)
        answer = strip_reasoning(content)
        if not answer:
            raise AIUnavailable('empty response')
        return answer

    def chat_json(self, system: str, user: str, *, temperature: float = 0.3,
                 max_tokens: int = 700) -> dict:
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
        try:
            content = self._post(system, user, temperature, max_tokens, json_mode=True)
        except AIUnavailable:
            content = self._post(system, user, temperature, max_tokens, json_mode=False)
        text = strip_reasoning(content)
        if not text:
            raise AIUnavailable('empty response')
        try:
            return json.loads(text)
        except ValueError:
            pass
        match = re.search(r'\{.*\}', text, re.DOTALL)   # model wrapped the JSON in prose
        if match:
            try:
                return json.loads(match.group(0))
            except ValueError:
                pass
        raise AIUnavailable('response was not valid JSON')
