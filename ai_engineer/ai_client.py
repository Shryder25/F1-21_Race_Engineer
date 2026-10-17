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

    def chat(self, system: str, user: str, *, temperature: float = 0.3,
             max_tokens: int = 600) -> str:
        """Return the model's final answer text, or raise AIUnavailable.

        ``max_tokens`` is generous on purpose: R1-style models spend tokens on
        reasoning before the answer, and a tight limit would cut the answer off.
        """
        body = json.dumps({
            'model': self.model,
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': user}],
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': False,
        }).encode('utf-8')
        request = urllib.request.Request(
            f'{self.base_url}/chat/completions', data=body, method='POST',
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {self.api_key}'})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode('utf-8'))
            content = payload['choices'][0]['message']['content']
        except (urllib.error.URLError, TimeoutError, OSError, ValueError,
                KeyError, IndexError, TypeError) as error:
            raise AIUnavailable(str(error)) from error
        answer = strip_reasoning(content)
        if not answer:
            raise AIUnavailable('empty response')
        return answer
