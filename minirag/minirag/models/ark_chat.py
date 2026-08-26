"""方舟 DeepSeek 对话模型适配器（OpenAI 兼容）。

强制结构化输出：用 response_model 的 JSON Schema 约束返回，解析失败重试，
重试耗尽后抛错，不做裸文本回退。
"""
from __future__ import annotations

import asyncio
import json

from openai import AsyncOpenAI

from minirag.config import ModelCfg
from minirag.models.base import TModel
from minirag.schemas import Message


def _strip_code_fence(content: str) -> str:
    """剥离模型偶尔套在 JSON 外层的 ```json ... ``` markdown 围栏。"""
    text = content.strip()
    if not text.startswith("```"):
        return text
    if "\n" in text:
        text = text.split("\n", 1)[1]
    text = text.rstrip()
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class ArkChatModel:
    def __init__(self, cfg: ModelCfg) -> None:
        self._cfg = cfg
        self._client = AsyncOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.effective_timeout,
        )

    async def generate(
        self,
        messages: list[Message],
        response_model: type[TModel],
        thinking: bool = True,
    ) -> TModel:
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
                "strict": True,
            },
        }
        payload = [m.model_dump() for m in messages]
        # thinking=False 时剔除 thinking 字段，用于抽取/关键词等机械任务提速。
        extra = dict(self._cfg.extra_body) if self._cfg.extra_body else None
        if not thinking and extra is not None:
            extra.pop("thinking", None)
            extra = extra or None
        last_err: Exception | None = None
        for attempt in range(self._cfg.max_retries + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._cfg.model,
                    messages=payload,
                    temperature=self._cfg.temperature,
                    response_format=schema,
                    extra_body=extra,
                )
                content = resp.choices[0].message.content or ""
                return response_model.model_validate_json(_strip_code_fence(content))
            except (json.JSONDecodeError, ValueError) as err:
                last_err = err
            except Exception as err:
                last_err = err
                await asyncio.sleep(2**attempt)
        raise RuntimeError(
            f"ArkChatModel.generate 结构化输出失败（{response_model.__name__}）: {last_err}"
        )
