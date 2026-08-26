"""查询关键词抽取：从问题中抽取 high_level / low_level 双层关键词。"""
from __future__ import annotations

from minirag.core import prompts
from minirag.models.base import ChatModel
from minirag.schemas import Keywords


async def extract_keywords(query: str, chat: ChatModel) -> Keywords:
    # 关键词抽取是机械任务，禁用 thinking 提速
    return await chat.generate(prompts.build_keywords_messages(query), Keywords, thinking=False)
