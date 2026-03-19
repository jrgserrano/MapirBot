"""LangChain wrapper for Graphiti LLM client."""

import logging
from typing import Any, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)

class LangChainGraphitiClient(OpenAIClient):
    """
    A Graphiti-compatible LLM client that uses LangChain internally.
    """
    def __init__(self, config: LLMConfig):
        # We don't need the internal OpenAI client, but we call super to satisfy parents
        super().__init__(config=config)
        self.llm = ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature or 0
        )
        logger.info(f"LangChainGraphitiClient initialized with model: {config.model}")

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: Optional[type[BaseModel]] = None,
        max_tokens: int = 2000,
        model_size: ModelSize = ModelSize.medium,
    ) -> tuple[dict[str, Any], int, int]:
        """
        Overrides the internal generation to use LangChain.
        """
        lc_messages = []
        total_chars = 0
        for m in messages:
            content = m.content
            # Basic sanity check/truncation for extreme cases in background extraction
            if len(content) > 10000:
                logger.warning(f"Truncating extremely long message (orig len {len(content)})")
                content = content[:10000] + "... [TRUNCATED]"
            
            total_chars += len(content)
            
            if m.role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        logger.info(f"LangChain generation starting. Total chars: {total_chars}")

        try:
            if response_model:
                logger.debug(f"LangChain: Structured output for {response_model.__name__}")
                # Use with_structured_output for robust JSON/Schema handling
                structured_llm = self.llm.with_structured_output(response_model)
                response = await structured_llm.ainvoke(lc_messages)
                
                # If constrained decoding is active, this will be highly reliable
                # response is already a Pydantic object
                # We return the dict representation as expected by Graphiti internal handlers
                return response.model_dump(), 0, 0
            else:
                logger.debug("LangChain: Standard completion")
                response = await self.llm.ainvoke(lc_messages)
                content = response.content
                
                # Try to parse as JSON if it looks like JSON (standard for Graphiti non-model calls)
                import json
                import re
                try:
                    # Basic cleaning
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                    if content.startswith("```"):
                        content = re.sub(r"```[a-z]*\n?(.*?)\n?```", r"\1", content, flags=re.DOTALL).strip()
                    
                    data = json.loads(content)
                    return data, 0, 0
                except:
                    # Fallback for plain text
                    return {"content": content}, 0, 0
                    
        except Exception as e:
            msg_dump = "\n".join([f"[{type(m).__name__}]: {m.content[:200]}..." for m in lc_messages])
            logger.error(f"LangChain generation failed: {e}\nMessages overview:\n{msg_dump}")
            raise

    # We override these to do nothing, as _generate_response handles everything now
    async def _create_completion(self, *args, **kwargs):
        pass
        
    async def _create_structured_completion(self, *args, **kwargs):
        pass
