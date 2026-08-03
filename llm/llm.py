import re
import httpx
import logging
from backend.config import settings
import os

logger = logging.getLogger("llm")

class LLMService:
    @staticmethod
    def _create_rag_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
        
        system_prompt = (
            "You are an expert technical documentation assistant. "
            "Use the provided documentation snippets to write ONE single, comprehensive, well-structured answer. "
            "Do NOT respond with separate per-snippet summaries. "
            "Synthesize all relevant information into a single flowing response with clear sections if needed. "
            "Use markdown formatting (headings, bullet lists, code blocks) where appropriate. "
            "Cite sources inline using brackets (e.g. [1], [2]) when referencing specific facts. "
            "If the answer cannot be found in the snippets, state that clearly."
        )
        
        context_str = ""
        for i, chunk in enumerate(chunks):
            header_str = f" > {chunk['parent_header']}" if chunk.get("parent_header") else ""
            context_str += f"[{i+1}] Source: {chunk['url']}\n"
            context_str += f"    Title: {chunk['title']}{header_str}\n"
            context_str += f"    Content: {chunk['content']}\n\n"
            
        user_prompt = f"Documentation snippets:\n{context_str}\nQuestion: {query}\n\nWrite a single comprehensive answer:"
        return system_prompt, user_prompt

    async def generate_answer(self, query: str, chunks: list[dict], provider_override: str = None) -> str:
        provider = provider_override or settings.DEFAULT_LLM_PROVIDER
        logger.debug(f"[DEBUG] Generating answer for query='{query}' using provider='{provider}' and {len(chunks)} source chunks...")
        
        sys_prompt, user_prompt = self._create_rag_prompt(query, chunks)
        logger.debug(f"[DEBUG] Prompt generated: sys_len={len(sys_prompt)}, user_len={len(user_prompt)}")
        
        if provider == "openai":
            key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
            if not key:
                logger.debug("[DEBUG] OpenAI API key missing.")
                return "OpenAI API key is missing. Please set your OPENAI_API_KEY in LLM Settings."
            return await self._call_openai(sys_prompt, user_prompt, key)
            
        elif provider == "gemini":
            key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
            if not key:
                logger.debug("[DEBUG] Gemini API key missing.")
                return "Gemini API key is missing. Please set your GEMINI_API_KEY in LLM Settings."
            return await self._call_gemini(sys_prompt, user_prompt, key)
            
        elif provider == "groq":
            key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
            if not key:
                logger.debug("[DEBUG] Groq API key missing.")
                return "Groq API key is missing. Please set your GROQ_API_KEY in LLM Settings."
            return await self._call_groq(sys_prompt, user_prompt, key)

        elif provider == "ollama":
            return await self._call_ollama(sys_prompt, user_prompt)
            
        else:
            key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
            if key:
                return await self._call_gemini(sys_prompt, user_prompt, key)
            groq_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
            if groq_key:
                return await self._call_groq(sys_prompt, user_prompt, groq_key)
            logger.debug("[DEBUG] No active LLM provider key available.")
            return "No active LLM provider configured. Please select Gemini, Groq, OpenAI, or Ollama in LLM Settings and provide an API key."

    async def _call_openai(self, system_prompt: str, user_prompt: str, api_key: str) -> str:
        logger.debug("[DEBUG] Sending request to OpenAI API (model: gpt-4o-mini)...")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        }
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(url, json=data, headers=headers, timeout=30.0)
                if res.status_code == 200:
                    result = res.json()
                    answer = result["choices"][0]["message"]["content"].strip()
                    logger.debug(f"[DEBUG] OpenAI response received successfully (len={len(answer)} chars).")
                    return answer
                else:
                    logger.error(f"OpenAI error: {res.status_code} - {res.text}")
                    return f"Error from OpenAI API (Status {res.status_code}): {res.text[:100]}"
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {e}")
            return f"Error contacting OpenAI: {str(e)}"

    async def _call_gemini(self, system_prompt: str, user_prompt: str, api_key: str) -> str:
        logger.debug("[DEBUG] Sending request to Gemini API (model: gemini-2.0-flash)...")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
       
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{system_prompt}\n\n{user_prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2
            }
        }
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(url, json=data, headers=headers, timeout=30.0)
                if res.status_code == 200:
                    result = res.json()
                    answer = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                    logger.debug(f"[DEBUG] Gemini response received successfully (len={len(answer)} chars).")
                    return answer
                else:
                    logger.error(f"Gemini error: {res.status_code} - {res.text}")
                    return f"Error from Gemini API (Status {res.status_code}): {res.text[:100]}"
        except Exception as e:
            logger.error(f"Failed to connect to Gemini: {e}")
            return f"Error contacting Gemini: {str(e)}"

    async def _call_groq(self, system_prompt: str, user_prompt: str, api_key: str) -> str:
        logger.debug("[DEBUG] Sending request to Groq Cloud API (model: llama-3.3-70b-versatile)...")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        }
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(url, json=data, headers=headers, timeout=30.0)
                if res.status_code == 200:
                    result = res.json()
                    answer = result["choices"][0]["message"]["content"].strip()
                    logger.debug(f"[DEBUG] Groq response received successfully (len={len(answer)} chars).")
                    return answer
                else:
                    logger.error(f"Groq error: {res.status_code} - {res.text}")
                    return f"Error from Groq API (Status {res.status_code}): {res.text[:100]}"
        except Exception as e:
            logger.error(f"Failed to connect to Groq: {e}")
            return f"Error contacting Groq API: {str(e)}"

    async def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        logger.debug(f"[DEBUG] Sending request to local Ollama server at {settings.OLLAMA_BASE_URL} (model: llama3.2:3b)...")
        url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        
        data = {
            "model": "llama3.2:3b",
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(url, json=data, timeout=30.0)
                if res.status_code == 200:
                    result = res.json()
                    answer = result["response"].strip()
                    logger.debug(f"[DEBUG] Ollama response received successfully (len={len(answer)} chars).")
                    return answer
                else:
                    logger.error(f"Ollama error: {res.status_code} - {res.text}")
                    return f"Error from local Ollama (Status {res.status_code}): Make sure Ollama is running and has the 'llama3.2:3b' model loaded."
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return f"Could not connect to Ollama. Make sure it is running at {settings.OLLAMA_BASE_URL} and the model 'llama3.2:3b' is pulled."

# Global LLM service instance
llm_service = LLMService()
