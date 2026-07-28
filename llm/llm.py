import re
import httpx
import logging
from backend.config import settings

logger = logging.getLogger("llm")

class LLMService:
    @staticmethod
    def _create_rag_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
        """Creates system and user prompts for RAG."""
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
        """
        Generates an answer based on query and retrieved contexts.
        Dynamically falls back to Mock provider if keys/configs are missing.
        """
        provider = provider_override or settings.DEFAULT_LLM_PROVIDER
        
        # Format Prompt
        sys_prompt, user_prompt = self._create_rag_prompt(query, chunks)
        
        if provider == "openai":
            key = settings.OPENAI_API_KEY
            if not key:
                logger.warning("OpenAI API Key is missing. Falling back to Mock provider.")
                return self._generate_mock_answer(query, chunks)
            return await self._call_openai(sys_prompt, user_prompt, key)
            
        elif provider == "gemini":
            key = settings.GEMINI_API_KEY
            if not key:
                logger.warning("Gemini API Key is missing. Falling back to Mock provider.")
                return self._generate_mock_answer(query, chunks)
            return await self._call_gemini(sys_prompt, user_prompt, key)
            
        elif provider == "ollama":
            return await self._call_ollama(sys_prompt, user_prompt)
            
        else:
            return self._generate_mock_answer(query, chunks)

    async def _call_openai(self, system_prompt: str, user_prompt: str, api_key: str) -> str:
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
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    logger.error(f"OpenAI error: {res.status_code} - {res.text}")
                    return f"Error from OpenAI API (Status {res.status_code}): {res.text[:100]}"
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {e}")
            return f"Error contacting OpenAI: {str(e)}"

    async def _call_gemini(self, system_prompt: str, user_prompt: str, api_key: str) -> str:
        # Use Gemini 2.0 Flash (fast + capable)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        # Combine system prompt & user prompt for Gemini API structure
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
                    return result["candidates"][0]["content"]["parts"][0]["text"].strip()
                else:
                    logger.error(f"Gemini error: {res.status_code} - {res.text}")
                    return f"Error from Gemini API (Status {res.status_code}): {res.text[:100]}"
        except Exception as e:
            logger.error(f"Failed to connect to Gemini: {e}")
            return f"Error contacting Gemini: {str(e)}"

    async def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
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
                    return result["response"].strip()
                else:
                    logger.error(f"Ollama error: {res.status_code} - {res.text}")
                    return f"Error from local Ollama (Status {res.status_code}): Make sure Ollama is running and has the 'llama3.2:3b' model loaded."
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return f"Could not connect to Ollama. Make sure it is running at {settings.OLLAMA_BASE_URL} and the model 'llama3.2:3b' is pulled."

    def _generate_mock_answer(self, query: str, chunks: list[dict]) -> str:
        """
        Rule-based fallback when no LLM API is configured.
        Synthesizes ALL retrieved chunks into one comprehensive answer.
        """
        if not chunks:
            return "No matching documentation snippets were found to answer your question."

        query_words = set(query.lower().split())

        # Collect the most relevant sentences from ALL chunks (not just 3)
        all_sentences = []
        for idx, chunk in enumerate(chunks):
            content = chunk["content"]
            sentences = re.split(r"(?<=[.!?])\s+", content)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                overlap = len(set(sent.lower().split()) & query_words)
                all_sentences.append((overlap, idx, sent))

        # Sort by relevance (keyword overlap), then take top sentences
        all_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = all_sentences[:12]  # Up to 12 best sentences
        # Re-sort top sentences by their original chunk order for narrative flow
        top_sentences.sort(key=lambda x: x[1])

        # Build one cohesive answer
        primary_title = chunks[0]["title"]
        source_urls = list(dict.fromkeys(c["url"] for c in chunks))

        answer = f"## {query}\n\n"

        # Group sentences by section (chunk) for natural flow
        current_chunk = -1
        for _, chunk_idx, sent in top_sentences:
            chunk = chunks[chunk_idx]
            section = chunk.get("parent_header") or chunk["title"]
            if chunk_idx != current_chunk:
                current_chunk = chunk_idx
                answer += f"### {section}\n\n"
            answer += f"{sent} "

        answer = answer.strip()

        # Add sources footer
        answer += "\n\n---\n**Sources:**\n"
        for i, url in enumerate(source_urls, 1):
            answer += f"- [{i}] {url}\n"

        answer += "\n\n*Note: Using rule-based synthesis (no LLM key configured). For AI-generated answers, add a Gemini or OpenAI key in the Settings panel.*"

        return answer

# Global LLM service instance
llm_service = LLMService()
