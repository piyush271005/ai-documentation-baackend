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
            "Use the provided documentation snippets to answer the user's question accurately. "
            "If the answer cannot be found in the snippets, state that you do not know. "
            "Cite your sources using brackets (e.g. [Snippet 1], [Snippet 2]) when referencing facts."
        )
        
        context_str = ""
        for i, chunk in enumerate(chunks):
            header_str = f" > {chunk['parent_header']}" if chunk.get("parent_header") else ""
            context_str += f"--- Snippet {i+1} ---\n"
            context_str += f"Source: {chunk['url']}\n"
            context_str += f"Title: {chunk['title']}{header_str}\n"
            context_str += f"Content: {chunk['content']}\n\n"
            
        user_prompt = f"Context snippets:\n{context_str}\nQuestion: {query}\nAnswer:"
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
        # Use Gemini 1.5 Flash
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
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
        # We can use a lightweight model like llama3 or similar
        data = {
            "model": "llama3",
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
                    return f"Error from local Ollama (Status {res.status_code}): Make sure Ollama is running and has the 'llama3' model loaded."
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return f"Could not connect to Ollama. Make sure it is running at {settings.OLLAMA_BASE_URL} and the model 'llama3' is pulled."

    def _generate_mock_answer(self, query: str, chunks: list[dict]) -> str:
        """
        A rule-based answer generation fallback when no LLM API is available.
        Synthesizes a response by parsing the retrieved snippets.
        """
        if not chunks:
            return "No matching documentation snippets were found to answer your question."
            
        # Analyze query keywords
        query_words = set(query.lower().split())
        
        # Construct summary response
        answer = f"### [Local AI Mock Response]\n\nBased on your query **\"{query}\"**, I retrieved documentation from **{chunks[0]['title']}**:\n\n"
        
        for idx, chunk in enumerate(chunks[:3]):
            header_trail = f" > {chunk['parent_header']}" if chunk.get("parent_header") else ""
            answer += f"**From {chunk['title']}{header_trail} [Snippet {idx+1}]:**\n"
            
            # Simple content cleaning and highlighting matching sentences
            content = chunk["content"]
            sentences = re.split(r"(?<=[.!?])\s+", content)
            
            matched_sentences = []
            for sent in sentences:
                sent_lower = sent.lower()
                # If sentence has overlap with query keywords
                overlap = len(set(sent_lower.split()) & query_words)
                if overlap > 0:
                    matched_sentences.append((overlap, sent))
            
            # Sort by overlap score
            matched_sentences.sort(key=lambda x: x[0], reverse=True)
            
            if matched_sentences:
                # Output the top matching sentence, and some context
                answer += f"> ... {matched_sentences[0][1]} ...\n\n"
            else:
                # Output a snippet of the content
                excerpt = content[:200] + "..." if len(content) > 200 else content
                answer += f"> {excerpt}\n\n"
                
        answer += "\n*Note: Since no cloud/local LLM keys were detected, I am using a semantic keyword-extraction engine to summarize these source snippets. To enable realistic AI completions, configure your Gemini/OpenAI key in the Settings panel.*"
        
        return answer

# Global LLM service instance
llm_service = LLMService()
