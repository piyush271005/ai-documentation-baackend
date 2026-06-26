import hashlib
from typing import List, Dict, Any

class Chunker:
    def __init__(self, target_chunk_size: int = 800, overlap_sentences: int = 1):
        self.target_chunk_size = target_chunk_size
        self.overlap_sentences = overlap_sentences

    @staticmethod
    def generate_chunk_id(url: str, index: int, text: str) -> str:
        """Generate a unique deterministic hash ID for a text chunk."""
        hasher = hashlib.md5()
        hasher.update(f"{url}||{index}||{text}".encode("utf-8"))
        return hasher.hexdigest()

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        url = doc.get("url", "")
        title = doc.get("title", "")
        blocks = doc.get("blocks", [])

        chunks = []
        current_chunk_text = []
        current_chunk_size = 0
        current_header = None

        for block in blocks:
            block_type = block.get("type")
            block_text = block.get("text", "").strip()
            block_header = block.get("header")

            if not block_text:
                continue

            # Handle headings
            if block_type == "heading":
                # Flush previous section
                if current_chunk_text:
                    self._flush_chunk(
                        chunks,
                        current_chunk_text,
                        url,
                        title,
                        current_header
                    )
                    current_chunk_text = []
                    current_chunk_size = 0

                # Update to the new section
                current_header = block_text
                continue

            # Handle code blocks separately
            if block_type == "code_block":
                if current_chunk_text:
                    self._flush_chunk(
                        chunks,
                        current_chunk_text,
                        url,
                        title,
                        current_header
                    )
                    current_chunk_text = []
                    current_chunk_size = 0

                self._flush_chunk(
                    chunks,
                    [block_text],
                    url,
                    title,
                    block_header or current_header
                )
                continue

            block_len = len(block_text)

            # Check chunk size
            if (
                current_chunk_size > 0
                and current_chunk_size + block_len > self.target_chunk_size
            ):
                self._flush_chunk(
                    chunks,
                    current_chunk_text,
                    url,
                    title,
                    current_header
                )

                current_chunk_text = [block_text]
                current_chunk_size = block_len

            else:
                current_chunk_text.append(block_text)
                current_chunk_size += block_len

        # Flush remaining content
        if current_chunk_text:
            self._flush_chunk(
                chunks,
                current_chunk_text,
                url,
                title,
                current_header
            )

        return chunks
    def _flush_chunk(self, chunks_list: List[Dict[str, Any]], text_list: List[str], url: str, title: str, header: str):
        full_text = "\n\n".join(text_list)
        if len(full_text.strip()) < 30:  # Skip trivial chunks
            return
            
        chunk_index = len(chunks_list)
        chunk_id = self.generate_chunk_id(url, chunk_index, full_text)
        chunks_list.append({
            "id": chunk_id,
            "url": url,
            "title": title,
            "parent_header": header,
            "content": full_text
        })
