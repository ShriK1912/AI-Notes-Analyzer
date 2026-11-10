import asyncio
import logging
import random
import re
import time
from typing import List, Optional
from google import genai
from google.genai import types as genai_types
from app.core.config import settings

logger = logging.getLogger(__name__)

# Generation models in order of quality & fallback
DEFAULT_GENERATION_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
]

# Embedding models in order of fallback
DEFAULT_EMBEDDING_MODELS = [
    "models/gemini-embedding-001",
    "models/gemini-embedding-2-preview",
]

class KeySlot:
    """Represents a single Gemini API key with cooldown and usage tracking."""
    def __init__(self, key: str, index: int):
        self.key = key.strip()
        self.masked = f"{self.key[:6]}...{self.key[-4:]}" if len(self.key) > 10 else "invalid"
        self.index = index
        self.client = genai.Client(api_key=self.key)
        self.cooldown_until: float = 0.0
        self.success_count: int = 0
        self.error_count: int = 0

    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    def mark_rate_limited(self, cooldown_seconds: float = 60.0):
        self.cooldown_until = time.time() + cooldown_seconds
        self.error_count += 1
        logger.warning(
            f"KeySlot [{self.index}] ({self.masked}) rate-limited (429). "
            f"Set on cooldown for {cooldown_seconds:.1f}s."
        )

    def mark_success(self):
        self.success_count += 1
        # Clear cooldown if previously set
        if self.cooldown_until > 0 and time.time() >= self.cooldown_until:
            self.cooldown_until = 0.0


class GeminiPool:
    """
    Manages a pool of Gemini API keys with:
    - Round-robin selection across healthy keys
    - Automatic 429 failover to alternate keys in the pool
    - Multi-model fallback (e.g. 2.5-flash -> 2.0-flash -> 1.5-flash)
    - In-flight concurrency limiting to smooth out burst spikes
    """
    def __init__(self):
        self.slots: List[KeySlot] = []
        self._current_index = 0
        # Concurrency semaphore: max 12 simultaneous calls to avoid instant burst throttling
        self._semaphore = asyncio.Semaphore(12)
        self._initialize_keys()

    def _initialize_keys(self):
        raw_keys: List[str] = []
        
        # Check comma-separated GEMINI_API_KEYS first
        if settings.GEMINI_API_KEYS:
            raw_keys.extend([k.strip() for k in settings.GEMINI_API_KEYS.split(",") if k.strip()])
            
        # Also include single GEMINI_API_KEY if not already in list
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY not in raw_keys:
            raw_keys.append(settings.GEMINI_API_KEY.strip())
            
        for i, key in enumerate(raw_keys):
            self.slots.append(KeySlot(key, i))
            
        logger.info(f"GeminiPool initialized with {len(self.slots)} key(s). Total nominal capacity: {len(self.slots) * 15} RPM.")

    def get_healthy_slot(self) -> Optional[KeySlot]:
        """Get the next healthy key slot using round-robin, skipping slots on cooldown."""
        if not self.slots:
            return None
            
        # Try round-robin from current index
        total = len(self.slots)
        for offset in range(total):
            idx = (self._current_index + offset) % total
            slot = self.slots[idx]
            if slot.is_available():
                self._current_index = (idx + 1) % total
                return slot
                
        # If all slots are on cooldown, return the one that will recover earliest
        return min(self.slots, key=lambda s: s.cooldown_until)

    async def generate_content_with_failover(
        self,
        prompt: str,
        json_mode: bool = False,
        models: Optional[List[str]] = None,
        max_retries: int = 4
    ) -> str:
        """
        Generate content trying all available healthy keys and fallback models.
        Smooths out bursts and handles 429 rate limits gracefully.
        """
        candidate_models = models or DEFAULT_GENERATION_MODELS
        config = genai_types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
        
        last_error = None
        
        async with self._semaphore:
            for attempt in range(max_retries):
                for model in candidate_models:
                    slot = self.get_healthy_slot()
                    if not slot:
                        raise RuntimeError("No Gemini API keys configured in GeminiPool.")
                        
                    try:
                        loop = asyncio.get_running_loop()
                        response = await loop.run_in_executor(
                            None,
                            lambda m=model, c=slot.client: c.models.generate_content(
                                model=m,
                                contents=prompt,
                                config=config
                            )
                        )
                        text = (response.text or "").strip()
                        if text:
                            slot.mark_success()
                            return text
                    except Exception as e:
                        last_error = e
                        err_str = str(e)
                        
                        # Check for rate limit / capacity exhaustion
                        is_rate_limit = any(x in err_str for x in ("429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "503", "quota"))
                        
                        if is_rate_limit:
                            # Extract suggested retryDelay if Google provided one (e.g. 'retryDelay': '35s')
                            delay = 60.0
                            m_delay = re.search(r"'retryDelay':\s*'(\d+)s'", err_str)
                            if m_delay:
                                delay = float(m_delay.group(1)) + 1.0
                                
                            slot.mark_rate_limited(cooldown_seconds=delay)
                            logger.warning(
                                f"Model [{model}] failed on Key [{slot.index}] with rate limit. "
                                f"Failing over to next available key/model..."
                            )
                            continue
                        else:
                            # Non-transient error (e.g. invalid prompt, schema mismatch) -> raise directly
                            raise
                            
                # If all models failed across keys, back off briefly before next retry round
                backoff = 2.0 * (2 ** attempt) + random.uniform(0.5, 1.5)
                logger.warning(f"All Gemini models/keys exhausted on attempt {attempt + 1}/{max_retries}. Backing off {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                
        raise RuntimeError(f"Gemini generation failed after {max_retries} retries across all keys: {last_error}")

    async def embed_texts_with_failover(
        self,
        texts: List[str],
        task_type: str,
        max_dim: int = 768,
        models: Optional[List[str]] = None,
        max_retries: int = 3
    ) -> List[List[float]]:
        """
        Embed texts with automatic key rotation and retry on 429.
        """
        candidate_models = models or DEFAULT_EMBEDDING_MODELS
        last_error = None
        
        async with self._semaphore:
            for attempt in range(max_retries):
                for model in candidate_models:
                    slot = self.get_healthy_slot()
                    if not slot:
                        raise RuntimeError("No Gemini API keys configured for embeddings.")
                        
                    try:
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            None,
                            lambda m=model, c=slot.client: c.models.embed_content(
                                model=m,
                                contents=texts,
                                config=genai_types.EmbedContentConfig(task_type=task_type),
                            )
                        )
                        embeddings = result.embeddings
                        vecs = []
                        for e in embeddings:
                            vals = list(e.values)
                            if len(vals) > max_dim:
                                vals = vals[:max_dim]
                            vecs.append(vals)
                        slot.mark_success()
                        return vecs
                    except Exception as e:
                        last_error = e
                        err_str = str(e)
                        is_rate_limit = any(x in err_str for x in ("429", "RESOURCE_EXHAUSTED", "503", "quota"))
                        if is_rate_limit:
                            slot.mark_rate_limited(cooldown_seconds=45.0)
                            logger.warning(f"Embedding failed on Key [{slot.index}] for {model}. Failing over...")
                            continue
                        elif "not found" in err_str.lower() or "404" in err_str:
                            # Model not supported on this key, skip to next model
                            break
                        else:
                            raise
                await asyncio.sleep(2.0 * (attempt + 1))
                
        raise RuntimeError(f"Gemini embedding failed after retries: {last_error}")

# Singleton pool instance
gemini_pool = GeminiPool()
