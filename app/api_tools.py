import httpx
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def call_any_url(url: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            if response.status_code == 200:
                # Try JSON first, fallback to text
                try:
                    return response.json()
                except Exception:
                    return {"content": response.text, "content_type": response.headers.get('content-type', '')}
            return {"error": f"API returned status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}
