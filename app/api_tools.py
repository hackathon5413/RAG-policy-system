import httpx
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def call_any_url(url: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            return {"error": f"API returned status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}
