import hashlib
import uuid


def compute_content_hash(content: str, source: str = "") -> str:
    combined = f"{content}{source}".encode("utf-8", errors="ignore")
    digest = hashlib.sha256(combined).digest()
    return str(uuid.UUID(bytes=digest[:16]))
