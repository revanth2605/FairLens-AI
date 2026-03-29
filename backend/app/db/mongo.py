# Minimal MongoDB helper (placeholder)
from datetime import datetime

_store = []

def insert_record(rec):
    rec['ts'] = datetime.utcnow().isoformat()
    _store.append(rec)

def get_recent(limit=50):
    return list(reversed(_store))[:limit]
