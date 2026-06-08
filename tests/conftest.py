"""
Global pytest configuration.

All environment variables and sys.modules stubs are applied at module level —
before pytest imports any test file — so src/ modules never see missing
credentials or live infrastructure during the test run.
"""

import os
import sys
from unittest.mock import MagicMock

# ── 1. Environment variables ─────────────────────────────────────────────────
os.environ.setdefault("DSN", "postgresql://test:test@localhost:5432/testdb")
os.environ.setdefault("COLLECTION_NAME", "test_collection")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key-xxxx")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-key-xxxx")

# ── 2. Stub external packages that validate credentials at import time ────────

# langchain_groq — ChatGroq checks API key on construction
_mock_groq = MagicMock()
sys.modules.setdefault("langchain_groq", _mock_groq)

# langchain_google_genai — GoogleGenerativeAIEmbeddings checks API key
_mock_genai = MagicMock()
sys.modules.setdefault("langchain_google_genai", _mock_genai)

# langchain_postgres — PGVector attempts a DB connection
_mock_pgvector = MagicMock()
sys.modules.setdefault("langchain_postgres", _mock_pgvector)

# langchain_community — not installed in this venv; provides WebBaseLoader etc.
_mock_community = MagicMock()
sys.modules.setdefault("langchain_community", _mock_community)
sys.modules.setdefault("langchain_community.document_loaders", _mock_community)

# langchain_text_splitters — not installed in this venv
_mock_text_splitters = MagicMock()
sys.modules.setdefault("langchain_text_splitters", _mock_text_splitters)

# langchain.messages — not present in newer langchain; provide real classes as
# a compatibility shim so nodes.py's `from langchain.messages import ...` works
try:
    from langchain_core.messages import HumanMessage as _HM
    from langchain_core.messages import SystemMessage as _SM

    _mock_lc_messages = MagicMock()
    _mock_lc_messages.HumanMessage = _HM
    _mock_lc_messages.SystemMessage = _SM
    sys.modules.setdefault("langchain.messages", _mock_lc_messages)
except ImportError:
    pass

# langchain.chat_models.init_chat_model — prevent provider-key validation
import langchain.chat_models as _lc_chat_models

if not isinstance(getattr(_lc_chat_models, "init_chat_model", None), MagicMock):
    _lc_chat_models.init_chat_model = MagicMock(return_value=MagicMock())
