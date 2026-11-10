import pytest


@pytest.fixture(autouse=True)
def isolate_from_real_credentials(monkeypatch):
    """Tests must never use the real keys in .env: no API spend, no traces uploaded.

    load_dotenv() does not override variables that already exist, so setting them
    here (empty or dummy) wins over whatever is in .env.
    """
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("TAX_AGENT_MODEL", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")  # an accidental real call fails with 401
