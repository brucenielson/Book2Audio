import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Model used by all tests that call the real LLM (integration and canonical).
# Update this one variable to switch models across the entire test suite.
TEST_LLM_MODEL: str = 'llama3.1:8b'
