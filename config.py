"""
global configuration loader
loads config.yaml
"""
import os
from dotenv import load_dotenv
import yaml
from pathlib import Path
from typing import Any, Dict

load_dotenv()

CONFIG_DIR = Path(__file__).parent
CONFIG_FILE = CONFIG_DIR / "config.yaml"

def _substitute_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, "")
        return value
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value

def load_config(config_path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """loada yaml"""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Substitute environment variables
    config = _substitute_env_vars(config)
    
    return config

_config: Dict[str, Any] = {}

def get_config() -> Dict[str, Any]:
    """global conf"""
    global _config
    if not _config:
        _config = load_config()
    return _config

def reload_config() -> Dict[str, Any]:
    """force reload from disk"""
    global _config
    _config = load_config()
    return _config

def get_neo4j_config() -> Dict[str, Any]:
    return get_config().get("neo4j", {})

def get_embedding_config() -> Dict[str, Any]:
    return get_config().get("embeddings", {})

def get_chunking_config() -> Dict[str, Any]:
    return get_config().get("chunking", {})

def get_indexes_config() -> Dict[str, Any]:
    return get_config().get("indexes", {})

def get_similarity_config() -> Dict[str, Any]:
    return get_config().get("similarity", {})

def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """get config for a specific agent"""
    agents = get_config().get("agents", {})
    return agents.get(agent_name, {})

def get_rag_config() -> Dict[str, Any]:
    return get_config().get("rag", {})

def get_api_keys() -> Dict[str, Any]:
    return get_config().get("api_keys", {})

# load config on import
config = get_config()
