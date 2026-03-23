"""
Singleton manager for DuckLake connections.
Provides thread-safe connection caching.
"""

import threading
import hashlib
import logging
from typing import Dict
from contextlib import contextmanager
from .ducklake import DuckLake, create_local_ducklake

logger = logging.getLogger(__name__)


class DuckLakeSingleton:
    """Thread-safe singleton for DuckLake connections."""

    _instances: Dict[str, DuckLake] = {}
    _lock = threading.Lock()
    _configs: Dict[str, dict] = {}

    @classmethod
    def _get_config_hash(cls, **config) -> str:
        """Generate a unique hash for configuration."""
        config_str = str(sorted(config.items()))
        return hashlib.sha256(config_str.encode()).hexdigest()

    @classmethod
    def get_local_ducklake(cls, **config) -> DuckLake:
        """Get or create a local DuckLake instance."""
        config_hash = cls._get_config_hash(**config)

        # Fast path: check if instance exists
        if config_hash in cls._instances:
            instance = cls._instances[config_hash]
            # Verify connection is still alive
            try:
                # Local duckdb doesn't typically drop connection like postgres,
                # but we can try a simple query to ensure it's still healthy
                instance.connection.execute("SELECT 1").fetchone()
                return instance
            except Exception as e:
                # Connection is dead or invalidated, recreate below
                error_msg = str(e).lower()
                if "invalidated" in error_msg or "fatal error" in error_msg:
                    logger.warning(
                        f"Invalidated connection detected for {config_hash}, recreating"
                    )
                else:
                    logger.warning(
                        f"Dead connection detected for {config_hash}, recreating"
                    )
                with cls._lock:
                    if config_hash in cls._instances:
                        try:
                            cls._instances[config_hash].close()
                        except:
                            pass  # Ignore errors during close
                        del cls._instances[config_hash]

        # Slow path: create new instance with lock
        with cls._lock:
            # Double-check after acquiring lock
            if config_hash not in cls._instances:
                logger.info(
                    f"Creating new DuckLake instance for config hash: {config_hash[:8]}..."
                )
                instance = create_local_ducklake(**config)
                instance.connect()
                cls._instances[config_hash] = instance
                cls._configs[config_hash] = config

            return cls._instances[config_hash]

    @classmethod
    def reset_connection(cls, **config) -> None:
        """Force reset a connection by removing it from cache."""
        config_hash = cls._get_config_hash(**config)
        with cls._lock:
            if config_hash in cls._instances:
                try:
                    cls._instances[config_hash].close()
                except:
                    pass  # Ignore errors during close
                del cls._instances[config_hash]
                logger.info(f"Reset connection for config hash: {config_hash[:8]}...")

    @classmethod
    @contextmanager
    def get_connection(cls, **config):
        """Context manager for getting a DuckLake connection."""
        instance = cls.get_local_ducklake(**config)
        try:
            yield instance
        except Exception as e:
            error_msg = str(e).lower()
            if "invalidated" in error_msg or "fatal error" in error_msg:
                logger.error(f"Database invalidation error: {e}")
                # Force reset the connection for invalidation errors
                cls.reset_connection(**config)
                raise
            else:
                logger.error(f"Error during DuckLake operation: {e}")
                # Don't close connection on other errors - they might be recoverable
                raise

    @classmethod
    def close_all(cls):
        """Close all connections (useful for cleanup)."""
        with cls._lock:
            for config_hash, instance in cls._instances.items():
                try:
                    instance.close()
                    logger.info(f"Closed DuckLake connection: {config_hash[:8]}...")
                except Exception as e:
                    logger.error(f"Error closing connection {config_hash[:8]}: {e}")
            cls._instances.clear()
            cls._configs.clear()

    @classmethod
    def get_stats(cls) -> dict:
        """Get statistics about current connections."""
        return {
            "active_connections": len(cls._instances),
            "configs": list(cls._configs.keys()),
        }


# Convenience function for backwards compatibility
def get_singleton_ducklake(**config) -> DuckLake:
    """Get a singleton DuckLake instance."""
    return DuckLakeSingleton.get_local_ducklake(**config)
