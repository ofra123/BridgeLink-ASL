"""BridgeLink ASL package."""

from .config import AppConfig, load_config

__all__ = ["AppConfig", "MockSentenceInterpreter", "load_config", "run_wrapper"]
__version__ = "0.2.0"


def __getattr__(name: str):
    if name in {"MockSentenceInterpreter", "run_wrapper"}:
        from .wrapper import MockSentenceInterpreter, run_wrapper

        exported = {
            "MockSentenceInterpreter": MockSentenceInterpreter,
            "run_wrapper": run_wrapper,
        }
        return exported[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
