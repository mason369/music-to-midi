"""
Internationalization (i18n) translator module.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class Translator:
    """
    Translation manager for multi-language support.

    Supports:
    - JSON-based translation files
    - Dynamic language switching
    - Nested key access (e.g., "menu.file")
    - Fallback to default language
    """

    DEFAULT_LANGUAGE = "en_US"
    AVAILABLE_LANGUAGES = {
        "en_US": "English",
        "zh_CN": "中文"
    }

    def __init__(self, language: str = "zh_CN"):
        """
        Initialize translator.

        Args:
            language: Language code (e.g., 'en_US', 'zh_CN')
        """
        self.current_language = language
        self.translations: Dict[str, Dict] = {}
        self._load_translations()

    def _get_i18n_dir(self) -> Path:
        """Get i18n directory path."""
        return Path(__file__).parent

    def _load_translations(self) -> None:
        """Load all available translation files."""
        i18n_dir = self._get_i18n_dir()

        for lang_code in self.AVAILABLE_LANGUAGES:
            file_path = i18n_dir / f"{lang_code}.json"
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.translations[lang_code] = json.load(f)
                    logger.info(f"Loaded translations: {lang_code}")
                except Exception as e:
                    logger.error(f"Failed to load {lang_code}: {e}")
                    self.translations[lang_code] = {}

    def set_language(self, language: str) -> bool:
        """
        Set current language.

        Args:
            language: Language code

        Returns:
            True if language was set successfully
        """
        if language in self.AVAILABLE_LANGUAGES:
            self.current_language = language
            logger.info(f"Language set to: {language}")
            return True
        else:
            logger.warning(f"Unknown language: {language}")
            return False

    def get_language(self) -> str:
        """Get current language code."""
        return self.current_language

    def get_language_name(self, code: Optional[str] = None) -> str:
        """Get language display name."""
        code = code or self.current_language
        return self.AVAILABLE_LANGUAGES.get(code, code)

    def t(self, key: str, **kwargs) -> str:
        """
        Get translated string.

        Args:
            key: Translation key (supports dot notation, e.g., "menu.file")
            **kwargs: Format arguments

        Returns:
            Translated string, or key if not found
        """
        # Try current language
        value = self._get_nested(self.translations.get(self.current_language, {}), key)

        # Fallback to default language
        if value is None and self.current_language != self.DEFAULT_LANGUAGE:
            value = self._get_nested(self.translations.get(self.DEFAULT_LANGUAGE, {}), key)

        # Return key if not found
        if value is None:
            logger.debug(f"Translation not found: {key}")
            return key

        # Format with arguments
        if kwargs:
            try:
                value = value.format(**kwargs)
            except KeyError:
                pass

        return value

    def _get_nested(self, data: Dict, key: str) -> Optional[str]:
        """
        Get nested dictionary value using dot notation.

        Args:
            data: Dictionary to search
            key: Dot-separated key path

        Returns:
            Value if found, None otherwise
        """
        keys = key.split('.')
        value = data

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return None

        if isinstance(value, str):
            return value
        return None

    def get_all_keys(self, prefix: str = "") -> list:
        """
        Get all translation keys.

        Args:
            prefix: Key prefix filter

        Returns:
            List of matching keys
        """
        def extract_keys(data: Dict, current_prefix: str = "") -> list:
            keys = []
            for k, v in data.items():
                full_key = f"{current_prefix}.{k}" if current_prefix else k
                if isinstance(v, dict):
                    keys.extend(extract_keys(v, full_key))
                else:
                    keys.append(full_key)
            return keys

        all_keys = extract_keys(self.translations.get(self.current_language, {}))

        if prefix:
            all_keys = [k for k in all_keys if k.startswith(prefix)]

        return all_keys


# Global translator instance
_translator: Optional[Translator] = None


def get_translator() -> Translator:
    """Get global translator instance."""
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator


def set_language(language: str) -> bool:
    """Set global language."""
    return get_translator().set_language(language)


def t(key: str, **kwargs) -> str:
    """
    Translate a key using the global translator.

    Args:
        key: Translation key
        **kwargs: Format arguments

    Returns:
        Translated string
    """
    return get_translator().t(key, **kwargs)
