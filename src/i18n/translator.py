"""
国际化 (i18n) 翻译模块
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_resource_path(relative_path: str) -> Path:
    """
    获取资源文件的绝对路径，支持开发环境和PyInstaller打包后的环境

    参数:
        relative_path: 相对于项目根目录的路径

    返回:
        资源文件的绝对路径
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后的环境
        base_path = Path(sys._MEIPASS)
    else:
        # 开发环境
        base_path = Path(__file__).parent.parent.parent
    return base_path / relative_path


class Translator:
    """
    多语言支持的翻译管理器

    支持:
    - 基于JSON的翻译文件
    - 动态语言切换
    - 嵌套键访问（如 "menu.file"）
    - 回退到默认语言
    """

    DEFAULT_LANGUAGE = "en_US"
    AVAILABLE_LANGUAGES = {
        "en_US": "English",
        "zh_CN": "中文"
    }

    def __init__(self, language: str = "zh_CN"):
        """
        初始化翻译器

        参数:
            language: 语言代码（如 'en_US', 'zh_CN'）
        """
        self.current_language = language
        self.translations: Dict[str, Dict] = {}
        self._load_translations()

    def _get_i18n_dir(self) -> Path:
        """获取i18n目录路径"""
        return get_resource_path("src/i18n")

    def _load_translations(self) -> None:
        """加载所有可用的翻译文件"""
        i18n_dir = self._get_i18n_dir()

        for lang_code in self.AVAILABLE_LANGUAGES:
            file_path = i18n_dir / f"{lang_code}.json"
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.translations[lang_code] = json.load(f)
                    logger.info(f"已加载翻译: {lang_code}")
                except Exception as e:
                    logger.error(f"加载 {lang_code} 失败: {e}")
                    self.translations[lang_code] = {}

    def set_language(self, language: str) -> bool:
        """
        设置当前语言

        参数:
            language: 语言代码

        返回:
            如果语言设置成功则返回True
        """
        if language in self.AVAILABLE_LANGUAGES:
            self.current_language = language
            logger.info(f"语言已设置为: {language}")
            return True
        else:
            logger.warning(f"未知语言: {language}")
            return False

    def get_language(self) -> str:
        """获取当前语言代码"""
        return self.current_language

    def get_language_name(self, code: Optional[str] = None) -> str:
        """获取语言显示名称"""
        code = code or self.current_language
        return self.AVAILABLE_LANGUAGES.get(code, code)

    def t(self, key: str, **kwargs) -> str:
        """
        获取翻译字符串

        参数:
            key: 翻译键（支持点号表示法，如 "menu.file"）
            **kwargs: 格式化参数

        返回:
            翻译后的字符串，如果未找到则返回键
        """
        # 尝试当前语言
        value = self._get_nested(self.translations.get(self.current_language, {}), key)

        # 回退到默认语言
        if value is None and self.current_language != self.DEFAULT_LANGUAGE:
            value = self._get_nested(self.translations.get(self.DEFAULT_LANGUAGE, {}), key)

        # 如果未找到则返回键
        if value is None:
            logger.debug(f"未找到翻译: {key}")
            return key

        # 使用参数格式化
        if kwargs:
            try:
                value = value.format(**kwargs)
            except KeyError:
                pass

        return value

    def _get_nested(self, data: Dict, key: str) -> Optional[str]:
        """
        使用点号表示法获取嵌套字典值

        参数:
            data: 要搜索的字典
            key: 点分隔的键路径

        返回:
            如果找到则返回值，否则返回None
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
        获取所有翻译键

        参数:
            prefix: 键前缀过滤

        返回:
            匹配键的列表
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


# 全局翻译器实例
_translator: Optional[Translator] = None


def get_translator() -> Translator:
    """获取全局翻译器实例"""
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator


def set_language(language: str) -> bool:
    """设置全局语言"""
    return get_translator().set_language(language)


def t(key: str, **kwargs) -> str:
    """
    使用全局翻译器翻译键

    参数:
        key: 翻译键
        **kwargs: 格式化参数

    返回:
        翻译后的字符串
    """
    return get_translator().t(key, **kwargs)
