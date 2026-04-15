from __future__ import annotations

from app.utils import safe_get_str


class TestSafeGetStr:
    def test_missing_key_returns_empty_string(self):
        assert safe_get_str({}, "missing") == ""

    def test_present_key_returns_stripped_value(self):
        assert safe_get_str({"key": "  value  "}, "key") == "value"

    def test_none_value_returns_none_string(self):
        assert safe_get_str({"key": None}, "key") == "None"

    def test_numeric_value_returns_string(self):
        assert safe_get_str({"key": 42}, "key") == "42"

    def test_custom_default(self):
        assert safe_get_str({}, "missing", "fallback") == "fallback"

    def test_empty_string_value(self):
        assert safe_get_str({"key": ""}, "key") == ""

    def test_whitespace_only_value(self):
        assert safe_get_str({"key": "   "}, "key") == ""

    def test_boolean_value(self):
        assert safe_get_str({"key": True}, "key") == "True"

    def test_default_with_whitespace(self):
        assert safe_get_str({}, "missing", "  padded  ") == "padded"
