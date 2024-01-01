import json
from pathlib import Path

import pytest

from vizard.configs import JsonConfigHandler
from vizard.snorkel import LABEL_MODEL_CONFIGS


json_config_handler = JsonConfigHandler()


class TestJsonConfigHandler:
    """Tests for the JsonConfigHandler class."""

    def test_load_valid_json(self, tmp_path):
        """Tests loading a valid JSON file."""

        dest_path = tmp_path / "test_config.json"
        dest_path.write_text('{"a": 1, "b": 2}')

        configs = json_config_handler.load(dest_path)
        assert configs == {"a": 1, "b": 2}

    def test_load_invalid_json(self, tmp_path):
        """Tests loading an invalid JSON file."""
        dest_path = tmp_path / "invalid_config.json"
        dest_path.write_text("invalid json")

        with pytest.raises(json.JSONDecodeError):
            json_config_handler.load(dest_path)

    def test_load_nonexistent_file(self):
        """Tests loading a non-existent file."""
        dest_path = Path("nonexistent.json")
        with pytest.raises(FileNotFoundError):
            json_config_handler.load(dest_path)

    def test_parse_label_model_configs(self):
        """Tests parsing configs for the LabelModel class."""
        parsed_configs = json_config_handler.parse(
            filename=LABEL_MODEL_CONFIGS, target="LabelModel"
        )
        assert parsed_configs["method_init"]["cardinality"] == 2
        assert parsed_configs["method_fit"]["optimizer"] == "adam"
