"""YAML configuration loading and argparse-compatible validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any, Sequence

import yaml


COMMANDS = {"dedup", "remove-like", "select"}


class _Yaml12SafeLoader(yaml.SafeLoader):
    """Safe loader with YAML 1.2 boolean rules (true/false, not yes/no)."""


_Yaml12SafeLoader.yaml_implicit_resolvers = {
    key: list(resolvers)
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
for first_character in "yYnNoOtTfF":
    resolvers = _Yaml12SafeLoader.yaml_implicit_resolvers.get(first_character, [])
    _Yaml12SafeLoader.yaml_implicit_resolvers[first_character] = [
        resolver for resolver in resolvers if resolver[0] != "tag:yaml.org,2002:bool"
    ]
_Yaml12SafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded or validated."""


def load_config(argv: Sequence[str]) -> tuple[str | None, dict[str, Any]]:
    """Load the file named by ``--config`` without parsing the full CLI."""
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known, _ = bootstrap.parse_known_args(argv)
    if known.config is None:
        return None, {}

    path = Path(known.config)
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.load(handle, Loader=_Yaml12SafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load config '{path}': {exc}") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ConfigError("config root must be a YAML mapping")
    if any(not isinstance(key, str) for key in loaded):
        raise ConfigError("config option names must be strings")

    command = loaded.get("command", "dedup")
    if command not in COMMANDS:
        choices = ", ".join(sorted(COMMANDS))
        raise ConfigError(f"config 'command' must be one of: {choices}")
    return str(path), loaded


def cli_command(argv: Sequence[str]) -> str:
    """Return the workflow explicitly selected by the CLI, if any."""
    for value in argv:
        if value in {"remove-like", "select"}:
            return value
    return "dedup"


def apply_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config: dict[str, Any],
    command: str,
) -> argparse.Namespace:
    """Apply config after CLI parsing, giving config values final precedence."""
    if not config:
        return args

    configured_command = config.get("command", "dedup")
    if configured_command != command:
        parser.error(
            f"config command '{configured_command}' conflicts with CLI command '{command}'"
        )

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "config", "command"}
    }
    unknown = sorted(set(config) - set(actions) - {"command"})
    if unknown:
        parser.error(f"unknown config option(s): {', '.join(unknown)}")

    for key, value in config.items():
        if key == "command":
            continue
        action = actions[key]
        try:
            converted = _convert_value(action, value)
        except (TypeError, ValueError) as exc:
            parser.error(f"invalid config value for '{key}': {exc}")
        setattr(args, key, converted)
    return args


def _convert_value(action: argparse.Action, value: Any) -> Any:
    if value is None:
        if action.default is not None:
            raise TypeError("null is not allowed")
        return None

    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if not isinstance(value, bool):
            raise TypeError("expected true or false")
        return value

    is_list = action.nargs in {"+", "*"}
    if is_list and not isinstance(value, list):
        raise TypeError("expected a YAML list")
    if not is_list and isinstance(value, list):
        raise TypeError("expected a scalar value")
    values = value if is_list else [value]

    converted = []
    converter = action.type or str
    for item in values:
        if isinstance(item, (dict, list)):
            raise TypeError("expected a scalar value")
        converted_item = converter(item if isinstance(item, str) else str(item))
        if action.choices is not None and converted_item not in action.choices:
            choices = ", ".join(str(choice) for choice in action.choices)
            raise ValueError(f"expected one of: {choices}")
        converted.append(converted_item)
    return converted if is_list else converted[0]
