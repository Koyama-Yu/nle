import collections
import re
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from nle import nethack


CATEGORY_NAMES = {
    nethack.WEAPON_CLASS: "weapon",
    nethack.ARMOR_CLASS: "armor",
    nethack.RING_CLASS: "ring",
    nethack.AMULET_CLASS: "amulet",
    nethack.TOOL_CLASS: "tool",
    nethack.FOOD_CLASS: "food",
    nethack.POTION_CLASS: "potion",
    nethack.SCROLL_CLASS: "scroll",
    nethack.SPBOOK_CLASS: "spellbook",
    nethack.WAND_CLASS: "wand",
    nethack.COIN_CLASS: "gold",
    nethack.GEM_CLASS: "gem",
    nethack.ROCK_CLASS: "rock",
    nethack.BALL_CLASS: "ball",
    nethack.CHAIN_CLASS: "chain",
    nethack.VENOM_CLASS: "venom",
}


TRACKED_COMMANDS = {
    nethack.Command.EAT: "eat",
    nethack.Command.QUAFF: "quaff",
    nethack.Command.READ: "read",
    nethack.Command.ZAP: "zap",
    nethack.Command.APPLY: "apply",
    nethack.Command.INVOKE: "invoke",
    nethack.Command.PUTON: "puton",
    nethack.Command.REMOVE: "remove",
    nethack.Command.TAKEOFF: "takeoff",
    nethack.Command.WEAR: "wear",
    nethack.Command.WIELD: "wield",
    nethack.Command.THROW: "throw",
}


@dataclass
class InventorySnapshot:
    counts: collections.Counter
    categories: collections.Counter
    name_to_category: Dict[str, str]


class InventoryStatsTracker:
    """Keeps track of inventory additions/usages within an episode."""

    _PARENS = re.compile(r"\([^)]*\)")
    _LETTER_PREFIX = re.compile(r"^[a-zA-Z]\s*[-)]\s+")

    def __init__(self, inv_strs_index: Optional[int], inv_oclasses_index: Optional[int]):
        self._inv_strs_index = inv_strs_index
        self._inv_oclasses_index = inv_oclasses_index
        self._current_snapshot: Optional[InventorySnapshot] = None
        self._reset_stats()

    def enabled(self) -> bool:
        return (
            self._inv_strs_index is not None and self._inv_oclasses_index is not None
        )

    def start_episode(self, observation):
        if not self.enabled():
            return
        self._reset_stats()
        self._current_snapshot = self._extract_snapshot(observation)

    def record_step(self, action_value: int, observation):
        if not self.enabled():
            return
        if self._current_snapshot is None:
            self._current_snapshot = self._extract_snapshot(observation)
            return

        next_snapshot = self._extract_snapshot(observation)
        if next_snapshot is None:
            return

        self._register_pickups(self._current_snapshot, next_snapshot)

        action_label = TRACKED_COMMANDS.get(action_value)
        if action_label:
            self._register_usage(action_label, self._current_snapshot, next_snapshot)

        self._current_snapshot = next_snapshot

    def finalize_episode(self) -> Dict[str, Dict]:
        if not self.enabled():
            return {}
        metadata = {
            "inv_pickups_by_name": self._counter_to_dict(self._pickups_by_name),
            "inv_pickups_by_class": self._counter_to_dict(self._pickups_by_class),
            "inv_uses_by_action": self._counter_to_dict(self._uses_by_action),
            "inv_uses_by_name": self._nested_counter_to_dict(self._uses_by_name),
            "inv_uses_by_class": self._nested_counter_to_dict(self._uses_by_class),
        }
        has_data = any(metadata.values())
        self._reset_stats()
        return metadata if has_data else {}

    def _reset_stats(self):
        self._pickups_by_name = collections.Counter()
        self._pickups_by_class = collections.Counter()
        self._uses_by_action = collections.Counter()
        self._uses_by_name = collections.defaultdict(collections.Counter)
        self._uses_by_class = collections.defaultdict(collections.Counter)
        self._current_snapshot = None

    def _extract_snapshot(self, observation) -> Optional[InventorySnapshot]:
        if observation is None:
            return None

        inv_strs = observation[self._inv_strs_index]
        inv_oclasses = observation[self._inv_oclasses_index]

        counts = collections.Counter()
        categories = collections.Counter()
        name_to_category: Dict[str, str] = {}

        for raw_line, oclass in zip(inv_strs, inv_oclasses):
            if int(oclass) == int(nethack.MAXOCLASSES):
                continue
            if not np.any(raw_line):
                continue
            decoded = self._decode_line(raw_line)
            if not decoded:
                continue
            normalized = self._normalize_name(decoded)
            if not normalized:
                continue
            counts[normalized] += 1
            category_name = CATEGORY_NAMES.get(int(oclass), "unknown")
            categories[category_name] += 1
            name_to_category.setdefault(normalized, category_name)

        return InventorySnapshot(
            counts=counts, categories=categories, name_to_category=name_to_category
        )

    def _register_pickups(
        self, before: InventorySnapshot, after: InventorySnapshot
    ) -> None:
        for name, count in after.counts.items():
            diff = count - before.counts.get(name, 0)
            if diff > 0:
                self._pickups_by_name[name] += diff
                category = after.name_to_category.get(name)
                if category:
                    self._pickups_by_class[category] += diff

    def _register_usage(
        self, action_label: str, before: InventorySnapshot, after: InventorySnapshot
    ) -> None:
        for name, prev_count in before.counts.items():
            current = after.counts.get(name, 0)
            diff = prev_count - current
            if diff > 0:
                self._uses_by_action[action_label] += diff
                self._uses_by_name[name][action_label] += diff
                category = before.name_to_category.get(name)
                if category:
                    self._uses_by_class[category][action_label] += diff

    @classmethod
    def _decode_line(cls, raw_line) -> str:
        data = raw_line.tobytes() if hasattr(raw_line, "tobytes") else bytes(raw_line)
        data = data.split(b"\0", 1)[0].decode("utf-8", "ignore").strip()
        return data

    def _normalize_name(self, text: str) -> str:
        text = text.strip()
        match = self._LETTER_PREFIX.match(text)
        if match:
            text = text[match.end() :]
        if text.startswith("- "):
            text = text[2:]
        text = text.strip()
        text = self._PARENS.sub("", text)
        text = re.sub(r"\s+", " ", text)
        for prefix in ("the ", "an ", "a "):
            if text.lower().startswith(prefix):
                text = text[len(prefix) :]
                break
        text = re.sub(r"^\d+\s+", "", text)
        return text.lower().strip()

    @staticmethod
    def _counter_to_dict(counter):
        return {key: int(val) for key, val in counter.items() if val}

    @staticmethod
    def _nested_counter_to_dict(nested):
        result = {}
        for key, counter in nested.items():
            as_dict = {
                inner_key: int(inner_val) for inner_key, inner_val in counter.items()
            }
            if as_dict:
                result[key] = as_dict
        return result
