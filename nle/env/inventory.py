import collections
import re
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import Optional

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
    nethack.Command.CAST: "cast",
    nethack.Command.DIP: "dip",
    nethack.Command.DROP: "drop",
    nethack.Command.DROPTYPE: "drop",
    nethack.Command.ENGRAVE: "engrave",
    nethack.Command.FIRE: "fire",
    nethack.Command.INVOKE: "invoke",
    nethack.Command.LOOT: "loot",
    nethack.Command.OFFER: "offer",
    nethack.Command.PAY: "pay",
    nethack.Command.PICKUP: "pickup",
    nethack.Command.PUTON: "puton",
    nethack.Command.REMOVE: "remove",
    nethack.Command.TAKEOFF: "takeoff",
    nethack.Command.TAKEOFFALL: "takeoffall",
    nethack.Command.TIP: "tip",
    nethack.Command.RUB: "rub",
    nethack.Command.WEAR: "wear",
    nethack.Command.WIELD: "wield",
    nethack.Command.THROW: "throw",
    nethack.Command.UNTRAP: "untrap",
}

UNKNOWN_ACTION_LABEL = "unknown"


@dataclass
class InventorySnapshot:
    counts: collections.Counter
    categories: collections.Counter
    name_to_category: Dict[str, str]


@dataclass
class ItemStats:
    acquired: int = 0
    actions: collections.Counter = field(default_factory=collections.Counter)


class InventoryStatsTracker:
    """Keeps track of inventory additions/usages within an episode."""

    _PARENS = re.compile(r"\([^)]*\)")
    _LETTER_PREFIX = re.compile(r"^[a-zA-Z]\s*[-)]\s+")

    def __init__(
        self,
        inv_strs_index: Optional[int],
        inv_oclasses_index: Optional[int],
    ):
        self._inv_strs_index = inv_strs_index
        self._inv_oclasses_index = inv_oclasses_index
        self._current_snapshot: Optional[InventorySnapshot] = None
        self._reset_stats()

    def enabled(self) -> bool:
        return self._inv_strs_index is not None and self._inv_oclasses_index is not None

    def start_episode(self, observation):
        if not self.enabled():
            return
        self._reset_stats()
        snapshot = self._extract_snapshot(observation)
        if snapshot is None:
            self._current_snapshot = None
            return
        self._current_snapshot = snapshot
        self._register_initial_inventory(snapshot)

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

        command_label = TRACKED_COMMANDS.get(action_value)
        if command_label:
            self._pending_action_label = command_label

        action_label = self._action_to_label(action_value)
        effective_label = self._pending_action_label or action_label

        used = self._register_usage(
            effective_label, self._current_snapshot, next_snapshot
        )
        if used and self._pending_action_label:
            self._pending_action_label = None

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
            "inv_by_name": self._stats_to_dict(self._items_by_name),
            "inv_by_category": self._stats_to_dict(self._categories_by_name),
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
        self._items_by_name: Dict[str, ItemStats] = collections.defaultdict(ItemStats)
        self._categories_by_name: Dict[str, ItemStats] = collections.defaultdict(
            ItemStats
        )
        self._current_snapshot = None
        self._pending_action_label: Optional[str] = None

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
                category = after.name_to_category.get(name)
                self._record_acquisition(name, diff, category)

    def _register_initial_inventory(self, snapshot: InventorySnapshot) -> None:
        for name, count in snapshot.counts.items():
            category = snapshot.name_to_category.get(name)
            self._record_acquisition(name, count, category)

    def _record_acquisition(self, name: str, quantity: int, category: Optional[str]):
        if quantity <= 0:
            return
        self._pickups_by_name[name] += quantity
        if category:
            self._pickups_by_class[category] += quantity
            category_stats = self._categories_by_name[category]
            category_stats.acquired += quantity
        stats = self._items_by_name[name]
        stats.acquired += quantity

    def _register_usage(
        self, action_label: str, before: InventorySnapshot, after: InventorySnapshot
    ) -> bool:
        used_any = False
        for name, prev_count in before.counts.items():
            current = after.counts.get(name, 0)
            diff = prev_count - current
            if diff > 0:
                self._uses_by_action[action_label] += diff
                self._uses_by_name[name][action_label] += diff
                category = before.name_to_category.get(name)
                if category:
                    self._uses_by_class[category][action_label] += diff
                    category_stats = self._categories_by_name[category]
                    category_stats.actions[action_label] += diff
                stats = self._items_by_name[name]
                stats.actions[action_label] += diff
                used_any = True
        return used_any

    @classmethod
    def _decode_line(cls, raw_line) -> str:
        data = raw_line.tobytes() if hasattr(raw_line, "tobytes") else bytes(raw_line)
        data = data.split(b"\0", 1)[0].decode("utf-8", "ignore").strip()
        return data

    def _action_to_label(self, action_value) -> str:
        if action_value in TRACKED_COMMANDS:
            return TRACKED_COMMANDS[action_value]
        enum_type = type(action_value)
        name = getattr(action_value, "name", "")
        compass = getattr(nethack, "CompassDirection", None)
        if compass is not None and isinstance(action_value, compass):
            return f"move_{name.lower()}"
        compass_long = getattr(nethack, "CompassDirectionLonger", None)
        if compass_long is not None and isinstance(action_value, compass_long):
            return f"move_{name.lower()}"
        misc_dir = getattr(nethack, "MiscDirection", None)
        if misc_dir is not None and isinstance(action_value, misc_dir):
            mapping = {
                "UP": "move_up",
                "DOWN": "move_down",
                "WAIT": "wait",
            }
            return mapping.get(name, f"move_{name.lower()}")
        misc_action = getattr(nethack, "MiscAction", None)
        if misc_action is not None and isinstance(action_value, misc_action):
            return name.lower()
        command_enum = getattr(nethack, "Command", None)
        if command_enum is not None and isinstance(action_value, command_enum):
            return name.lower()
        return UNKNOWN_ACTION_LABEL

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

    def _stats_to_dict(self, mapping):
        result = {}
        for key, stats in mapping.items():
            actions = self._counter_to_dict(stats.actions)
            entry = {"acquired": int(stats.acquired), "actions": actions}
            if entry["acquired"] or actions:
                result[key] = entry
        return result
