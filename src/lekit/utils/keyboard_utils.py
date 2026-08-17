from typing import Callable, Union

from pynput import keyboard

# Define a type for keys that can be either a Key or a KeyCode
KeyType = Union[keyboard.Key, keyboard.KeyCode]


class KeyboardListener:
    # Define a common keys mapping for easier binding
    _KEY_MAP = {
        "space": keyboard.Key.space,
        "esc": keyboard.Key.esc,
        "enter": keyboard.Key.enter,
        "tab": keyboard.Key.tab,
        "shift": keyboard.Key.shift,
        "ctrl": keyboard.Key.ctrl,
        "alt": keyboard.Key.alt,
        "backspace": keyboard.Key.backspace,
        "caps_lock": keyboard.Key.caps_lock,
        "f1": keyboard.Key.f1,
        "f2": keyboard.Key.f2,
        "f3": keyboard.Key.f3,
        "f4": keyboard.Key.f4,
        "f5": keyboard.Key.f5,
        "f6": keyboard.Key.f6,
        "f7": keyboard.Key.f7,
        "f8": keyboard.Key.f8,
        "f9": keyboard.Key.f9,
        "f10": keyboard.Key.f10,
        "f11": keyboard.Key.f11,
        "f12": keyboard.Key.f12,
        "up": keyboard.Key.up,
        "down": keyboard.Key.down,
        "left": keyboard.Key.left,
        "right": keyboard.Key.right,
    }

    def __init__(self):
        # Store the mapping of keys to callback functions {key: callback}
        self.bindings = {}
        self.listener = None

    def _parse_key(self, key_str: str) -> Union[keyboard.Key, keyboard.KeyCode]:
        """Parse a key string into a Key or KeyCode object"""
        key_str = key_str.lower().strip()

        # 1. Check for control keys (like 'space', 'esc')
        if key_str in self._KEY_MAP:
            return self._KEY_MAP[key_str]

        # 2. Check for single-character keys
        if len(key_str) == 1:
            return keyboard.KeyCode.from_char(key_str)

        raise ValueError(f"Unsupported key: '{key_str}'.")

    def bind_key(self, key_name: str, callback: Callable[[KeyType], None], description: str = ""):
        """
        bind a key to a callback function
        :param key: the key to bind, e.g., keyboard.Key.enter or keyboard.KeyCode.from_char('a')
        :param callback: the callback function to call when the key is pressed
        :param description: a brief description of the key binding for user reference
        """
        try:
            real_key = self._parse_key(key_name)
        except ValueError:
            name = self.__class__.__name__
            print(f"[{name}] Error: Invalid key string '{key_name}'. Check your spelling.")
            return
        if real_key in self.bindings:
            callback_name = self.bindings[real_key][1].__name__ if self.bindings[real_key][1] else "None"
            print(f"[{name}] Warning: Key '{key_name}' is already bound to {callback_name}. Overwriting.")
        self.bindings[real_key] = (key_name, callback, description)  # Store the key name, callback, and description

    def bind(self, key_name: str, callback: Callable[[KeyType], None], description: str = ""):
        """Shortcut for bind_key"""
        self.bind_key(key_name, callback, description)

    def _on_press(self, key: KeyType):
        """Internal core method: handle key press events"""
        if key in self.bindings:
            key_name, callback, _ = self.bindings[key]
            callback(key_name)
            return

        if hasattr(key, "char") and key.char:
            for bound_key, (key_name, callback, _) in self.bindings.items():
                if hasattr(bound_key, "char") and bound_key.char == key.char:
                    callback(key_name)
                    break

    def start(self):
        """Start the keyboard listener in background mode."""
        if self.listener is None:
            self.listener = keyboard.Listener(on_press=self._on_press)
            self.listener.start()

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener = None


if __name__ == "__main__":
    import time

    keyboard_listener = KeyboardListener()

    keyboard_listener.bind_key("space", lambda key: print(f"[{key}] is Pressed"))
    keyboard_listener.bind_key("a", lambda key: print(f"[{key}] is Pressed"))
    keyboard_listener.bind_key("esc", lambda key: print(f"[{key}] is Pressed"))
    keyboard_listener.bind_key("enter", lambda key: print(f"[{key}] is Pressed"))
    keyboard_listener.bind_key("f1", lambda key: print(f"[{key}] is Pressed"))
    keyboard_listener.bind_key("f2", lambda key: print(f"[{key}] is Pressed"))
    keyboard_listener.bind_key("[", lambda key: print(f"[{key}] is Pressed"))

    keyboard_listener.start()

    try:
        while keyboard_listener.listener and keyboard_listener.listener.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        keyboard_listener.stop()
