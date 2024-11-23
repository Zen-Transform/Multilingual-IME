import keyboard
import time
import threading

from .ime_handler import IMEHandler
from .ime import (
    BOPOMOFO_VALID_KEYSTROKE_SET,
    ENGLISH_VALID_KEYSTROKE_SET,
    PINYIN_VALID_KEYSTROKE_SET,
    CANGJIE_VALID_KEYSTROKE_SET,
)

TOTAL_VALID_KEYSTROKE_SET = (
    BOPOMOFO_VALID_KEYSTROKE_SET.union(ENGLISH_VALID_KEYSTROKE_SET)
    .union(PINYIN_VALID_KEYSTROKE_SET)
    .union(CANGJIE_VALID_KEYSTROKE_SET)
)


class KeyEventProcessor:
    def __init__(self, time_threshold):
        self.time_threshold = time_threshold
        self.last_key_event = None
        self.timer = None

        self.dynamic_keystrokes = ""
        self.dynamic_output = ""
        self.show_output = ""

        self.counter = 0
        self.avg_time = 0

        keyboard.on_press(self.process_key_event)

    def key_handler(self, event):
        if event.name == "enter":
            self.dynamic_keystrokes = ""
        elif event.name == "backspace":
            self.dynamic_keystrokes = self.dynamic_keystrokes[:-1]
        elif event.name == "space":
            self.dynamic_keystrokes += " "
            self.show_output = self.show_output + " "
        elif event.name in TOTAL_VALID_KEYSTROKE_SET:
            self.dynamic_keystrokes += event.name
            self.show_output = self.show_output + event.name
        else:
            # print(f"Invalid key: {event.name}")
            pass

    def process_key_event(self, event):
        self.last_key_event = event

        self.key_handler(event)
        

        print(f"\r{self.show_output} {' ' * (50 - len(self.show_output))}", end="")

        # Cancel any existing timer
        if self.timer is not None:
            self.timer.cancel()

        # Start a new timer to process the latest key event after TIME_THRESHOLD seconds
        self.timer = threading.Timer(self.time_threshold, self.execute_last_event)
        self.timer.start()

    def execute_last_event(self):
        if self.last_key_event is not None:
            self.counter += 1

            start_time = time.time()
            # result = my_IMEHandler.get_candidate_sentences(self.dynamic_keystrokes)
            self.dynamic_output = my_IMEHandler.get_best_sentence(
                self.dynamic_keystrokes
            )
            end_time = time.time()
            self.avg_time = (
                self.avg_time * (self.counter - 1) + end_time - start_time
            ) / self.counter
            self.show_output = self.dynamic_output
            print(f"\r{self.show_output} {' ' * (50 - len(self.show_output))}", end="")

    def run(self):
        keyboard.wait("esc")


if __name__ == "__main__":
    stat_time = time.time()
    my_IMEHandler = IMEHandler(verbose_mode=False)
    print("Initialization time: ", time.time() - stat_time)
    processor = KeyEventProcessor(time_threshold=0.2)
    processor.run()
