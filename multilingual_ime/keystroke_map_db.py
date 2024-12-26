from pathlib import Path
import sqlite3

from .trie import modified_levenshtein_distance
from .core.custom_decorators import lru_cache_with_doc

# from fuzzywuzzy import fuzz

MAX_LEVENSHTEIN_DISTANCE = 1


class KeystrokeMappingDB:
    def __init__(self, db_path: str):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database file {db_path} not found")

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._cursor = self._conn.cursor()
        self._conn.create_function("levenshtein", 2, modified_levenshtein_distance)

    def get(self, keystroke: str) -> list[str]:
        self._cursor.execute(
            "SELECT keystroke, word, frequency FROM keystroke_map WHERE keystroke = ?",
            (keystroke,),
        )
        return self._cursor.fetchall()

    @lru_cache_with_doc(maxsize=128)
    def fuzzy_get(
        self, keystroke: str, max_distance: int = MAX_LEVENSHTEIN_DISTANCE
    ) -> list[str]:
        self._cursor.execute(
            f"SELECT keystroke, word, frequency FROM keystroke_map WHERE levenshtein(keystroke, ?) <= {max_distance}",
            (keystroke,),
        )
        return self._cursor.fetchall()

    @lru_cache_with_doc(maxsize=128)
    def get_closest(self, keystroke: str) -> list[tuple[str, str, int]]:
        """
        Get the closest words to the given keystroke.

        Args:
            keystroke (str): The keystroke to search for

        Returns:
            list: A list of **tuples (keystroke, word, frequency)** containing the closest words
        """

        # Search for the direct match first
        result = self.get(keystroke)
        if result:
            return result

        for i in range(MAX_LEVENSHTEIN_DISTANCE + 1):
            result = self.fuzzy_get(keystroke, i)
            if result:
                return result
        return []

    def create_table(self):
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS keystroke_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keystroke TEXT,
                word TEXT,
                frequency INTEGER
            )
            """
        )
        self._cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_keystroke ON keystroke_map (keystroke)"
        )
        self._conn.commit()

    def insert(self, keystroke: str, word: str, frequency: int):
        if (keystroke, word, frequency) not in self.get(keystroke):
            self._cursor.execute(
                "INSERT INTO keystroke_map (keystroke, word, frequency) VALUES (?, ?, ?)",
                (keystroke, word, frequency),
            )
            self._conn.commit()

    def insert_many(self, data: list[tuple[str, str, int]]):
        for keystroke, word, frequency in data:
            self.insert(keystroke, word, frequency)

    def __del__(self):
        self._conn.close()

    def keystroke_exists(self, keystroke: str) -> bool:
        return bool(self.get(keystroke))

    def is_word_within_distance(
        self, keystroke: str, distance: int = MAX_LEVENSHTEIN_DISTANCE
    ) -> bool:
        return self.closest_word_distance(keystroke) <= distance

    def closest_word_distance(self, keystroke: str) -> int:
        distance = 0
        while True:
            if self.fuzzy_get(keystroke, distance):
                return distance
            else:
                distance += 1

    def word_to_keystroke(self, word: str) -> str:
        self._cursor.execute(
            "SELECT keystroke FROM keystroke_map WHERE word = ?", (word,)
        )
        if word := self._cursor.fetchone():
            return word[0]
        else:
            return None

    def word_exists(self, word: str) -> bool:
        self._cursor.execute("SELECT word FROM keystroke_map WHERE word = ?", (word,))
        return bool(self._cursor.fetchone())

    def increment_word_frequency(self, word: str):
        self._cursor.execute(
            "UPDATE keystroke_map SET frequency = frequency + 1 WHERE word = ?",
            (word,),
        )
        self._conn.commit()

    def get_word_frequency(self, word: str) -> int:
        self._cursor.execute(
            "SELECT frequency FROM keystroke_map WHERE word = ?", (word,)
        )
        if frequency := self._cursor.fetchone():
            return frequency[0]
        else:
            return 0


if __name__ == "__main__":
    import pathlib
    import time
    db_path = pathlib.Path(__file__).parent / "src" / "bopomofo_keystroke_map.db"

    db = KeystrokeMappingDB(
        db_path
    )
    start_time = time.time()
    test_cases = ["su3", "cl3", "t ", "gjo4", "s86", "1p4", "ru04", "1l4", "fu04", "503", "g8 ", "al6", "xu4", "b.6", "wj/ ", "1u/4", "fu06", "1l4", "ejo3", "w.6"]
    for test_case in test_cases:
        print(db.get_closest(test_case))
    print("Time taken: ", time.time() - start_time)
    print("Average time taken: ", (time.time() - start_time) / len(test_cases), f"({len(test_cases)} test cases)")

    print(db.get_closest("u04counsel"))
    print(db.closest_word_distance("u04counsel"))
    print(db.is_word_within_distance("u04counsel"))
