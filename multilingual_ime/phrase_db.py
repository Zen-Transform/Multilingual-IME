import threading
from pathlib import Path

import sqlite3


class PhraseDataBase:
    """
    A class to handle the phrase database operations.
    This class provides methods to create a phrase table, insert phrases,
    retrieve phrases, update phrase frequencies, and delete phrases.
    It uses SQLite for database operations and is thread-safe.
    """

    def __init__(self, db_path: str | Path) -> None:
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database file {db_path} not found")

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._cursor = self._conn.cursor()
        self._lock = threading.Lock()

    def __del__(self) -> None:
        self._conn.commit()
        self._conn.close()

    def create_phrase_table(self) -> None:
        """Create the phrase table if it does not exist."""
        with self._lock:
            self._cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS phrase_table (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    initial_word TEXT,
                    phrase TEXT,
                    frequency INTEGER
                )
                """
            )

            self._cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS index_initial_word ON phrase_table (initial_word)"""
            )
            self._conn.commit()

    def get_phrase(self, phrase: str) -> list[str]:
        """
        Retrieve a phrase from the database.
        Args:
            phrase (str): The phrase to retrieve.
        Returns:
            list[tuple[str, int]]: A list of tuples containing the phrase and its frequency.
        """

        if not phrase:
            return []

        with self._lock:
            self._cursor.execute(
                "SELECT phrase FROM phrase_table WHERE phrase = ?", (phrase,)
            )
            return [row[0] for row in self._cursor.fetchall()]

    def get_phrase_with_prefix(self, prefix: str) -> list[tuple[str, int]]:
        """
        Retrieve phrases that start with a given prefix.
        Args:
            prefix (str): The prefix to search for.

        Returns:
        list[tuple[str, int]]: A list of tuples containing phrases and their frequencies.
        """
        if not prefix:
            return []
        with self._lock:
            self._cursor.execute(
                "SELECT phrase, frequency FROM phrase_table WHERE initial_word = ?",
                (prefix,),
            )
            return list(self._cursor.fetchall())

    def insert_phrase(self, phrase: str, frequency: int = 0) -> None:
        """
        Insert a new phrase into the database.

        Args:
            phrase (str): The phrase to insert.
            frequency (int): The frequency of the phrase.

        Raises:
            ValueError: If the phrase already exists in the database.
        """

        if not phrase:
            return

        if self.get_phrase(phrase):
            raise ValueError(f"Phrase '{phrase}' already exists in the database.")

        if not self.get_phrase(phrase):
            with self._lock:
                initial_word = phrase[0]
                self._cursor.execute(
                    "INSERT INTO phrase_table (initial_word, phrase, frequency) VALUES (?, ?, ?))",
                    (initial_word, phrase, frequency),
                )
                self._conn.commit()

    def update_phrase_frequency(self, phrase: str, frequency: int) -> None:
        """
        Update the frequency of an existing phrase in the database.

        Args:
            phrase (str): The phrase to update.
            frequency (int): The new frequency of the phrase.
        Raises:
            ValueError: If the phrase does not exist in the database.
        """

        if not self.get_phrase(phrase):
            raise ValueError(f"Phrase '{phrase}' does not exist in the database.")

        with self._lock:
            self._cursor.execute(
                "UPDATE phrase_table SET frequency = ? WHERE phrase = ?",
                (frequency, phrase),
            )
            self._conn.commit()

    def delete_phrase(self, phrase: str) -> None:
        """
        Delete a phrase from the database.
        Args:
            phrase (str): The phrase to delete.
        Raises:
            ValueError: If the phrase does not exist in the database.
        """
        if not self.get_phrase(phrase):
            raise ValueError(f"Phrase '{phrase}' does not exist in the database.")

        with self._lock:
            self._cursor.execute("DELETE FROM phrase_table WHERE phrase = ?", (phrase,))
            self._conn.commit()

    def increment_phrase_frequency(self, phrase: str) -> None:
        """
        Increment the frequency of a phrase by 1.
        Args:
            phrase (str): The phrase whose frequency is to be incremented.
        Raises:
            ValueError: If the phrase does not exist in the database.
        """
        if not self.get_phrase(phrase):
            raise ValueError(f"Phrase '{phrase}' does not exist in the database.")

        with self._lock:
            self._cursor.execute(
                "UPDATE phrase_table SET frequency = frequency + 1 WHERE phrase = ?",
                (phrase,),
            )
            self._conn.commit()


if __name__ == "__main__":
    db = PhraseDataBase(Path(__file__).parent / "src" / "chinese_phrase.db")
    db.create_phrase_table()
