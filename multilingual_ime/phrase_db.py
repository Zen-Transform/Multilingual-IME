from pathlib import Path

import sqlite3


class PhraseDataBase:
    def __init__(self, db_path: str) -> None:
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database file {db_path} not found")

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._cursor = self._conn.cursor()

    def __del__(self) -> None:
        self._conn.commit()
        self._conn.close()

    def create_table(self) -> None:
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS phrase_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phrase TEXT,
                frequency INTEGER
            )
            """
        )
        self._conn.commit()

    def getphrase(self, phrase: str) -> list[tuple[str, int]]:
        self._cursor.execute(
            f"SELECT phrase FROM phrase_table WHERE phrase = '{phrase}'"
        )
        return [row[0] for row in self._cursor.fetchall()]

    def get_phrase_with_prefix(self, prefix: str) -> list[tuple[str, int]]:
        if not prefix:
            return []

        self._cursor.execute(
            f"SELECT phrase, frequency FROM phrase_table WHERE phrase LIKE '{prefix}%'"
        )
        return self._cursor.fetchall()

    def insert(self, phrase: str, frequency: int) -> None:
        if not self.getphrase(phrase):
            self._cursor.execute(
                f"INSERT INTO phrase_table (phrase, frequency) VALUES ('{phrase}', {frequency})"
            )
            self._conn.commit()


if __name__ == "__main__":
    db = PhraseDataBase(Path(__file__).parent / "src" / "chinese_phrase.db")
    db.create_table()
