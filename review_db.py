
import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path("outputs/medicalai_reviews.db")


def initialize_database():

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            decision TEXT NOT NULL,
            reviewer TEXT NOT NULL,
            comments TEXT,
            reviewed_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_review(
    image_name,
    predicted_class,
    confidence,
    decision,
    reviewer,
    comments
):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO reviews (
            image_name,
            predicted_class,
            confidence,
            decision,
            reviewer,
            comments,
            reviewed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        image_name,
        predicted_class,
        confidence,
        decision,
        reviewer,
        comments,
        datetime.now().isoformat(timespec="seconds")
    ))

    connection.commit()
    connection.close()


def get_all_reviews():

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            image_name,
            predicted_class,
            confidence,
            decision,
            reviewer,
            comments,
            reviewed_at
        FROM reviews
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    connection.close()

    return records


if __name__ == "__main__":

    initialize_database()

    print("MedicalAI review database initialized!")
    print("Database location:", DB_PATH)
