import sqlite3
from dataclasses import dataclass

@dataclass
class Item:
    guid: str
    feed_id: int
    title: str
    original_xml: str
    original_url: str
    original_type: str
    pub_date: str|None
    transcript_tokens: str|None
    transcript_timestamps: str|None
    audio_filepath: str|None
    error: str|None

def init_db(cursor: sqlite3.Cursor):
    # cursor.execute("drop table if exists feeds;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feeds (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            xml_noitems TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # cursor.execute("drop table if exists items;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS items (
            guid TEXT PRIMARY KEY,
            feed_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            original_xml TEXT NOT NULL,
            original_url TEXT NOT NULL,
            original_type TEXT NOT NULL,
            pub_date TEXT,
            transcript_tokens TEXT,
            transcript_timestamps TEXT,
            audio_filepath TEXT,
            error TEXT,
            FOREIGN KEY (feed_id) REFERENCES feeds(id)
        );
    """)
