import os
import sqlite3
import xml.etree.ElementTree as et

from litestar import Litestar, get
from litestar.datastructures import State
from litestar.enums import MediaType
from litestar.static_files import create_static_files_router
import dotenv


def get_db_conn(app: Litestar) -> sqlite3.Connection:
    if not getattr(app.state, "conn", None):
        app.state.conn = sqlite3.connect("castward.db")
    return app.state.conn

def close_db_conn(app: Litestar) -> None:
    if getattr(app.state, "conn", None):
        app.state.conn.close()

@get("/feeds/{feed_guid:str}", media_type=MediaType.XML)
async def get_feed(feed_guid: str, state: State) -> str:
    cursor = state.conn.cursor()
    cursor.execute("SELECT xml_noitems FROM feeds where id = ?", (feed_guid,))
    feed = cursor.fetchone()
    if feed:
        xml = et.fromstring(feed[0])
    else:
        raise ValueError("Feed not found")
    cursor.execute("SELECT original_xml, audio_filepath FROM items WHERE feed_id = ? AND audio_filepath IS NOT NULL", (feed_guid,))
    items = cursor.fetchall()

    xml_channel = xml.find("channel")
    if xml_channel is None:
        raise ValueError("Invalid feed XML")
    for item in items:
        item_et = et.fromstring(item[0])
        enclosure = item_et.find("enclosure")
        if enclosure is not None:
            enclosure.set("url", os.path.join("/static", item[1]))
        xml_channel.append(item_et)

    return et.tostring(xml, encoding="unicode")


dotenv.load_dotenv()
app = Litestar(
    route_handlers=[
        get_feed,
        create_static_files_router(path="/static", directories=[os.getenv("AUDIO_DIR", "./audio")])
    ],
    on_startup=[get_db_conn],
    on_shutdown=[close_db_conn],
    debug=True
)
