# TODO:
#     Transcript chunking may be necessary for very long podcasts (> 2 hours, say).
#     Log errors to DB
#     Unify guid-derived filepaths
#     Ensure pipeline re-runs if the original audio is redownloaded for any reason

import sys
import os
import tomllib
import requests
import xml.etree.ElementTree as et
import sqlite3
from datetime import datetime, timedelta
from datetime import time as dt_time
import time
from email.utils import parsedate_tz
import dotenv
from typing import Tuple, List
from dataclasses import dataclass
import json
import subprocess
import math
import re
import logging
import uuid

import onnx_asr
# import nemo.collections.asr as nemo_asr
from google import genai

import db
from db import Item


logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(stream_handler)

PROMPT = """The following is a transcript of a podcast. For each sentence, there is a numeric ID followed by the transcribed sentence text. Identify any advertisements in the transcript and respond with a list of the start and end IDs of each advertisement.
Some advertisements may be for other podcasts - if you can tell that a sentence is promoting a different podcast (rather than this one), consider that to be an advertisement. Also consider any brief transitions _between_ advertisements to be part of the advertisement.

For example, if a transcript had IDs 1 through 10, and the sentences with ID 1, 2, 3, and 10 were part of an advertisement, you would respond "[[1, 3], [10, 10]]". If there are no advertisements, reply with an empty list: "[]".

Transcript:

"""

# TODO: this function is too complicated, break it up
def ingest_feed(cursor: sqlite3.Cursor, url: str):
    feed_data = requests.get(url)
    root = et.fromstring(feed_data.content)
    title_element = root[0].find("title")
    if title_element is not None:
        title = "🫷" + getattr(title_element, "text", "Unknown Title")
        title_element.text = title
    else:
        title = "🫷 Unknown Title"

    existing_feed_result = cursor.execute("SELECT id FROM feeds WHERE url=?", (url,)).fetchone()
    if existing_feed_result:
        feed_id = existing_feed_result[0]
    else:
        feed_id = str(uuid.uuid4())
        cursor.execute("INSERT INTO feeds (id, url, title) VALUES (?, ?, ?)", (feed_id, url, title))

    items = root[0].findall("item")
    to_remove = []
    for item in items:
        to_remove.append(item)
        try:
            # check if item already exists
            guid = getattr(item.find("guid"), "text")
            existing_item_result = cursor.execute("SELECT guid FROM items WHERE guid=?", (guid,)).fetchone()
            if existing_item_result:
                continue
            title = getattr(item.find("title"), "text")

            enclosure = item.find("enclosure")
            if enclosure is not None:
                enclosure_url = enclosure.get("url")
                enclosure_type = enclosure.get("type")
            else:
                raise ValueError("Enclosure not found")

            pubDate_raw = getattr(item.find("pubDate"), "text")
            pub_date = None
            # convert weird email time format to something which can be sorted lexicographically
            if pubDate_raw:
                pubDate_time = parsedate_tz(pubDate_raw)
                if pubDate_time:
                    pub_date = str(datetime(*pubDate_time[:6]))

            item_xml = et.tostring(item, encoding="unicode", method="xml")

            # TODO: handle null values in NOT NULL fields
            cursor.execute(
                "INSERT INTO items (guid, feed_id, title, original_xml, original_url, original_type, pub_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (guid, feed_id, title, item_xml, enclosure_url, enclosure_type, pub_date))
        except Exception as e:
            print(f"Error processing item: {e}")
            print(et.tostring(item).decode())
            raise
    for item in to_remove:
        root[0].remove(item)
    noitems_str = et.tostring(root, encoding="unicode", method="xml")
    cursor.execute("UPDATE feeds SET xml_noitems = ? WHERE id = ?", (noitems_str, feed_id))


def pathsafe_guid(guid: str) -> str:
    return re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", guid)


def download_item(url: str, dest: str):
    if os.path.exists(dest):
        return
    response = requests.get(url, stream=True)
    response.raise_for_status()
    os.makedirs(os.path.split(dest)[0], exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


# "but binary search would be faster!"
# see if I care
def nearest_timestamp_i(timestamp, timestamps):
    for i, t in enumerate(timestamps):
        if t > timestamp:
            if i > 0:
                return i - 1
            return 0
    return len(timestamps) - 1


# TODO: too many args
def shift_timestamps(timestamps, tokens, duration, overlap, start_time, is_first, is_last):
    start = nearest_timestamp_i(overlap / 2, timestamps)
    end = nearest_timestamp_i(duration + overlap / 2, timestamps)
    if is_first and is_last:
        return timestamps, tokens
    if is_first:
        return timestamps[:end], tokens[:end]
    if is_last:
        return [t + start_time for t in timestamps[start:]], tokens[start:]
    return [t + start_time for t in timestamps[start:end]], tokens[start:end]


def asr(asr_model, item: Item, original_audio_filepath: str) -> Tuple[List[str], List[float]]:
    duration_probe_result = subprocess.run(f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {original_audio_filepath}", shell=True, capture_output=True)
    if duration_probe_result.returncode != 0:
        raise Exception(f"Failed to probe duration of {original_audio_filepath}; {duration_probe_result.stderr.decode()}")

    chunks_path = os.path.join(
        os.path.split(original_audio_filepath)[0], pathsafe_guid(item.guid) + "_chunks")
    os.makedirs(chunks_path, exist_ok=True)

    # split into 300 second chunks with 20 second overlap
    CHUNK_DURATION = 300
    CHUNK_OVERLAP = 20
    timestamps = []
    tokens = []
    last_chunk_start = math.ceil(float(duration_probe_result.stdout.decode().strip()))
    for start_time in range(0, last_chunk_start, CHUNK_DURATION):
        chunk_wav_filename = os.path.join(chunks_path, f"chunk_{start_time}.wav")
        subprocess.call(f"ffmpeg -y -i {original_audio_filepath} -ss {start_time} -t {CHUNK_DURATION + CHUNK_OVERLAP} -acodec pcm_s16le -ac 1 -ar 16000 {chunk_wav_filename}", shell=True)
        asr_result = asr_model.recognize(chunk_wav_filename)
        chunk_data = {"timestamps": asr_result.timestamps, "tokens": asr_result.tokens}
        os.remove(chunk_wav_filename)
        shifted_timestamps, shifted_tokens = shift_timestamps(
            chunk_data["timestamps"],
            chunk_data["tokens"],
            CHUNK_DURATION,
            CHUNK_OVERLAP,
            start_time,
            start_time == 0,
            start_time == last_chunk_start
        )
        timestamps += shifted_timestamps
        tokens += shifted_tokens
    return tokens, timestamps

def make_segments(
    tokens: List[str],
    timestamps: List[float],
    seps = [".", "?", "!", "..."]
) -> Tuple[List[str], List[List[float]]]:
    PAUSE_THRESHOLD = 1
    segments = []
    segment_times = []
    current_segment = ""
    current_start = 0
    current_end = 0
    for i, token in enumerate(tokens):
        was_separator = False
        for sep in seps:
            sep_i = token.find(sep)
            if (sep_i != -1 and ((sep_i < len(token) - 1 and token[sep_i + 1] == " ")
                or (sep_i == len(token) - 1 and i + 1 < len(tokens) and tokens[i + 1][0] == " "))
            ):
                was_separator = True
                current_segment += token[:sep_i + 1]
                current_end = timestamps[i]
                segments.append(current_segment.strip())
                segment_times.append([current_start, current_end])
                current_segment = ""
                current_start = timestamps[i + 1]
                current_end = timestamps[i + 1]
                break
        if not was_separator:
            if (i + 1 < len(tokens) and timestamps[i + 1] - timestamps[i] > PAUSE_THRESHOLD):
                # TODO: awk repetition
                current_segment += token
                current_end = timestamps[i]
                segments.append(current_segment.strip())
                segment_times.append([current_start, current_end])
                current_segment = ""
                current_start = timestamps[i + 1]
                current_end = timestamps[i + 1]
            else:
                current_segment += token
                current_end = timestamps[i]
    if current_segment:
        segments.append(current_segment.strip())
        segment_times.append([current_start, current_end])
    return segments, segment_times


def gemini_generate(prompt: str) -> str:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=[genai.types.Content(
            role="user",
            parts=[genai.types.Part.from_text(text=prompt)],
        )],
        config=genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        ),
    )
    if response.text:
        return response.text
    raise Exception # TODO: better exceptions


def join_ranges(ad_ranges: List[List[int]]) -> List[List[int]]:
    joined_ad_ranges = []
    if not ad_ranges:
        return joined_ad_ranges
    range_start = ad_ranges[0][0]
    range_end = ad_ranges[0][1]
    for i in range(1, len(ad_ranges)):
        if ad_ranges[i][0] <= range_end + 1:
            range_end = ad_ranges[i][1]
        else:
            joined_ad_ranges.append([range_start, range_end])
            range_start, range_end = ad_ranges[i]
    joined_ad_ranges.append([range_start, range_end])
    return joined_ad_ranges


def build_trim_command(times: List[List[float]], ad_ranges: List[List[int]], source: str, dest: str) -> str:
    trim_ranges = ""
    for ad_range in ad_ranges:
        if trim_ranges:
            trim_ranges += "+"
        trim_ranges += "between(t," + str(int(times[ad_range[0]][0])) + "," + str(int(times[ad_range[1]][1])) + ")"
    return f"ffmpeg -i {source} -af \"aselect='not({trim_ranges})'\" {dest}"


def parse_model_response(model_response:str):
    list_regex = r'\[(?:.|\n)*\]'
    list_matches = re.findall(list_regex, model_response)
    if list_matches:
        return json.loads(list_matches[-1])
    return None


def get_noads_audio_filepath(feed_id: int, guid: str) -> str:
    audio_dir = os.path.join(os.path.dirname(__file__), "audio")
    os.makedirs(audio_dir, exist_ok=True)
    return os.path.join(audio_dir, str(feed_id), (f"{pathsafe_guid(guid)}_noads.mp3"))


def adblock_item(cursor: sqlite3.Cursor, item: Item):
    url = item.original_url
    audio_dir = os.getenv("AUDIO_DIR", "./audio")
    os.makedirs(audio_dir, exist_ok=True)
    original_audio_filepath = os.path.join(audio_dir, str(item.feed_id), f"{pathsafe_guid(item.guid)}_orig.mp3")
    download_item(url, original_audio_filepath)

    tokens, timestamps = asr(asr_model, item, original_audio_filepath)
    cursor.execute(
        "UPDATE items SET transcript_tokens=?, transcript_timestamps=? WHERE feed_id=? AND guid=?",
        (json.dumps(tokens), json.dumps(timestamps), item.feed_id, item.guid))
    segments, times = make_segments(tokens, timestamps)

    prompt = PROMPT
    for i, (segment, time) in enumerate(zip(segments, times)):
        prompt += f"{i}\n{segment}\n"
    model_response = gemini_generate(prompt)
    with open(
        os.path.join(
            os.path.split(original_audio_filepath)[0],
            pathsafe_guid(item.guid) + "_model_response.txt"
        ), "w"
    ) as f:
        f.write(f"prompt:\n{prompt}\n\nmodel_response:\n{model_response}")
    model_response_list = parse_model_response(model_response)
    # TODO: skip (write original path back to db) if model_response_list if Falsey
    if not model_response_list:
        logger.error(f"Failed to parse model response for {item.feed_id} {item.guid}")
        return

    noads_filepath = os.path.join(audio_dir, str(item.feed_id), (f"{pathsafe_guid(item.guid)}_noads.mp3"))
    trim_command = build_trim_command(
        times,
        model_response_list,
        original_audio_filepath,
        noads_filepath)
    logger.info(trim_command)
    trim_result = subprocess.call(trim_command, shell=True)
    if trim_result != 0:
        logger.error(f"Trimming failed for {item.feed_id} {item.guid}")
    else:
        cursor.execute(
            "UPDATE items SET audio_filepath = ? WHERE feed_id = ? AND guid = ?",
            (
                os.path.join(str(item.feed_id), (f"{pathsafe_guid(item.guid)}_noads.mp3")),
                item.feed_id,
                item.guid
            )
        )


if __name__ == "__main__":
    dotenv.load_dotenv()
    # asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    asr_model = onnx_asr.load_model(
        os.getenv("ASR_MODEL_NAME", "nemo-parakeet-tdt-0.6b-v2"),
        quantization="int8").with_timestamps()

    gemini_rpd = int(os.getenv("GEMINI_RPD", 500))
    gemini_rpm = int(os.getenv("GEMINI_RPM", 10))
    today = datetime.now().date()
    gemini_requests_today = 0
    gemini_last_request_time = datetime.fromtimestamp(0)
    update_feeds_interval_s = int(os.getenv("UPDATE_FEEDS_INTERVAL_S", 1800))
    last_update_feeds_time = datetime.fromtimestamp(0)

    def gemini_wait_time():
        global gemini_requests_today, today, gemini_rpd, gemini_rpm
        if gemini_requests_today >= gemini_rpd:
            # wait until tomorrow
            wait_time = datetime.combine(today + timedelta(days=1), dt_time()) - datetime.now()
            logger.info(f"Waiting {wait_time.total_seconds() + 10}s (Gemini RPD)")
            time.sleep(wait_time.total_seconds() + 10)
            gemini_requests_today = 0
            today = datetime.now().date()
        else:
            wait_time = max(
                0,
                ((60 / gemini_rpm) - (datetime.now() - gemini_last_request_time).total_seconds()))
            if wait_time > 0:
                logger.info(f"Waiting {wait_time}s (Gemini RPM)")
                time.sleep(wait_time)

    while True:
        with open("pods.toml", "rb") as f:
            pods_config = tomllib.load(f)

        with sqlite3.connect(os.getenv("DB_FILE", "castward.db"), timeout=10, autocommit=True) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            db.init_db(cursor)

            if (datetime.now() - last_update_feeds_time).total_seconds() > update_feeds_interval_s:
                for url in pods_config["feeds"]:
                    ingest_feed(cursor, url)
            unprocessed = cursor.execute(
                "SELECT * FROM items WHERE error IS NULL AND audio_filepath IS NULL ORDER BY pub_date DESC"
            ).fetchone()
            if not unprocessed:
                exit() # TODO: no
            item = Item(**unprocessed)
            gemini_last_request_time = datetime.now()
            gemini_requests_today += 1
            adblock_item(cursor, item)
