# Castward

Castward is adblock for podcasts.

It has two components:
* `cruncher.py`
    * downloads podcasts from the specified RSS feeds
    * transcribes them using Nvidia's Parakeet ASR model
    * asks Google Gemini to identify advertisements
        * (rate limited to fit within free tier)
    * removes them from the original audio files with ffmpeg
* `serve.py`
    * mirrors the original RSS feeds, with the edited audio files substituted

Support your favorite podcasts.  Consider subscribing to their paid feed, donating to their creators, recommending them to others, etc.  This tool serves each feed under a random GUID to intentionally prevent easy discovery.  In order to preserve download counts of the original feed, please consider _not_ sharing a Castward server/feed URL with others.

### Installation

Note: This software is experimental and currently only tested on Linux.

Clone this repository:
```
git clone https://github.com/jwlarocque/castward.git
cd castward
```

Install `uv` (optional; remainder of this document will assume you're using it): https://docs.astral.sh/uv/getting-started/installation/

Create a virtual environment, then follow the on-screen instructions to activate it:
```
uv venv --python 3.13
source .venv/bin/activate
```

Install the ONNX Runtime for Python: https://onnxruntime.ai/getting-started  
It is highly recommended to use a hardware-accelerated runtime if your system supports it.  However, if you're not sure which version to install, you can use the default:
```
uv pip install onnxruntime
```

Install other dependencies:
```
uv pip install -r requirements.txt
```

If it is not already installed (try `ffmpeg -version`), install `ffmpeg`: https://www.ffmpeg.org/download.html

### Configuration

You *must* provide a Gemini API key via the `GEMINI_API_KEY` environment variable.  You can obtain one here: https://aistudio.google.com/app/apikey  
This and other configuration options can be set via a `.env` file (see `.env.example` for defaults).

Provide a list of RSS feeds via `pods.toml`, like so:

```
[
    "https://example.com/feed1.rss", # comments are allowed
    "https://example.com/thisisapodcastfeed"
]
```

Castward will ingest the most recent episodes across all feeds first.  If you want to prioritize a certain feed, you must comment out or remove all other feeds from the `pods.toml` file, then add them back after all episodes have been processed.  (Don't worry, this will not affect any already processed episodes.)

### Non-RSS Podcast Apps

Some podcast apps do not support directly reading from an RSS feed, but they usually have some mechanism to make private feeds available.  For example, for Pocket Casts you can follow these instructions: https://support.pocketcasts.com/knowledge-base/private-or-members-only-feeds/  (note that this requires a publicly accessible Castward feed).  
If your podcast app does not provide such functionality, you will not be able to use Castward with it.
