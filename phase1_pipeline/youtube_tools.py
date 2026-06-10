from __future__ import annotations

import os
import time
import random
from googleapiclient.discovery import build
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    IpBlocked,
    RequestBlocked,
)
from dotenv import load_dotenv

# Per-run cap so one execution never hammers YouTube into an IP ban.
MAX_PER_RUN = int(os.getenv("MAX_TRANSCRIPTS_PER_RUN", "30"))

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def search_channel_playlists(channel_query: str = "Campus X", max_results: int = 20) -> list[dict]:
    """Search for Campus X channel and get its playlists."""
    search_response = youtube.search().list(
        q=channel_query,
        type="channel",
        part="snippet",
        maxResults=5
    ).execute()

    if not search_response["items"]:
        print("No channel found.")
        return []

    channel_id = search_response["items"][0]["snippet"]["channelId"]
    print(f"Found channel ID: {channel_id}")

    playlists_response = youtube.playlists().list(
        channelId=channel_id,
        part="snippet,contentDetails",
        maxResults=max_results
    ).execute()

    playlists = []
    for item in playlists_response.get("items", []):
        title = item["snippet"]["title"].lower()
        keywords = ["machine learning", "deep learning", "neural", "nlp", "ai", "artificial", "data science", "python", "transformer", "llm"]
        if any(kw in title for kw in keywords):
            playlists.append({
                "playlist_id": item["id"],
                "title": item["snippet"]["title"],
                "video_count": item["contentDetails"]["itemCount"]
            })

    print(f"Found {len(playlists)} AI/ML related playlists.")
    return playlists


def get_videos_from_playlist(playlist_id: str) -> list[dict]:
    """Get all video IDs and titles from a playlist."""
    videos = []
    next_page_token = None

    while True:
        response = youtube.playlistItems().list(
            playlistId=playlist_id,
            part="snippet",
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        for item in response.get("items", []):
            videos.append({
                "video_id": item["snippet"]["resourceId"]["videoId"],
                "title": item["snippet"]["title"],
                "playlist_id": playlist_id
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return videos


def _pick_transcript(transcript_list):
    """Pick best transcript: native English > translatable-to-English > anything."""
    items = list(transcript_list)
    for t in items:
        if t.language_code in ("en", "en-IN"):
            return t, "en"
    for t in items:
        if t.is_translatable:
            return t.translate("en"), "en"
    if items:
        return items[0], items[0].language_code
    return None, None


def fetch_transcript(video_id: str, title: str) -> dict | None:
    """Fetch one transcript with a single list() + single fetch(), backoff on IP blocks."""
    api = YouTubeTranscriptApi()

    for attempt in range(5):
        try:
            transcript_list = api.list(video_id)
            transcript, lang = _pick_transcript(transcript_list)
            if transcript is None:
                return None
            fetched = transcript.fetch()
            full_text = " ".join(snippet.text for snippet in fetched)
            return {
                "video_id": video_id,
                "title": title,
                "transcript": full_text,
                "language": lang,
                "url": f"https://www.youtube.com/watch?v={video_id}"
            }
        except (TranscriptsDisabled, NoTranscriptFound):
            return None
        except (IpBlocked, RequestBlocked):
            wait = 15 * (2 ** attempt) + random.uniform(0, 5)
            print(f"  IP-blocked. Backing off {wait:.0f}s (attempt {attempt+1}/5)...")
            time.sleep(wait)
        except Exception as e:
            print(f"  Error for {title}: {e}")
            return None

    print(f"  Gave up (IP-blocked): {title}")
    return None


def fetch_all_transcripts(videos: list[dict], skip_ids: set | None = None) -> list[dict]:
    """Fetch transcripts with rate-limit-safe pacing.

    skip_ids: video_ids already stored — skipped so reruns resume instead of restart.
    Processes at most MAX_PER_RUN new videos per call.
    """
    skip_ids = skip_ids or set()
    transcripts = []
    failed = []

    todo = [v for v in videos if v["video_id"] not in skip_ids]
    skipped = len(videos) - len(todo)
    if skipped:
        print(f"Skipping {skipped} already-stored videos.")

    batch = todo[:MAX_PER_RUN]
    print(f"Processing {len(batch)} this run (of {len(todo)} remaining).")

    consecutive_fail = 0
    for i, video in enumerate(batch):
        print(f"Fetching transcript {i+1}/{len(batch)}: {video['title']}")
        result = fetch_transcript(video["video_id"], video["title"])

        if result:
            transcripts.append(result)
            consecutive_fail = 0
            print(f"  ✓ [{result['language']}] {len(result['transcript'].split())} words")
        else:
            failed.append(video)
            consecutive_fail += 1

        # Circuit breaker: 5 failures in a row likely = sustained IP block. Bail
        # and keep what we have so reruns resume cleanly.
        if consecutive_fail >= 5:
            print("\n--- 5 consecutive failures; stopping run to avoid hard IP ban. ---")
            break

        # Random delay between requests to avoid tripping IP rate limits.
        time.sleep(random.uniform(4, 8))

        # Longer break every 25 videos.
        if (i + 1) % 25 == 0:
            print("\n--- 30s break to avoid rate limiting ---\n")
            time.sleep(30)

    print(f"\nFetched: {len(transcripts)}  Failed: {len(failed)}")
    return transcripts
