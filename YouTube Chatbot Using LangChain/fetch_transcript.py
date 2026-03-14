import json # to parse subtitle data returned by YouTube
import yt_dlp # to extract captions if transcript fetch fails
import requests # to download caption file from YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

def extract_video_id(url: str):
    try:
        video_id = url[17:17+11]
        return video_id
    except:
        return None

def fetch_transcript(url: str):
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    try:
        transcript_fetcher = YouTubeTranscriptApi()
        transcript = transcript_fetcher.fetch(video_id, languages=['en','hi','bn'])
        text = " ".join([t.text for t in transcript])
        return text
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video")
    except Exception:
        try:
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "quiet": True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            subtitles = info.get("subtitles") or info.get("automatic_captions")
            if not subtitles:
                raise RuntimeError("No captions available")
            preferred = ['en', 'hi', 'bn']
            lang = None
            for p in preferred:
                if p in subtitles:
                    lang = p
                    break

            if not lang:
                lang = list(subtitles.keys())[0]
            caption_url = subtitles[lang][0]["url"]
            r = requests.get(caption_url)
            data = json.loads(r.text)
            text = " ".join(
                seg["utf8"]
                for event in data.get("events", [])
                for seg in event.get("segs", [])
                if "utf8" in seg
            )
            return text
        except Exception as e:
            raise RuntimeError(str(e))