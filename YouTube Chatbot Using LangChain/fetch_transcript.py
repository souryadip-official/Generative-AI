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
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video")
    except Exception as e:
        raise RuntimeError(str(e))
    text = " ".join([t.text for t in transcript])
    return text