import os
import requests
import subprocess
from pathlib import Path
from loguru import logger

AUDIO_FILES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "audio_files")


def download_audio(
    url: str, output_dir: str = AUDIO_FILES_DIR, filename: str = None
) -> str:
    """
    Download audio from a YouTube video using yt-dlp.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the audio file
        filename: Optional filename (without extension). If None, uses video title.

    Returns:
        Path to the downloaded audio file
    """
    os.makedirs(output_dir, exist_ok=True)

    output_template = os.path.join(output_dir, filename if filename else "%(title)s")
    if Path(output_template).exists():
        logger.info(f"File already exists: {output_template}")
        return output_template

    cmd = [
        "yt-dlp",
        "-x",  # extract audio
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",  # best quality
        "-o",
        f"{output_template}.%(ext)s",
        "--no-playlist",
        "--js-runtimes",
        "deno,node",  # use deno or node for JS extraction
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    # Find the downloaded file
    if filename:
        return os.path.join(output_dir, f"{filename}.mp3")

    # Parse output to find actual filename
    for line in result.stdout.split("\n"):
        if "Destination:" in line and ".mp3" in line:
            return line.split("Destination:")[-1].strip()

    # Fallback: return the directory
    return output_dir


def search_and_download(query: str, output_dir: str = AUDIO_FILES_DIR) -> dict:
    """
    Search YouTube and download the first result.

    Args:
        query: Song name to search for
        output_dir: Directory to save the audio file

    Returns:
        Dict with video info and downloaded file path
    """
    results = search_youtube(query, max_results=1)
    if not results:
        raise ValueError(f"No results found for: {query}")

    result = results[0]
    filepath = download_audio(result["url"], output_dir)
    result["filepath"] = filepath

    return result


def search_youtube(query: str, max_results: int = 3) -> list[dict]:
    """
    Search YouTube for a song and return the first results.

    Args:
        query: The song name to search for
        max_results: Number of results to return (default 3)

    Returns:
        List of dicts with title, video_id, url, channel, and duration
    """
    # Use YouTube's internal search endpoint (no API key needed)
    url = "https://www.youtube.com/results"
    params = {"search_query": query}
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    # Extract video data from the page's initial data
    import re
    import json

    # Find the ytInitialData JSON in the page
    match = re.search(r"var ytInitialData = ({.*?});", response.text)
    if not match:
        return []

    data = json.loads(match.group(1))

    results = []
    try:
        contents = data["contents"]["twoColumnSearchResultsRenderer"][
            "primaryContents"
        ]["sectionListRenderer"]["contents"][0]["itemSectionRenderer"]["contents"]

        for item in contents:
            if "videoRenderer" not in item:
                continue

            video = item["videoRenderer"]
            video_id = video.get("videoId")
            if not video_id:
                continue

            results.append(
                {
                    "title": video.get("title", {})
                    .get("runs", [{}])[0]
                    .get("text", ""),
                    "video_id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "channel": video.get("ownerText", {})
                    .get("runs", [{}])[0]
                    .get("text", ""),
                    "duration": video.get("lengthText", {}).get("simpleText", ""),
                }
            )

            if len(results) >= max_results:
                break

    except (KeyError, IndexError):
        pass

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python youtube_search.py <song name>")
        print("       python youtube_search.py --download <song name>")
        sys.exit(1)

    download_mode = sys.argv[1] == "--download"
    song_name = " ".join(sys.argv[2:] if download_mode else sys.argv[1:])

    print(f"Searching YouTube for: {song_name}\n")

    results = search_youtube(song_name)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   Channel: {result['channel']}")
        print(f"   Duration: {result['duration']}")
        print(f"   URL: {result['url']}")
        print()

    if download_mode and results:
        print(f"Downloading: {results[0]['title']}...")
        filepath = download_audio(results[0]["url"])
        print(f"Saved to: {filepath}")
