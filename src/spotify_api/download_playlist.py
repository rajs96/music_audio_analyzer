import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from src.spotify_api.youtube_search import search_youtube, download_audio

AUTH_TOKEN = os.environ.get("SPOTIFY_AUTH_TOKEN", "")  # Replace with your actual token
if not AUTH_TOKEN:
    raise ValueError("SPOTIFY_AUTH_TOKEN is not set")

AUDIO_FILES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "audio_files")


def get_headers():
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def get_current_user_playlists():
    """Fetch all playlists for the current user."""
    playlists = []
    url = "https://api.spotify.com/v1/me/playlists"

    while url:
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        data = response.json()
        playlists.extend(data["items"])
        url = data.get("next")

    return playlists


def find_playlist_by_name(playlists, name):
    """Find a playlist by its name."""
    for playlist in playlists:
        if playlist["name"] == name:
            return playlist
    return None


def get_playlist_tracks(playlist_id):
    """Fetch all tracks from a playlist."""
    tracks = []
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"

    while url:
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        data = response.json()
        tracks.extend(data["items"])
        url = data.get("next")

    return tracks


def sanitize_filename(name):
    """Remove or replace characters that are invalid in filenames."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name


def download_track(track_info: dict, debug: bool = False) -> dict:
    """Download a single track. Returns result dict with status."""
    track = track_info.get("track")
    if not track:
        return {"status": "skipped", "reason": "no_track"}

    track_name = track.get("name")
    if not track_name:
        return {"status": "unknown", "reason": "no_name"}

    artists = ", ".join(artist["name"] for artist in track.get("artists", []))
    search_query = f"{track_name} {artists}"
    filename = sanitize_filename(f"{artists} - {track_name}")

    logger.info(f"Searching YouTube for: {search_query}")

    try:
        results = search_youtube(search_query, max_results=1)
        if not results:
            logger.warning(f"No YouTube results for: {search_query}")
            return {"status": "skipped", "reason": "no_results", "query": search_query}

        video_url = results[0]["url"]
        if debug:
            logger.info(f"Video URL: {video_url}")
            return {"status": "debug", "url": video_url, "query": search_query}

        logger.info(f"Downloading: {results[0]['title']}")
        download_audio(video_url, AUDIO_FILES_DIR, filename)
        return {
            "status": "downloaded",
            "title": results[0]["title"],
            "query": search_query,
        }

    except Exception as e:
        logger.error(f"Error downloading {search_query}: {e}")
        return {"status": "error", "error": str(e), "query": search_query}


def main(debug: bool = False, first_n: int = None):
    os.makedirs(AUDIO_FILES_DIR, exist_ok=True)

    num_workers = os.cpu_count() or 4
    logger.info(f"Using {num_workers} parallel workers (CPU count)")

    logger.info("Fetching playlists...")
    playlists = get_current_user_playlists()

    playlist = find_playlist_by_name(playlists, "2023")
    if not playlist:
        logger.error("Playlist '2023' not found!")
        return

    logger.info(f"Found playlist: {playlist['name']} ({playlist['id']})")

    logger.info("Fetching tracks...")
    tracks = get_playlist_tracks(playlist["id"])
    logger.info(f"Found {len(tracks)} tracks")

    if first_n:
        tracks = tracks[:first_n]
        logger.info(f"Downloading first {first_n} tracks")
    else:
        logger.info("Downloading all tracks")

    downloaded = 0
    skipped = 0
    unknown = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_track, track, debug): track for track in tracks
        }

        for future in as_completed(futures):
            result = future.result()
            status = result.get("status")

            if status == "downloaded" or status == "debug":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            elif status == "unknown":
                unknown += 1
            elif status == "error":
                errors += 1

    logger.info(
        f"\nDone! Downloaded {downloaded}, skipped {skipped}, errors {errors}, unknown {unknown}"
    )


if __name__ == "__main__":
    main(debug=False, first_n=None)
