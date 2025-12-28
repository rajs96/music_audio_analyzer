import os
import time
import difflib
import requests
import pandas as pd
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.spotify_api.youtube_search import search_youtube, download_audio

SPOTIFY_AUTH_TOKEN = os.environ.get("SPOTIFY_AUTH_TOKEN", "")
if not SPOTIFY_AUTH_TOKEN:
    raise ValueError("SPOTIFY_AUTH_TOKEN is not set")

AUDIO_FILES_DIR = Path(__file__).parent.parent.parent / "eval_audio_files"

# 20 diverse genres covering different instrument profiles
TARGET_GENRES = [
    "rock",
    "pop",
    "hip-hop",
    "jazz",
    "classical",
    "electronic",
    "r-n-b",
    "country",
    "metal",
    "folk",
    "blues",
    "reggae",
    "latin",
    "indie",
    "soul",
    "funk",
    "punk",
    "ambient",
    "acoustic",
    "piano",
]

# Known Spotify genre seeds (as of 2024)
# Source: https://gist.github.com/drumnation/91a789da6f17f2ee20db8f55382b6653
SPOTIFY_GENRE_SEEDS = [
    "acoustic",
    "afrobeat",
    "alt-rock",
    "alternative",
    "ambient",
    "anime",
    "black-metal",
    "bluegrass",
    "blues",
    "bossanova",
    "brazil",
    "breakbeat",
    "british",
    "cantopop",
    "chicago-house",
    "children",
    "chill",
    "classical",
    "club",
    "comedy",
    "country",
    "dance",
    "dancehall",
    "death-metal",
    "deep-house",
    "detroit-techno",
    "disco",
    "disney",
    "drum-and-bass",
    "dub",
    "dubstep",
    "edm",
    "electro",
    "electronic",
    "emo",
    "folk",
    "forro",
    "french",
    "funk",
    "garage",
    "german",
    "gospel",
    "goth",
    "grindcore",
    "groove",
    "grunge",
    "guitar",
    "happy",
    "hard-rock",
    "hardcore",
    "hardstyle",
    "heavy-metal",
    "hip-hop",
    "holidays",
    "honky-tonk",
    "house",
    "idm",
    "indian",
    "indie",
    "indie-pop",
    "industrial",
    "iranian",
    "j-dance",
    "j-idol",
    "j-pop",
    "j-rock",
    "jazz",
    "k-pop",
    "kids",
    "latin",
    "latino",
    "malay",
    "mandopop",
    "metal",
    "metal-misc",
    "metalcore",
    "minimal-techno",
    "movies",
    "mpb",
    "new-age",
    "new-release",
    "opera",
    "pagode",
    "party",
    "philippines-opm",
    "piano",
    "pop",
    "pop-film",
    "post-dubstep",
    "power-pop",
    "progressive-house",
    "psych-rock",
    "punk",
    "punk-rock",
    "r-n-b",
    "rainy-day",
    "reggae",
    "reggaeton",
    "road-trip",
    "rock",
    "rock-n-roll",
    "rockabilly",
    "romance",
    "sad",
    "salsa",
    "samba",
    "sertanejo",
    "show-tunes",
    "singer-songwriter",
    "ska",
    "sleep",
    "songwriter",
    "soul",
    "soundtracks",
    "spanish",
    "study",
    "summer",
    "swedish",
    "synth-pop",
    "tango",
    "techno",
    "trance",
    "trip-hop",
    "turkish",
    "work-out",
    "world-music",
]

SONGS_PER_GENRE = 50


def get_headers():
    return {"Authorization": f"Bearer {SPOTIFY_AUTH_TOKEN}"}


def fuzzy_match_genre(
    target: str, available: list[str], threshold: float = 0.6
) -> str | None:
    """
    Find the best matching genre from available list using fuzzy matching.

    Args:
        target: The genre we want to match
        available: List of available Spotify genre seeds
        threshold: Minimum similarity ratio (0-1) to consider a match

    Returns:
        Best matching genre or None if no good match found
    """
    # First try exact match
    if target in available:
        return target

    # Normalize for comparison (lowercase, remove special chars)
    target_normalized = target.lower().replace(" ", "-").replace("&", "n")

    # Try normalized exact match
    for genre in available:
        if genre.lower() == target_normalized:
            return genre

    # Common aliases mapping
    aliases = {
        "rnb": "r-n-b",
        "r&b": "r-n-b",
        "hiphop": "hip-hop",
        "hip hop": "hip-hop",
        "edm": "electronic",
        "dnb": "drum-and-bass",
        "drum n bass": "drum-and-bass",
    }

    if target.lower() in aliases:
        alias_match = aliases[target.lower()]
        if alias_match in available:
            return alias_match

    # Fuzzy match using difflib
    matches = difflib.get_close_matches(
        target_normalized, available, n=1, cutoff=threshold
    )

    if matches:
        return matches[0]

    return None


def match_genres(
    target_genres: list[str], available_genres: list[str]
) -> dict[str, str | None]:
    """
    Match target genres to available Spotify genres using fuzzy matching.

    Returns:
        Dict mapping target genre -> matched Spotify genre (or None)
    """
    matches = {}

    for target in target_genres:
        match = fuzzy_match_genre(target, available_genres)
        matches[target] = match

        if match and match != target:
            logger.info(f"Fuzzy matched '{target}' -> '{match}'")
        elif not match:
            logger.warning(f"No match found for genre: '{target}'")

    return matches


def get_available_genres() -> list[str]:
    """Fetch available genre seeds from Spotify API, fall back to cached list."""
    url = "https://api.spotify.com/v1/recommendations/available-genre-seeds"

    try:
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        genres = response.json().get("genres", [])
        if genres:
            logger.info(f"Fetched {len(genres)} genres from Spotify API")
            return genres
    except Exception as e:
        logger.warning(f"Failed to fetch genres from API: {e}")

    logger.info(f"Using cached genre list ({len(SPOTIFY_GENRE_SEEDS)} genres)")
    return SPOTIFY_GENRE_SEEDS


def search_tracks_by_genre(genre: str, limit: int = 50) -> list[dict]:
    """Search for popular tracks in a genre."""
    tracks = []
    url = "https://api.spotify.com/v1/search"

    params = {
        "q": f"genre:{genre}",
        "type": "track",
        "limit": min(limit, 50),
        "market": "US",
    }

    response = requests.get(url, headers=get_headers(), params=params)
    response.raise_for_status()
    data = response.json()

    for item in data.get("tracks", {}).get("items", []):
        tracks.append(
            {
                "track_id": item["id"],
                "track_name": item["name"],
                "artists": ", ".join(a["name"] for a in item["artists"]),
                "album": item["album"]["name"],
                "popularity": item["popularity"],
                "duration_ms": item["duration_ms"],
                "preview_url": item.get("preview_url"),
                "genre": genre,
            }
        )

    return tracks


def get_recommendations_by_genre(genre: str, limit: int = 50) -> list[dict]:
    """Get track recommendations for a genre seed."""
    url = "https://api.spotify.com/v1/recommendations"

    params = {
        "seed_genres": genre,
        "limit": min(limit, 100),
        "market": "US",
    }

    response = requests.get(url, headers=get_headers(), params=params)
    response.raise_for_status()
    data = response.json()

    tracks = []
    for item in data.get("tracks", []):
        tracks.append(
            {
                "track_id": item["id"],
                "track_name": item["name"],
                "artists": ", ".join(a["name"] for a in item["artists"]),
                "album": item["album"]["name"],
                "popularity": item["popularity"],
                "duration_ms": item["duration_ms"],
                "preview_url": item.get("preview_url"),
                "genre": genre,
            }
        )

    return tracks


def sanitize_filename(name: str) -> str:
    """Remove or replace characters that are invalid in filenames."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name


def download_track(track: dict, output_dir: Path) -> dict:
    """Download a single track via YouTube. Returns result dict."""
    search_query = f"{track['track_name']} {track['artists']}"
    filename = sanitize_filename(f"{track['artists']} - {track['track_name']}")

    try:
        results = search_youtube(search_query, max_results=1)
        if not results:
            return {"status": "skipped", "reason": "no_youtube_results", **track}

        video_url = results[0]["url"]
        download_audio(video_url, str(output_dir), filename)

        return {
            "status": "downloaded",
            "filepath": str(output_dir / f"{filename}.mp3"),
            "youtube_url": video_url,
            **track,
        }

    except Exception as e:
        logger.error(f"Error downloading {search_query}: {e}")
        return {"status": "error", "error": str(e), **track}


def build_dataset(download: bool = False) -> pd.DataFrame:
    """Build the eval dataset by fetching tracks for each genre."""
    logger.info("Fetching available genres...")
    available_genres = get_available_genres()

    # Match target genres to Spotify genres using fuzzy matching
    logger.info("Matching target genres to Spotify genre seeds...")
    genre_mapping = match_genres(TARGET_GENRES, available_genres)

    valid_genres = {k: v for k, v in genre_mapping.items() if v is not None}
    invalid_genres = [k for k, v in genre_mapping.items() if v is None]

    if invalid_genres:
        logger.warning(f"Could not match genres: {invalid_genres}")

    logger.info(f"Successfully matched {len(valid_genres)}/{len(TARGET_GENRES)} genres")

    all_tracks = []
    seen_track_ids = set()

    for target_genre, spotify_genre in valid_genres.items():
        logger.info(f"Fetching tracks for: {target_genre} (Spotify: {spotify_genre})")

        # Try search first, fall back to recommendations
        tracks = search_tracks_by_genre(spotify_genre, limit=SONGS_PER_GENRE)

        if len(tracks) < SONGS_PER_GENRE:
            logger.info(
                f"Search returned {len(tracks)}, supplementing with recommendations"
            )
            rec_tracks = get_recommendations_by_genre(
                spotify_genre, limit=SONGS_PER_GENRE - len(tracks)
            )
            tracks.extend(rec_tracks)

        # Add original target genre for reference
        for track in tracks:
            track["target_genre"] = target_genre

        # Deduplicate
        unique_tracks = []
        for track in tracks:
            if track["track_id"] not in seen_track_ids:
                seen_track_ids.add(track["track_id"])
                unique_tracks.append(track)

        all_tracks.extend(unique_tracks[:SONGS_PER_GENRE])
        logger.info(f"  -> Got {len(unique_tracks[:SONGS_PER_GENRE])} unique tracks")

        # Rate limiting
        time.sleep(0.1)

    logger.info(f"Total tracks collected: {len(all_tracks)}")

    # Create DataFrame
    df = pd.DataFrame(all_tracks)

    # Save metadata
    output_dir = AUDIO_FILES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "eval_dataset_metadata.csv"
    df.to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata to {metadata_path}")

    # Also save as JSON for easier inspection
    json_path = output_dir / "eval_dataset_metadata.json"
    df.to_json(json_path, orient="records", indent=2)
    logger.info(f"Saved JSON to {json_path}")

    # Download audio if requested
    if download:
        logger.info(f"Downloading {len(all_tracks)} tracks...")
        num_workers = os.cpu_count()

        downloaded = 0
        errors = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(download_track, track, output_dir): track
                for track in all_tracks
            }

            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "downloaded":
                    downloaded += 1
                else:
                    errors += 1

                if (downloaded + errors) % 10 == 0:
                    logger.info(f"Progress: {downloaded + errors}/{len(all_tracks)}")

        logger.info(f"Download complete: {downloaded} succeeded, {errors} failed")

    return df


def print_genre_summary(df: pd.DataFrame):
    """Print summary statistics by genre."""
    logger.info("\n=== Dataset Summary ===")
    summary = (
        df.groupby("genre")
        .agg(
            count=("track_id", "count"),
            avg_popularity=("popularity", "mean"),
            avg_duration_min=("duration_ms", lambda x: (x.mean() / 60000)),
        )
        .round(2)
    )

    print(summary.to_string())
    print(f"\nTotal tracks: {len(df)}")
    print(f"Unique genres: {df['genre'].nunique()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build eval dataset from Spotify genres"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download audio files via YouTube",
    )
    parser.add_argument(
        "--songs-per-genre",
        type=int,
        default=50,
        help="Songs per genre (default: 50)",
    )
    args = parser.parse_args()

    global SONGS_PER_GENRE
    SONGS_PER_GENRE = args.songs_per_genre

    df = build_dataset(download=args.download)
    print_genre_summary(df)


if __name__ == "__main__":
    main()
