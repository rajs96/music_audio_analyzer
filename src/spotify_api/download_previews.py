import os
import requests

AUTH_TOKEN = os.getenv("SPOTIFY_AUTH_TOKEN")  # Replace with your actual token

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


def download_preview(preview_url, filename):
    """Download a preview MP3 file."""
    response = requests.get(preview_url)
    response.raise_for_status()

    filepath = os.path.join(AUDIO_FILES_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(response.content)

    return filepath


def sanitize_filename(name):
    """Remove or replace characters that are invalid in filenames."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name


def main():
    os.makedirs(AUDIO_FILES_DIR, exist_ok=True)

    print("Fetching playlists...")
    playlists = get_current_user_playlists()

    playlist = find_playlist_by_name(playlists, "2023")
    if not playlist:
        print("Playlist '2023' not found!")
        return

    print(f"Found playlist: {playlist['name']} ({playlist['id']})")

    print("Fetching tracks...")
    tracks = get_playlist_tracks(playlist["id"])
    print(f"Found {len(tracks)} tracks")

    downloaded = 0
    skipped = 0
    unknown = 0

    for item in tracks:
        track = item.get("track")
        if not track:
            continue

        track_name = track.get("name")
        if not track_name:
            print(f"  Skipping (no name): {track}")
            unknown += 1
            continue

        artists = ", ".join(artist["name"] for artist in track.get("artists", []))
        preview_url = track.get("preview_url")

        if not preview_url:
            print(f"  Skipping (no preview): {track_name} - {artists}")
            skipped += 1
            continue

        filename = sanitize_filename(f"{artists} - {track_name}.mp3")
        print(f"  Downloading: {filename}")

        try:
            download_preview(preview_url, filename)
            downloaded += 1
        except Exception as e:
            print(f"    Error downloading: {e}")
            skipped += 1

    print(f"\nDone! Downloaded {downloaded} previews, skipped {skipped}")


if __name__ == "__main__":
    main()
