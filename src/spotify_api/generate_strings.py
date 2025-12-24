"""
Generate code verified and code challenge strings for Spotify OAuth2.
"""

import os, base64, hashlib

if __name__ == "__main__":
    # 1) code_verifier: 43-128 chars. We'll use 32 random bytes -> ~43 chars after base64url
    code_verifier = base64.urlsafe_b64encode(os.urandom(32)).decode("utf-8").rstrip("=")

    # 2) code_challenge = base64url(SHA256(code_verifier))
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")

    print("code_verifier:", code_verifier)
    print("code_challenge:", code_challenge)
