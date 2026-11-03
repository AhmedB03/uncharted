"""Small GitHub REST client with rate limit handling."""

import base64
import os
import time

import requests

API = "https://api.github.com"


class GitHub:
    def __init__(self, token=None):
        self.session = requests.Session()
        token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "uncharted",
        })
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        else:
            print("Warning: no GITHUB_TOKEN set, you will hit the 60 req/hour limit fast.")

    def get(self, path, params=None, tries=6):
        url = path if path.startswith("http") else f"{API}{path}"
        for attempt in range(tries):
            r = self.session.get(url, params=params, timeout=30)
            if r.status_code in (403, 429) and (
                r.headers.get("X-RateLimit-Remaining") == "0" or "rate limit" in r.text.lower()
            ):
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(5, reset - time.time() + 2)
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    wait = int(retry_after) + 1
                print(f"  rate limited, sleeping {int(wait)}s")
                time.sleep(min(wait, 900))
                continue
            if r.status_code >= 500 or r.status_code == 202:
                # 202 means GitHub is computing stats (contributors); retry shortly
                time.sleep(2 ** attempt)
                continue
            return r
        return r

    def json(self, path, params=None):
        r = self.get(path, params)
        if r.status_code != 200:
            return None
        return r.json()

    def raw(self, owner, name, branch, path, max_chars):
        """Fetch a file body from raw.githubusercontent.com (no REST quota, no auth sent)."""
        try:
            r = requests.get(f"https://raw.githubusercontent.com/{owner}/{name}/{branch}/{path}",
                             timeout=20, headers={"User-Agent": "uncharted"})
        except requests.RequestException:
            return None
        return r.text[:max_chars] if r.status_code == 200 else None

    def file_text(self, owner, name, path, max_chars):
        return self.decoded(self.json(f"/repos/{owner}/{name}/contents/{path}"), max_chars)

    @staticmethod
    def decoded(data, max_chars):
        if not data or isinstance(data, list) or data.get("encoding") != "base64":
            return None
        try:
            text = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except Exception:
            return None
        return text[:max_chars]
