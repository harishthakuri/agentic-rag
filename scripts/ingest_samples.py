"""Upload sample_data/ into a collection and wait until every document is ready.

    RAG_API_KEY=arag_... uv run python scripts/ingest_samples.py [--collection samples]

Any folder works (subfolders included); the collection is created if needed:

    RAG_API_KEY=arag_... uv run python scripts/ingest_samples.py \
        --dir data/benchmark/docs --collection benchmark

Requires the API (`make run`) and the worker (`make worker`) to be running.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"
SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}
SKIP = {"ATTRIBUTION.md", "README.md"}  # credits and notes, not documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--collection", default="samples")
    parser.add_argument("--dir", type=Path, default=SAMPLE_DIR, help="folder to upload")
    parser.add_argument("--description", default="Sample documents")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=300, help="seconds to wait for ingestion")
    args = parser.parse_args()

    api_key = os.environ.get("RAG_API_KEY")
    if not api_key:
        sys.exit("Set RAG_API_KEY (create one with: make api-key name=dev)")

    with httpx.Client(
        base_url=f"{args.base_url}/api/v1",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    ) as client:
        collection_id = _ensure_collection(client, args.collection, args.description)
        files = sorted(
            p for p in args.dir.rglob("*") if p.suffix in SUPPORTED and p.name not in SKIP
        )
        if not files:
            sys.exit(f"No documents found in {args.dir}")
        for path in files:
            response = client.post(
                f"/collections/{collection_id}/documents",
                files={"file": (path.name, path.read_bytes())},
            )
            if response.status_code == 409:
                print(f"  = {path.name} (already uploaded)")
            else:
                response.raise_for_status()
                print(f"  + {path.name} (queued)")
        _wait_until_ready(client, collection_id, args.timeout)


def _ensure_collection(client: httpx.Client, name: str, description: str) -> str:
    response = client.post("/collections", json={"name": name, "description": description})
    if response.status_code == 409:
        for collection in client.get("/collections", params={"limit": 100}).json()["items"]:
            if collection["name"] == name:
                print(f"Using existing collection '{name}'")
                return str(collection["id"])
    response.raise_for_status()
    print(f"Created collection '{name}'")
    return str(response.json()["id"])


def _wait_until_ready(client: httpx.Client, collection_id: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        documents = client.get(
            f"/collections/{collection_id}/documents", params={"limit": 100}
        ).json()["items"]
        pending = [d for d in documents if d["status"] in ("pending", "processing")]
        if not pending:
            break
        if time.monotonic() > deadline:
            sys.exit(f"Timed out; still processing: {[d['source_filename'] for d in pending]}")
        print(f"  … waiting for {len(pending)} document(s) (is `make worker` running?)")
        time.sleep(3)

    print()
    for d in documents:
        detail = f"{d['chunk_count']} chunks" if d["status"] == "ready" else d["error"]
        print(f"  {d['status']:8} {d['title']:40} {detail}")


if __name__ == "__main__":
    main()
