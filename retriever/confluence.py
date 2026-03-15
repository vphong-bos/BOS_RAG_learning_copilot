#!/usr/bin/env python3
import base64
import html
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from weasyprint import HTML

load_dotenv()

CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL", "").rstrip("/")
CONFLUENCE_EMAIL = os.getenv("CONFLUENCE_EMAIL", "")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN", "")
ROOT_ID = os.getenv("ROOT_ID", "").strip()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "retrieved_docs"))
POLLING_INTERVAL = int(os.getenv("POLLING_INTERVAL_IN_SECONDS", "2"))

# Add page/folder IDs here to skip completely.
# If a folder ID is here, that whole subtree will not be traversed.
SKIP_IDS = {
    "116621984",
    "158960566",
}


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180] or "untitled"


def build_auth_header(email: str, api_token: str) -> str:
    raw = f"{email}:{api_token}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def http_json(url: str, method: str = "GET", headers=None, body=None) -> dict:
    headers = headers or {}
    data = None

    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers = {**headers, "Content-Type": "application/json"}

    req = Request(url, data=data, headers=headers, method=method)

    try:
        with urlopen(req) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code} calling {url}\n{detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error calling {url}: {exc}") from exc


class ConfluenceClient:
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": build_auth_header(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN),
            "Accept": "application/json",
        }

    def get_root_folder_info(self, folder_id: str) -> dict:
        url = f"{CONFLUENCE_BASE_URL}/wiki/api/v2/folders/{folder_id}"
        return http_json(url, headers=self.headers())

    def _fetch_paginated_children(self, url: str) -> List[dict]:
        items: List[dict] = []

        while url:
            data = http_json(url, headers=self.headers())
            items.extend(data.get("results", []))

            next_link = data.get("_links", {}).get("next")
            if next_link:
                if next_link.startswith("http://") or next_link.startswith("https://"):
                    url = next_link
                else:
                    url = f"{CONFLUENCE_BASE_URL}{next_link}"
            else:
                url = None

        return items

    def get_direct_children(self, content_type: str, content_id: str) -> List[dict]:
        if content_type == "folder":
            url = f"{CONFLUENCE_BASE_URL}/wiki/api/v2/folders/{content_id}/direct-children?limit=100"
            return self._fetch_paginated_children(url)

        if content_type == "page":
            direct_url = f"{CONFLUENCE_BASE_URL}/wiki/api/v2/pages/{content_id}/direct-children?limit=100"
            try:
                return self._fetch_paginated_children(direct_url)
            except RuntimeError as exc:
                if "HTTP 404" not in str(exc):
                    raise

                print(f"  WARN: direct-children 404 for page {content_id}, fallback to /children")

                children_url = f"{CONFLUENCE_BASE_URL}/wiki/api/v2/pages/{content_id}/children?limit=100"
                try:
                    page_children = self._fetch_paginated_children(children_url)
                    normalized = []
                    for child in page_children:
                        normalized.append(
                            {
                                **child,
                                "type": "page",
                            }
                        )
                    return normalized
                except RuntimeError as exc2:
                    if "HTTP 404" in str(exc2):
                        print(f"  WARN: /children also 404 for page {content_id}, skipping")
                        return []
                    raise

        return []

    def collect_page_and_folder_items(self, root_folder_id: str) -> List[dict]:
        items: List[dict] = []
        seen = set()

        def walk(content_type: str, content_id: str) -> None:
            if content_id in SKIP_IDS:
                print(f"  SKIP: subtree skipped for {content_type} {content_id}")
                return

            try:
                children = self.get_direct_children(content_type, content_id)
            except Exception as exc:
                print(f"  WARN: failed to list children for {content_type} {content_id}: {exc}")
                return

            for child in children:
                child_type = (child.get("type") or "").lower()
                child_id = str(child.get("id"))

                if child_type not in {"page", "folder"}:
                    continue

                if child_id in SKIP_IDS:
                    print(f"  SKIP: {child_type} {child.get('title') or child_id} ({child_id})")
                    continue

                if child_id in seen:
                    continue
                seen.add(child_id)

                items.append(
                    {
                        "id": child_id,
                        "type": child_type,
                        "title": child.get("title") or f"{child_type}-{child_id}",
                        "parentId": str(child.get("parentId")) if child.get("parentId") is not None else content_id,
                    }
                )

                walk(child_type, child_id)

        walk("folder", root_folder_id)
        return items

    def get_page_storage(self, page_id: str) -> str:
        url = f"{CONFLUENCE_BASE_URL}/wiki/rest/api/content/{page_id}?expand=body.storage"
        data = http_json(url, headers=self.headers())
        return data["body"]["storage"]["value"]

    def start_export_conversion(self, page_id: str, storage_html: str) -> str:
        url = (
            f"{CONFLUENCE_BASE_URL}/wiki/rest/api/contentbody/convert/async/export_view"
            f"?contentIdContext={page_id}"
        )
        body = {"value": storage_html, "representation": "storage"}
        data = http_json(url, method="POST", headers=self.headers(), body=body)
        return data["asyncId"]

    def wait_for_export(self, async_id: str) -> str:
        url = f"{CONFLUENCE_BASE_URL}/wiki/rest/api/contentbody/convert/async/{async_id}"

        while True:
            time.sleep(POLLING_INTERVAL)
            result = http_json(url, headers=self.headers())

            if "value" in result:
                return result["value"]

            status = result.get("status")
            if status in ("WORKING", "PENDING", None):
                continue

            raise RuntimeError(f"Conversion failed: {json.dumps(result)}")


def build_html(title: str, body_html: str) -> str:
    safe_title = html.escape(title)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{ size: A4; margin: 18mm 14mm; }}
body {{ font-family: Arial, sans-serif; margin: 0; color: #222; }}
h1 {{ border-bottom: 1px solid #ccc; padding-bottom: 10px; margin-bottom: 20px; }}
h2, h3, h4, h5, h6 {{ page-break-after: avoid; }}
img {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ccc; padding: 6px; vertical-align: top; }}
pre {{ white-space: pre-wrap; word-wrap: break-word; }}
.wrapper {{ padding: 24px; }}
</style>
</head>
<body>
<div class="wrapper">
  <h1>{safe_title}</h1>
  {body_html}
</div>
</body>
</html>
"""


def write_pdf(html_text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_text, base_url=CONFLUENCE_BASE_URL).write_pdf(str(path))


def validate_env() -> None:
    if not CONFLUENCE_BASE_URL:
        raise RuntimeError("Missing CONFLUENCE_BASE_URL in .env")
    if not CONFLUENCE_EMAIL:
        raise RuntimeError("Missing CONFLUENCE_EMAIL in .env")
    if not CONFLUENCE_API_TOKEN:
        raise RuntimeError("Missing CONFLUENCE_API_TOKEN in .env")
    if not ROOT_ID:
        raise RuntimeError("Missing ROOT_ID in .env")


def print_type_summary(items: List[dict]) -> None:
    counts: Dict[str, int] = {}
    for item in items:
        t = item.get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1

    print("Item type summary:")
    for t in sorted(counts):
        print(f"  {t}: {counts[t]}")


def print_tree(items: List[dict], root_id: str, root_title: str) -> None:
    children_by_parent: Dict[str, List[dict]] = {}
    unresolved: List[dict] = []

    for item in items:
        parent_id = item.get("parentId")
        if parent_id:
            children_by_parent.setdefault(parent_id, []).append(item)
        else:
            unresolved.append(item)

    for children in children_by_parent.values():
        children.sort(key=lambda x: (x["type"], x["title"].lower()))

    print("\nTree:")
    print(f"[folder] {root_title} ({root_id})")

    visited = set()

    def walk(parent_id: str, depth: int) -> None:
        for child in children_by_parent.get(parent_id, []):
            visited.add(child["id"])
            indent = "  " * depth
            print(f"{indent}- [{child['type']}] {child['title']} ({child['id']})")
            walk(child["id"], depth + 1)

    walk(root_id, 1)

    leftovers = [x for x in items if x["id"] not in visited]
    if leftovers or unresolved:
        print("\nUnattached items:")
        shown = set()
        for item in unresolved + leftovers:
            if item["id"] in shown:
                continue
            shown.add(item["id"])
            print(
                f"- [{item['type']}] {item['title']} ({item['id']}) "
                f"parentId={item.get('parentId')}"
            )


def export_tree() -> None:
    validate_env()

    client = ConfluenceClient()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    root = client.get_root_folder_info(ROOT_ID)
    root_title = root.get("title", ROOT_ID)
    print(f"Root folder: {root_title} ({ROOT_ID})")

    items = client.collect_page_and_folder_items(ROOT_ID)
    print(f"Items found: {len(items)}")
    print_type_summary(items)
    print_tree(items, ROOT_ID, root_title)
    print()

    pages = [x for x in items if x["type"] == "page"]
    folders = [x for x in items if x["type"] == "folder"]

    print(f"\nFolders found: {len(folders)}")
    print(f"Pages found: {len(pages)}\n")

    failures = []

    for i, page in enumerate(pages, start=1):
        page_id = page["id"]
        title = page["title"]

        if page_id in SKIP_IDS:
            print(f"[{i}/{len(pages)}] SKIP export: {title} ({page_id})")
            continue

        safe = sanitize_filename(title)
        pdf_file = OUTPUT_DIR / f"{safe}_{page_id}.pdf"

        print(f"[{i}/{len(pages)}] Exporting page: {title} ({page_id})")

        try:
            storage = client.get_page_storage(page_id)
            async_id = client.start_export_conversion(page_id, storage)
            export_html = client.wait_for_export(async_id)

            full_html = build_html(title, export_html)
            write_pdf(full_html, pdf_file)

            print("  Saved PDF:", pdf_file)
        except Exception as exc:
            print("  ERROR:", exc)
            failures.append((page_id, title, str(exc)))

    print()
    print(f"Pages exported to PDF: {len(pages) - len(failures)}/{len(pages)}")

    if failures:
        print("\nFailed pages:")
        for page_id, title, err in failures:
            print(f"- {title} ({page_id}): {err}")


if __name__ == "__main__":
    export_tree()