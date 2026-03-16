#!/usr/bin/env python3
import base64
import html
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Set
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from markdownify import markdownify as md

try:
    from kaggle_secrets import UserSecretsClient  # type: ignore
except ImportError:
    UserSecretsClient = None


def get_config_value(name: str, default: str = "") -> str:
    """
    Resolve config in this order:
    1. Kaggle secrets (if available)
    2. Environment variables / .env
    3. Provided default
    """
    if UserSecretsClient is not None:
        try:
            user_secrets = UserSecretsClient()
            value = user_secrets.get_secret(name)
            if value is not None and str(value).strip() != "":
                return str(value).strip()
        except Exception:
            pass

    value = os.getenv(name, default)
    return str(value).strip() if value is not None else default


load_dotenv()

CONFLUENCE_BASE_URL = get_config_value("CONFLUENCE_BASE_URL", "").rstrip("/")
CONFLUENCE_EMAIL = get_config_value("CONFLUENCE_EMAIL", "")
CONFLUENCE_API_TOKEN = get_config_value("CONFLUENCE_API_TOKEN", "")
ROOT_ID = get_config_value("ROOT_ID", "")
OUTPUT_DIR = Path(get_config_value("OUTPUT_DIR", "retrieved_docs"))
POLLING_INTERVAL = int(get_config_value("POLLING_INTERVAL_IN_SECONDS", "2"))

# Only keep IDs here if you want to skip them manually.
# If a folder ID is here, its whole subtree will not be traversed.
MANUAL_SKIP_IDS: Set[str] = {
    "116621984",
    "158960566",
}


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180] or "untitled"

def html_to_markdown(html_text: str) -> str:
    """
    Convert Confluence export HTML to Markdown.
    """
    markdown = md(
        html_text,
        heading_style="ATX",
        bullets="-",
    )

    markdown = re.sub(r"\n{3,}", "\n\n", markdown)

    return markdown.strip()


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

    def collect_page_and_folder_items(self, root_folder_id: str, skip_ids: Set[str]) -> List[dict]:
        items: List[dict] = []
        seen = set()

        def walk(content_type: str, content_id: str) -> None:
            if content_id in skip_ids:
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

                if child_id in skip_ids:
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

    existing_md_ids = {
        re.search(r"_(\d+)\.md$", f.name).group(1)
        for f in OUTPUT_DIR.glob("*.md")
        if re.search(r"_(\d+)\.md$", f.name)
    }

    print(f"Existing exported Markdown files: {len(existing_md_ids)}")
    if existing_md_ids:
        print("Existing page IDs:", ", ".join(sorted(existing_md_ids)))

    root = client.get_root_folder_info(ROOT_ID)
    root_title = root.get("title", ROOT_ID)
    print(f"Root folder: {root_title} ({ROOT_ID})")

    items = client.collect_page_and_folder_items(ROOT_ID, MANUAL_SKIP_IDS)
    print(f"Items found: {len(items)}")
    print_type_summary(items)
    print_tree(items, ROOT_ID, root_title)
    print()

    pages = [x for x in items if x["type"] == "page"]
    folders = [x for x in items if x["type"] == "folder"]

    print(f"\nFolders found: {len(folders)}")
    print(f"Pages found: {len(pages)}\n")

    failures = []
    exported_count = 0
    skipped_existing_count = 0
    skipped_manual_count = 0

    for i, page in enumerate(pages, start=1):
        page_id = page["id"]
        title = page["title"]

        if page_id in MANUAL_SKIP_IDS:
            print(f"[{i}/{len(pages)}] SKIP manual: {title} ({page_id})")
            skipped_manual_count += 1
            continue

        if page_id in existing_md_ids:
            print(f"[{i}/{len(pages)}] SKIP existing md: {title} ({page_id})")
            skipped_existing_count += 1
            continue

        safe = sanitize_filename(title)
        md_file = OUTPUT_DIR / f"{safe}_{page_id}.md"

        print(f"[{i}/{len(pages)}] Exporting page: {title} ({page_id})")

        try:
            storage = client.get_page_storage(page_id)
            async_id = client.start_export_conversion(page_id, storage)
            export_html = client.wait_for_export(async_id)

            markdown = html_to_markdown(export_html)

            md_file.write_text(markdown, encoding="utf-8")

            print("  Saved Markdown:", md_file)
            exported_count += 1
        except Exception as exc:
            print("  ERROR:", exc)
            failures.append((page_id, title, str(exc)))

    print()
    print(f"Pages total: {len(pages)}")
    print(f"Pages exported now: {exported_count}")
    print(f"Pages skipped (existing md): {skipped_existing_count}")
    print(f"Pages skipped (manual): {skipped_manual_count}")
    print(f"Pages failed: {len(failures)}")

    if failures:
        print("\nFailed pages:")
        for page_id, title, err in failures:
            print(f"- {title} ({page_id}): {err}")

if __name__ == "__main__":
    export_tree()