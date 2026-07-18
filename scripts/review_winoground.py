#!/usr/bin/env python3
"""Local, resumable visual-review helper for one Winoground canonical run."""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import os
import tempfile
import threading
import webbrowser
from collections import Counter
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

REVIEW_FIELDS = (
    "sample_id",
    "review_group",
    "mapping_checked",
    "visual_review_status",
    "annotation_issue",
    "notes",
)
VISUAL_STATUSES = {"pass", "issue", "uncertain"}
ANNOTATION_ISSUES = {"none", "possible", "confirmed"}
GROUP_ORDER = {
    "tie": 0,
    "both_directions_incorrect": 1,
    "image_to_text_only": 2,
    "text_to_image_only": 3,
    "group_correct": 4,
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_CSV = REPOSITORY_ROOT / (
    "reports/experiments/winoground/reviewed_sample_ids.csv"
)


class ReviewConflict(ValueError):
    """Raised when a completed review row would be overwritten."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a local-only, resumable Winoground human-review page"
    )
    parser.add_argument("--run-dir", required=True, help="canonical Winoground run directory")
    parser.add_argument(
        "--review-csv",
        default=str(DEFAULT_REVIEW_CSV),
        help="promotion-compatible review CSV to update",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate inputs and print review progress without starting a server",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {line_number} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path} line {line_number} must contain an object")
        rows.append(row)
    return rows


def _resolve_path(value: str, *, root: Path = REPOSITORY_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _review_group(prediction: dict[str, Any]) -> str:
    if prediction.get("tie") is True:
        return "tie"
    if prediction.get("group_correct") is True:
        return "group_correct"
    if prediction.get("image_to_text_correct") is True:
        return "image_to_text_only"
    if prediction.get("text_to_image_correct") is True:
        return "text_to_image_only"
    return "both_directions_incorrect"


def _completed(row: dict[str, str]) -> bool:
    mapping = row["mapping_checked"].strip().lower()
    visual = row["visual_review_status"].strip().lower()
    annotation = row["annotation_issue"].strip().lower()
    notes = row["notes"].strip()
    return (
        mapping == "true"
        and visual in VISUAL_STATUSES
        and annotation in ANNOTATION_ISSUES
        and (visual not in {"issue", "uncertain"} or bool(notes))
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


@dataclass
class ReviewWorkspace:
    run_dir: Path
    review_path: Path
    image_root: Path
    rows: list[dict[str, str]]
    items: list[dict[str, Any]]

    @classmethod
    def load(cls, run_dir: str | Path, review_path: str | Path) -> ReviewWorkspace:
        run = _resolve_path(str(run_dir))
        review = _resolve_path(str(review_path))
        config_path = run / "config.resolved.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
            raise ValueError("resolved config must contain a data mapping")
        data = config["data"]
        annotation_value = data.get("annotation_file")
        image_root_value = data.get("image_root")
        if not isinstance(annotation_value, str) or not annotation_value.strip():
            raise ValueError("resolved config does not declare data.annotation_file")
        if not isinstance(image_root_value, str) or not image_root_value.strip():
            raise ValueError("resolved config does not declare data.image_root")
        annotation_path = _resolve_path(annotation_value)
        image_root = _resolve_path(image_root_value)

        predictions = _load_jsonl(run / "predictions.jsonl")
        annotations = _load_jsonl(annotation_path)
        with review.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            missing = [field for field in REVIEW_FIELDS if field not in header]
            if missing:
                raise ValueError(f"review CSV is missing headers: {', '.join(missing)}")
            review_rows = [
                {field: (row.get(field) or "") for field in REVIEW_FIELDS} for row in reader
            ]

        prediction_ids = [str(row.get("sample_id") or "") for row in predictions]
        annotation_ids = [str(row.get("sample_id") or "") for row in annotations]
        review_ids = [row["sample_id"].strip() for row in review_rows]
        if len(prediction_ids) != 400 or len(set(prediction_ids)) != 400:
            raise ValueError("canonical predictions must contain 400 unique sample IDs")
        if annotation_ids != prediction_ids:
            raise ValueError("normalized annotation IDs/order do not match canonical predictions")
        if review_ids != prediction_ids:
            raise ValueError("review queue IDs/order do not match canonical predictions")

        items: list[dict[str, Any]] = []
        for index, (prediction, annotation, review_row) in enumerate(
            zip(predictions, annotations, review_rows, strict=True)
        ):
            expected_group = _review_group(prediction)
            if review_row["review_group"].strip() != expected_group:
                raise ValueError(
                    f"review_group mismatch for {review_row['sample_id']}: "
                    f"expected {expected_group}"
                )
            scores = prediction.get("scores")
            if not isinstance(scores, list) or len(scores) != 4:
                raise ValueError(f"prediction scores must have four values at row {index}")
            images: list[Path] = []
            for field in ("image_0", "image_1"):
                value = annotation.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"annotation {index} has invalid {field}")
                image_path = image_root / value
                if not _inside(image_path, image_root):
                    raise ValueError(f"annotation {index} image escapes the configured image root")
                if not image_path.is_file():
                    raise FileNotFoundError(f"review image is missing: {image_path}")
                images.append(image_path)
            items.append(
                {
                    "sample_id": review_row["sample_id"],
                    "review_group": expected_group,
                    "caption_0": str(annotation.get("caption_0") or ""),
                    "caption_1": str(annotation.get("caption_1") or ""),
                    "tags": annotation.get("tags") or [],
                    "scores": scores,
                    "images": images,
                    "original_index": index,
                }
            )
        items.sort(
            key=lambda item: (
                GROUP_ORDER.get(item["review_group"], len(GROUP_ORDER)),
                item["original_index"],
            )
        )
        return cls(run, review, image_root, review_rows, items)

    def summary(self) -> dict[str, Any]:
        visual = Counter()
        annotation = Counter()
        completed = 0
        for row in self.rows:
            visual_value = row["visual_review_status"].strip().lower()
            annotation_value = row["annotation_issue"].strip().lower()
            if visual_value in VISUAL_STATUSES:
                visual[visual_value] += 1
            if annotation_value in ANNOTATION_ISSUES:
                annotation[annotation_value] += 1
            completed += int(_completed(row))
        return {
            "review_csv": self.review_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "rows": len(self.rows),
            "completed_rows": completed,
            "remaining_rows": len(self.rows) - completed,
            "visual_review_status_counts": {
                status: visual[status] for status in sorted(VISUAL_STATUSES)
            },
            "annotation_issue_counts": {
                status: annotation[status] for status in sorted(ANNOTATION_ISSUES)
            },
        }

    def public_items(self) -> list[dict[str, Any]]:
        rows_by_id = {row["sample_id"]: row for row in self.rows}
        public: list[dict[str, Any]] = []
        for item in self.items:
            row = rows_by_id[item["sample_id"]]
            public.append(
                {
                    key: item[key]
                    for key in (
                        "sample_id",
                        "review_group",
                        "caption_0",
                        "caption_1",
                        "tags",
                        "scores",
                    )
                }
                | {
                    "completed": _completed(row),
                    "mapping_checked": row["mapping_checked"],
                    "visual_review_status": row["visual_review_status"],
                    "annotation_issue": row["annotation_issue"],
                    "notes": row["notes"],
                }
            )
        return public

    def image_path(self, sample_id: str, index: int) -> Path:
        if index not in {0, 1}:
            raise ValueError("image index must be 0 or 1")
        for item in self.items:
            if item["sample_id"] == sample_id:
                return item["images"][index]
        raise KeyError(sample_id)

    def save(
        self,
        *,
        sample_id: str,
        mapping_checked: bool,
        visual_review_status: str,
        annotation_issue: str,
        notes: str,
    ) -> None:
        row = next(
            (candidate for candidate in self.rows if candidate["sample_id"] == sample_id),
            None,
        )
        if row is None:
            raise ValueError("unknown sample_id")
        if _completed(row):
            raise ReviewConflict("completed review rows are read-only")
        visual = visual_review_status.strip().lower()
        annotation = annotation_issue.strip().lower()
        note_text = notes.strip()
        if mapping_checked is not True:
            raise ValueError("mapping_checked must be true")
        if visual not in VISUAL_STATUSES:
            raise ValueError("visual_review_status is invalid")
        if annotation not in ANNOTATION_ISSUES:
            raise ValueError("annotation_issue is invalid")
        if visual in {"issue", "uncertain"} and not note_text:
            raise ValueError(f"{visual} rows require notes")
        row.update(
            {
                "mapping_checked": "true",
                "visual_review_status": visual,
                "annotation_issue": annotation,
                "notes": note_text,
            }
        )
        self._write()

    def _write(self) -> None:
        self.review_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.review_path.name}.",
            suffix=".tmp",
            dir=self.review_path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, quoting=csv.QUOTE_ALL)
                writer.writeheader()
                writer.writerows(self.rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.review_path)
        finally:
            temporary.unlink(missing_ok=True)


def _page() -> bytes:
    title = html.escape("Winoground human review")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
body{{font:16px system-ui;margin:0;background:#f3f4f6;color:#111827}}
main{{max-width:1180px;margin:auto;padding:20px}} .bar,.panel{{background:white;border-radius:12px;
padding:16px;margin-bottom:16px;box-shadow:0 1px 4px #0002}} .images{{display:grid;
grid-template-columns:1fr 1fr;gap:16px}} img{{width:100%;max-height:480px;object-fit:contain;
background:#111;border-radius:8px}} table{{border-collapse:collapse}} td,th{{border:1px solid
#d1d5db;padding:8px}} label{{display:block;margin:10px 0}} select,textarea,button{{font:inherit;
padding:8px}} textarea{{width:100%;box-sizing:border-box}} button{{margin-right:8px}}
.warning{{color:#b45309}} .saved{{color:#047857}}
@media(max-width:760px){{.images{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="bar"><strong id="progress">Loading…</strong> <span id="position"></span></div>
<div class="panel"><h2 id="sample"></h2><p>Machine group: <code id="group"></code></p>
<p>Tags: <span id="tags"></span></p><div class="images">
<section><img id="image0"><h3>Caption 0</h3><p id="caption0"></p></section>
<section><img id="image1"><h3>Caption 1</h3><p id="caption1"></p></section></div></div>
<div class="panel"><h3>Score matrix</h3><table>
<tr><th></th><th>Caption 0</th><th>Caption 1</th></tr>
<tr><th>Image 0</th><td id="s00"></td><td id="s01"></td></tr>
<tr><th>Image 1</th><td id="s10"></td><td id="s11"></td></tr></table></div>
<div class="panel"><h3>Human review</h3><p class="warning">No field is preselected. Check both
image-caption mappings before saving.</p>
<label><input id="mapping" type="checkbox">
image_0↔caption_0 and image_1↔caption_1 mapping checked</label>
<label>Visual review status <select id="visual"><option value="">Select…</option>
<option>pass</option><option>issue</option><option>uncertain</option></select></label>
<label>Annotation issue <select id="annotation"><option value="">Select…</option>
<option>none</option><option>possible</option><option>confirmed</option></select></label>
<label>Notes (required for issue/uncertain)<textarea id="notes" rows="3"></textarea></label>
<p id="message"></p><button id="save">Save and next incomplete</button>
<button id="previous">Previous</button><button id="next">Next</button></div>
</main><script>
let items=[],index=0;
const q=id=>document.getElementById(id);
async function load(){{items=await (await fetch('/api/items')).json();
 const first=items.findIndex(x=>!x.completed);index=first<0?0:first;render();}}
function render(){{const x=items[index],done=items.filter(y=>y.completed).length;
 q('progress').textContent=`Completed ${{done}} / ${{items.length}};
 remaining ${{items.length-done}}`;
 q('position').textContent=`Queue position ${{index+1}} / ${{items.length}}`;
 q('sample').textContent=x.sample_id;q('group').textContent=x.review_group;
 q('tags').textContent=x.tags.join(', ');q('caption0').textContent=x.caption_0;
 q('caption1').textContent=x.caption_1;q('image0').src=`/image/${{x.sample_id}}/0`;
 q('image1').src=`/image/${{x.sample_id}}/1`;
 ['s00','s01','s10','s11'].forEach((id,i)=>q(id).textContent=Number(x.scores[i]).toFixed(8));
 q('mapping').checked=x.mapping_checked.toLowerCase()==='true';
 q('visual').value=x.visual_review_status;q('annotation').value=x.annotation_issue;
 q('notes').value=x.notes;q('message').textContent=x.completed?'Saved row (read-only).':'';
 q('message').className=x.completed?'saved':'';q('save').disabled=x.completed;}}
async function save(){{const x=items[index],payload={{sample_id:x.sample_id,
 mapping_checked:q('mapping').checked,visual_review_status:q('visual').value,
 annotation_issue:q('annotation').value,notes:q('notes').value}};
 const response=await fetch('/api/save',{{method:'POST',
 headers:{{'Content-Type':'application/json'}},
 body:JSON.stringify(payload)}});const result=await response.json();
 if(!response.ok){{q('message').textContent=result.error;q('message').className='warning';return;}}
 items=await (await fetch('/api/items')).json();const next=items.findIndex(y=>!y.completed);
 index=next<0?index:next;render();}}
q('save').onclick=save;q('previous').onclick=()=>{{index=(index+items.length-1)%items.length;render();}};
q('next').onclick=()=>{{index=(index+1)%items.length;render();}};load();
</script></body></html>""".encode()


def make_handler(workspace: ReviewWorkspace) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                body = _page()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/items":
                self._send_json(workspace.public_items())
                return
            parts = [unquote(part) for part in path.split("/") if part]
            if len(parts) == 3 and parts[0] == "image":
                try:
                    image = workspace.image_path(parts[1], int(parts[2]))
                    body = image.read_bytes()
                except (KeyError, OSError, ValueError):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                content_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 32768:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be an object")
                workspace.save(
                    sample_id=str(payload.get("sample_id") or ""),
                    mapping_checked=payload.get("mapping_checked") is True,
                    visual_review_status=str(payload.get("visual_review_status") or ""),
                    annotation_issue=str(payload.get("annotation_issue") or ""),
                    notes=str(payload.get("notes") or ""),
                )
            except ReviewConflict as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
                return
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"saved": True, "summary": workspace.summary()})

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> int:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("error: the review helper may bind only to a loopback address")
        return 2
    try:
        workspace = ReviewWorkspace.load(args.run_dir, args.review_csv)
    except (FileNotFoundError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(workspace.summary(), indent=2, sort_keys=True))
    if args.check_only:
        return 0
    server = ThreadingHTTPServer((args.host, args.port), make_handler(workspace))
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Local review page: {url}")
    print("Images remain local. Press Ctrl+C to stop; completed rows are saved immediately.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview helper stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
