"""Export the standalone survey database to Hugo using the survey's Python environment."""
from collections import defaultdict
from contextlib import closing
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import sys

SITE = Path(__file__).resolve().parents[1]
SOURCE = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else SITE.parent / "feed-forward-survey"
sys.path.insert(0, str(SOURCE))

from dataset_catalog import (
    cached_sizes, catalog, dataset_details, dataset_overrides, dataset_research_records,
    hub_names, saved_datasets, size_label,
)
from dataset_extract import ROLE_LABELS, aliases, dataset_key, dataset_name
from survey import paper_records
from web_view import (
    DATASET_FILTER, DECISION_LABELS, TABLE_SCRIPT, TIMELINE_SCRIPT, dataset_anchor, paper_rank,
    progress_steps, recent_papers, submission_month,
)

CONTENT = SITE / "content" / "survey"


def page(path, title, view=None, key=None, url=None):
    front = {"title": title, "type": "survey", "comments": False}
    if view:
        front["survey_view"] = view
    if key:
        front["survey_key"] = key
    if url:
        front["url"] = url
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(front, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cite_details(paper):
    metadata = {
        "arXiv ID": paper["id"], "arXiv version": paper["versioned_id"],
        "First submitted": paper["published"], "Last revised": paper["updated"],
        "Primary category": paper["primary_category"], "Categories": ", ".join(paper["categories"]),
        "DOI": paper["doi"], "Journal reference": paper["journal_ref"], "arXiv comment": paper["comment"],
        "Metadata fetched": paper["fetched_at"], "Saved HTML version": paper["html_version"],
    }
    return [{"label": label, "value": value} for label, value in metadata.items() if value]


def main():
    with closing(sqlite3.connect((SOURCE / "data/papers.db").as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        papers = list(paper_records(conn, os.environ.get("FF3D_LLM_MODEL", "qwen3.8-27b")))
        repo_ids = saved_datasets(conn)
        findings = dataset_research_records(conn)
        overrides = dataset_overrides(conn)
    data_path = SITE / "data/survey.json"
    previous = json.loads(data_path.read_text(encoding="utf-8")) if data_path.is_file() else {}
    metadata = catalog()
    measurements = cached_sizes(SOURCE / "data/dataset-sizes.json")
    known_repos = hub_names()
    alternate_names = defaultdict(set)
    spellings, _, families = aliases()
    for spelling in spellings.keys() | families.keys():
        alternate_names[dataset_key(spelling)].add(spelling)
    groups = {}
    months = defaultdict(lambda: [0] * 6)
    paper_data = {}
    order = sorted(papers, key=lambda paper: paper["title"].casefold())
    order.sort(key=paper_rank, reverse=True)
    for paper in order:
        pid = paper["id"]
        score, submitted = paper_rank(paper)
        month = submission_month(paper)
        if month:
            months[month][score] += 1
        screening = paper["screening"] or {}
        citations = paper.get("citation") or {}
        steps = progress_steps(paper)
        progress = next((index for index, (state, _) in enumerate(steps) if state != "done"), len(steps))
        extraction = paper["dataset_extraction"]
        paper_links = []
        for item in extraction["links"] if extraction else ():
            name = dataset_name(item["name"], pid)
            group = groups.setdefault(dataset_key(name), {"name": name, "papers": {}, "names": set()})
            group["names"].add(item["name"])
            group["papers"].setdefault(pid, set()).add(ROLE_LABELS[item["role"]])
            paper_links.append({"name": name, "anchor": dataset_anchor(name), "role": ROLE_LABELS[item["role"]],
                                "evidence": item["evidence"]})
        paper_data[pid] = {
            "title": paper["title"], "arxiv": paper["abs_url"], "submitted": submitted, "month": month,
            "score": score, "decision": DECISION_LABELS.get(screening.get("decision"), ""),
            "confidence": screening.get("confidence", ""), "screening": screening,
            "prompt": paper["prompt_version"], "model": paper["model"], "screened_at": paper["screened_at"],
            "abstract": paper["abstract"], "authors": ", ".join(author["name"] for author in paper["authors"]),
            "pdf": next((item["href"] for item in paper["links"] if item.get("title") == "pdf"), ""),
            "citations": citations["count"] if citations.get("count") is not None else -1,
            "citation_url": citations.get("url"), "progress": progress, "steps": [label for _, label in steps],
            "datasets": paper_links, "extraction": extraction["source"] if extraction else "",
            "metadata": cite_details(paper), "sources": paper["sources"],
        }
    for repo in repo_ids:
        name = known_repos.get(repo.casefold(), repo)
        group = groups.setdefault(dataset_key(name), {"name": name, "papers": {}, "names": set()})
        group["names"].add(repo)
        group["repo"] = repo

    dataset_data = {}
    for key, group in sorted(groups.items()):
        entry = metadata.get(key, {})
        ranked = sorted(group["papers"], key=lambda pid: (paper_data[pid]["score"], paper_data[pid]["submitted"]),
                        reverse=True)
        values = dataset_details(entry, measurements.get(key, {}), findings.get(key, {}),
                                 group.get("repo", ""), overrides.get(key))
        title = values.get("name") or group["name"]
        dataset_data[dataset_anchor(group["name"])] = {
            "name": title, "names": " | ".join(sorted(group["names"] | alternate_names[key] | {group["name"], title})),
            "papers": [{"id": pid, "roles": ", ".join(sorted(group["papers"][pid]))} for pid in ranked],
            "scores": "".join(str(paper_data[pid]["score"]) for pid in ranked),
            "url": values["dataset_url"], "sources": [source for source in values["sources"] if source != values["dataset_url"]],
            "sizes": [f'{item["value"]} · {item["scope"]}' for item in values["downloads"]],
            "size_gb": values["size_gb"], "size_label": size_label(values["size_gb"]), "size_scope": values["size_scope"],
            "depth": values["depth"], "origin": values["origin"], "sensors": values["sensors"],
            "data_types": values["data_types"], "edited_at": values["edited_at"],
            "description": values["description"], "description_source": values["description_source"],
            "dimensions": values["dimensions"],
            "publication": entry.get("publication"),
        }

    updated = datetime.now().astimezone()
    listed = [item["size_gb"] for item in dataset_data.values() if item["size_gb"] is not None]
    data = {
        "updated": updated.isoformat(timespec="seconds"), "updated_label": updated.strftime("%Y-%m-%d %H:%M"),
        "papers": paper_data, "paper_order": list(paper_data), "datasets": dataset_data,
        "new_paper_order": [paper["id"] for paper in recent_papers(papers, updated)],
        "dataset_order": sorted(dataset_data, key=lambda key: (-len(dataset_data[key]["papers"]),
                                                              dataset_data[key]["name"].casefold())),
        "months": sorted(months, reverse=True), "years": sorted({month[:4] for month in months}, reverse=True),
        "timeline": dict(months), "screened": sum(p["score"] > 0 for p in paper_data.values()),
        "central": sum(p["score"] == 5 for p in paper_data.values()),
        "datasets_checked": sum(bool(p["extraction"]) for p in paper_data.values()),
        "size_total": size_label(sum(listed) if listed else None), "sizes_listed": len(listed),
    }
    (SITE / "data").mkdir(exist_ok=True)
    data_path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    page(CONTENT / "_index.md", "Feed-forward 3D survey")
    page(CONTENT / "papers.md", "Papers", "papers", url="/survey/papers.html")
    page(CONTENT / "new-papers.md", "New papers", "new-papers", url="/survey/new-papers.html")
    page(CONTENT / "datasets.md", "Datasets", "datasets", url="/survey/datasets.html")
    page(CONTENT / "timeline.md", "Timeline", "timeline", url="/survey/timeline.html")

    for directory, values, view, url_dir, title in (
        ("papers", paper_data, "paper", "papers", "title"),
        ("dataset-details", dataset_data, "dataset", "datasets", "name"),
    ):
        folder = CONTENT / directory
        for key, value in values.items():
            page(folder / (key + ".md"), value[title], view, key, f"/survey/{url_dir}/{key}.html")
        for key in previous.get(url_dir, {}).keys() - values.keys():
            (folder / (key + ".md")).unlink(missing_ok=True)
    timeline_script = TIMELINE_SCRIPT.replace("index.html?month=", "papers.html?month=")
    for view, script in {"papers": TABLE_SCRIPT, "datasets": DATASET_FILTER, "timeline": timeline_script}.items():
        path = SITE / "assets/js/survey" / (view + ".js")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(script.strip() + "\n", encoding="utf-8")
    print(f"Imported {len(paper_data):,} papers and {len(dataset_data):,} datasets.")


if __name__ == "__main__":
    main()
