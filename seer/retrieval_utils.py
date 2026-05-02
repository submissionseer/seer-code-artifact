import json
import time
from array import array
from pathlib import Path

from seer.datasets import get_dataset_adapter

def load_hotpot_fullwiki(raw_dir: Path, split: str = "dev") -> list[dict]:
    hotpot_dir = raw_dir / "hotpot_fullwiki"
    if split == "dev":
        path = hotpot_dir / "hotpot_dev_fullwiki_v1.json"
    elif split == "train":
        sample_path = hotpot_dir / "hotpot_train_sample_10k.json"
        full_path = hotpot_dir / "hotpot_train_v1.1.json"
        path = sample_path if sample_path.exists() else full_path
    elif split == "test":
        path = hotpot_dir / "hotpot_test_fullwiki_v1.json"
    else:
        raise ValueError(f"Unknown split: {split}")
    
    if not path.exists():
        raise FileNotFoundError(f"HotpotQA not found at {path}")
    with open(path) as f:
        return json.load(f)

def get_num_hops(qid: str, dataset: str, question: dict | None = None) -> int:
    """
    Backward-compatible helper used by eval scripts.

    Prefers dataset adapter logic when the dataset is known; otherwise falls back
    to historical 2-hop default behavior.
    """
    try:
        adapter = get_dataset_adapter(dataset)
    except KeyError:
        return 2
    try:
        return int(adapter.infer_num_hops(qid, question))
    except Exception:
        return 2


def extract_gold_titles(question: dict, dataset: str | None = None) -> list[str]:
    """
    Backward-compatible helper for rollout/eval code.

    If dataset is not supplied, infer from the record shape when possible.
    """
    dataset_name = (dataset or question.get("dataset") or "").strip()
    if not dataset_name:
        if "question_decomposition" in question or "paragraphs" in question or "paragraphs_info" in question:
            dataset_name = "musique"
        elif "supporting_facts" in question or "context" in question:
            dataset_name = "hotpot"

    if dataset_name:
        try:
            return get_dataset_adapter(dataset_name).extract_gold_titles(question)
        except KeyError:
            pass

    # Historical fallback behavior (list-style supporting_facts, no dataset info)
    if "gold_titles" in question:
        return question["gold_titles"]
    supporting_facts = question.get("supporting_facts", [])
    if isinstance(supporting_facts, list):
        titles = [sf[0] if isinstance(sf, (list, tuple)) else sf.get("title", "") for sf in supporting_facts]
    elif isinstance(supporting_facts, dict):
        titles = list(supporting_facts.get("title", []) or [])
    else:
        titles = []

    seen = set()
    out = []
    for title in titles:
        t = str(title or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

class ColBERTRetriever:
    def __init__(self, index_path: Path, collection_path: Path, logger=None):
        self.logger = logger
        self.index_path = index_path
        self.collection_path = collection_path
        self.collection_tsv = self.collection_path / "collection.tsv"
        self.collection_file = None
        self.offsets = array("Q")
        self._load_collection_offsets()
        
        from colbert import Searcher
        from colbert.infra import ColBERTConfig
        self.searcher = Searcher(
            index=str(index_path.resolve()),
            config=ColBERTConfig(doc_maxlen=256, query_maxlen=32)
        )

    def _load_collection_offsets(self):
        self.collection_file = open(self.collection_tsv, "r", encoding="utf-8")
        _ = self.collection_file.readline()
        offset = self.collection_file.tell()
        line = self.collection_file.readline()
        while line:
            self.offsets.append(offset)
            offset = self.collection_file.tell()
            line = self.collection_file.readline()
        self.collection_file.seek(0)

    def _get_entry(self, pid: int):
        if pid <= 0 or pid > len(self.offsets): return None
        self.collection_file.seek(self.offsets[pid - 1])
        line = self.collection_file.readline()
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            # Malformed line - skip it
            return None
        return parts[1], parts[2] # text, title

    def retrieve(self, query: str, top_k: int = 10, exclude_pids: set = None) -> list[dict]:
        fetch_k = min(top_k + len(exclude_pids or []), 100)
        results = self.searcher.search(query, k=fetch_k)
        passages = []
        for pid, rank, score in zip(*results):
            if exclude_pids and pid in exclude_pids: continue
            entry = self._get_entry(int(pid))
            if not entry: continue
            text, title = entry
            passages.append({"doc_id": int(pid), "title": title.strip(), "text": text.strip(), "score": float(score)})
            if len(passages) >= top_k: break
        return passages

def init_retriever_with_retry(index_path, collection_path, logger=None):
    for attempt in range(3):
        try:
            return ColBERTRetriever(index_path, collection_path, logger)
        except Exception as e:
            print(f"Retriever init attempt {attempt+1} failed: {e}")
            time.sleep(5)
    raise RuntimeError("Failed to initialize retriever")
