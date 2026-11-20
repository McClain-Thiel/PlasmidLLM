"""Specificity, complete-ORF, and peptide reward; independent of the frozen v1 scorer.

The ORF term is a sequence-level training surrogate, not a biological function claim.
AMRFinderPlus remains an evaluation-only tool. All detectable hard tokens are searched
for every candidate, so unrequested components count against precision.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.Seq import Seq

from post_training.scorers.base import Scorer
from post_training.scorers.plannotate import _clean_dna, _parse_hard_tokens

TAG_PEPTIDES = {
    "<TAG_HIS>": {"HHHHHH"},
    "<TAG_FLAG>": {"DYKDDDDK"},
    "<TAG_HA>": {"YPYDVPDYA"},
    "<TAG_MYC>": {"EQKLISEEDL"},
    "<TAG_V5>": {"GKPIPNPLLGLDST"},
    "<TAG_NLS>": {"PKKKRKV", "KRPAATKKAGQAKKKK", "PPKKKRKV"},
}
STARTS = {"ATG", "GTG", "TTG"}
STOPS = {"TAA", "TAG", "TGA"}
_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_CODON_TABLE_11 = {
    a + b + c: str(Seq(a + b + c).translate(table=11))
    for a in "ACGT" for b in "ACGT" for c in "ACGT"
    if (a + b + c) not in STOPS
}


def complete_orfs(dna: str, min_aa: int = 30) -> list[str]:
    """Translate start-to-first-stop ORFs on both strands of a circular molecule.

    Include nested starts, but never wrap more than one molecule. Ambiguous ORFs
    cannot establish intact coding sequence. Table 11 alternative starts encode M.
    """
    dna = dna.upper()
    dna_len = len(dna)
    min_nt = min_aa * 3
    codon_table = _CODON_TABLE_11
    starts_set = STARTS
    stops_set = STOPS
    proteins: set[str] = set()
    rc = dna.translate(_RC_TABLE)[::-1]
    for strand in (dna, rc):
        doubled = strand + strand
        limit = len(doubled) - 2
        for frame in range(3):
            starts: list[tuple[int, int]] = []
            aa_buf: list[str] = []
            for pos in range(frame, limit, 3):
                codon = doubled[pos:pos + 3]
                aa = codon_table.get(codon)
                if aa is None:
                    if starts and codon in stops_set:
                        segment = "".join(aa_buf)
                        for start, aa_idx in starts:
                            length = pos - start
                            if min_nt <= length and length + 3 <= dna_len:
                                proteins.add("M" + segment[aa_idx + 1:])
                    starts.clear()
                    aa_buf.clear()
                else:
                    if pos < dna_len and codon in starts_set:
                        starts.append((pos, len(aa_buf)))
                    if starts:
                        aa_buf.append(aa)
    return sorted(proteins)


def reward_components(
    expected: set[str], detected: set[str], intact: set[str],
    f1_weight: float = 0.6, orf_weight: float = 0.3, tag_weight: float = 0.1,
) -> dict[str, Any]:
    """Normalize over applicable terms; absent AMR/TAG requests earn no free bonus."""
    found = expected & detected
    precision = len(found) / len(detected) if detected else 0.0
    recall = len(found) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if found else 0.0
    amr = {t for t in expected if t.startswith("<AMR_")}
    tags = {t for t in expected if t.startswith("<TAG_")}
    orf_recall = len(amr & intact) / len(amr) if amr else 0.0
    tag_recall = len(tags & detected) / len(tags) if tags else 0.0
    denominator = f1_weight + orf_weight * bool(amr) + tag_weight * bool(tags)
    reward = (f1_weight * f1 + orf_weight * orf_recall + tag_weight * tag_recall)
    reward = reward / denominator if expected else 0.0
    return dict(reward=reward, f1=f1, precision=precision, recall=recall,
                intact_orf_recall=orf_recall, peptide_tag_recall=tag_recall,
                extra_components=len(detected - expected),
                extra_ori_amr=sum(t.startswith(("<ORI_", "<AMR_"))
                                  for t in detected - expected),
                expected=sorted(expected), detected=sorted(detected), intact=sorted(intact))


class PlannotateV2Scorer(Scorer):
    def __init__(
        self, motif_registry_path: str | None = None,
        motif_registry_df: pd.DataFrame | None = None, db_dir: str | None = None,
        min_identity: float = 90.0, min_coverage: float = 80.0,
        intact_coverage: float = 90.0, num_threads: int = 4,
        f1_weight: float = 0.6, orf_weight: float = 0.3, tag_weight: float = 0.1,
    ) -> None:
        if motif_registry_df is None and motif_registry_path is None:
            raise ValueError("A motif registry is required")
        if not 0 < min_identity <= 100 or not 0 < min_coverage <= intact_coverage <= 100:
            raise ValueError("Identity/coverage must be percentages in (0, 100]")
        if f1_weight <= 0 or min(orf_weight, tag_weight) < 0 or num_threads < 1:
            raise ValueError("Invalid reward weights or thread count")
        registry = (motif_registry_df if motif_registry_df is not None
                    else pd.read_parquet(motif_registry_path))
        self.min_identity = min_identity
        self.min_coverage = min_coverage
        self.intact_coverage = intact_coverage
        self.num_threads = num_threads
        self.weights = dict(f1_weight=f1_weight, orf_weight=orf_weight, tag_weight=tag_weight)
        self.db_dir = Path(tempfile.mkdtemp(prefix="reward_v2_", dir=db_dir))
        self.tags = {t: set(peptides) for t, peptides in TAG_PEPTIDES.items()
                     if t in set(registry.token)}
        self.ref_tokens: dict[str, str] = {}
        nucleotide, protein, intact = [], [], []
        for row in registry.itertuples():
            token = row.token
            if not _parse_hard_tokens(token) or not isinstance(row.sequence, str):
                continue
            seq = row.sequence.upper().strip()
            if token.startswith("<TAG_"):
                if token == "<TAG_NLS>" and row.seq_type == "dna" and len(seq) % 3 == 0:
                    peptide = str(Seq(seq).translate(table=11))
                    if re.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+", peptide):
                        self.tags.setdefault(token, set()).add(peptide)
                if token == "<TAG_GST>" and row.sseqid == "GST" and row.seq_type == "dna":
                    peptide = str(Seq(seq).translate(table=11)).removesuffix("*")
                    if "*" in peptide:
                        raise ValueError("GST reference contains an internal stop")
                    self.tags.setdefault(token, set()).add(peptide)
                continue
            if row.seq_type not in {"dna", "nucleotide", "protein"}:
                continue
            ref = f"r{len(self.ref_tokens)}"
            self.ref_tokens[ref] = token
            if row.seq_type == "protein":
                peptide = seq.removesuffix("*")
                if re.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+", peptide):
                    protein.append((ref, peptide))
                    if token.startswith("<AMR_"):
                        intact.append((ref, peptide))
            elif re.fullmatch("[ACGT]+", seq):
                is_amr = token.startswith("<AMR_")
                # Promoter matches cannot stand in for the resistance gene.
                if is_amr and ("promoter" in row.sseqid.lower() or len(seq) < 180):
                    continue
                nucleotide.append((ref, seq))
                if is_amr and len(seq) % 3 == 0 and seq[:3] in STARTS and seq[-3:] in STOPS:
                    peptide = str(Seq(seq).translate(table=11, cds=True))
                    intact.append((ref, peptide))
        self.databases = {}
        self.sizes = {}
        self.supported = set(self.tags)
        self.intact_supported = {self.ref_tokens[r] for r, _ in intact}
        for name, entries, kind in (("dna", nucleotide, "nucl"),
                                    ("protein", protein, "prot"),
                                    ("intact", intact, "prot")):
            if not entries:
                continue
            path = self.db_dir / name
            self._write_fasta(path.with_suffix(".fasta"), entries)
            self._run(["makeblastdb", "-in", str(path.with_suffix(".fasta")),
                       "-dbtype", kind, "-out", str(path), "-parse_seqids"])
            self.databases[name] = path
            self.sizes[name] = len(entries)
            if name != "intact":
                self.supported.update(self.ref_tokens[r] for r, _ in entries)
        self.last_diagnostics: dict[str, float] = {}

    @staticmethod
    def _run(cmd: list[str]) -> None:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"{cmd[0]} exited {result.returncode}: {result.stderr.strip()}")

    @staticmethod
    def _write_fasta(path: Path, entries: list[tuple[str, str]]) -> None:
        with path.open("w") as f:
            for name, seq in entries:
                f.write(f">{name}\n{seq}\n")

    def _search(self, queries: list[tuple[str, str]], name: str, program: str,
                coverage: float) -> dict[str, set[str]]:
        if not queries or name not in self.databases:
            return {}
        query = self.db_dir / f"query_{name}.fasta"
        output = self.db_dir / f"hits_{name}.tsv"
        self._write_fasta(query, queries)
        cmd = [program, "-query", str(query), "-db", str(self.databases[name]),
               "-out", str(output), "-outfmt", "6 qseqid sseqid pident sstart send slen",
               "-evalue", "1e-5", "-max_target_seqs", str(self.sizes[name]),
               "-num_threads", str(self.num_threads)]
        if program == "blastn":
            cmd.extend(["-task", "blastn", "-word_size", "7"])
        self._run(cmd)
        hits: dict[str, set[str]] = {}
        for line in output.read_text().splitlines():
            q, subject, identity, start, end, length = line.split("\t")
            cov = 100 * (abs(int(end) - int(start)) + 1) / int(length)
            if float(identity) >= self.min_identity and cov >= coverage:
                ref = subject.rsplit("|", 1)[-1]
                hits.setdefault(q, set()).add(self.ref_tokens[ref])
        return hits

    def score_batch_detailed(self, prompts: list[str], sequences: list[str]) -> list[dict]:
        if len(prompts) != len(sequences):
            raise ValueError("Prompt/sequence counts differ")
        expected = [set(_parse_hard_tokens(p)) for p in prompts]
        requested = set().union(*expected)
        unsupported = requested - self.supported
        unsupported |= {t for t in requested if t.startswith("<AMR_")} - self.intact_supported
        if unsupported:
            raise ValueError(f"No validated reward references for: {sorted(unsupported)}")
        dna = [_clean_dna(s)[0] for s in sequences]
        orfs = [complete_orfs(s) for s in dna]
        unique_dna = list(dict.fromkeys(s for s in dna if len(s) >= 20))
        dna_to_uid = {s: f"u{idx}" for idx, s in enumerate(unique_dna)}
        dna_queries = [(dna_to_uid[s], s + s) for s in unique_dna]

        unique_prots = list(dict.fromkeys(p for ps in orfs for p in ps))
        prot_to_pid = {p: f"p{idx}" for idx, p in enumerate(unique_prots)}
        protein_queries = [(pid, p) for p, pid in prot_to_pid.items()]

        with ThreadPoolExecutor(max_workers=3) as pool:
            f_dna = pool.submit(self._search, dna_queries, "dna", "blastn", self.min_coverage)
            f_prot = pool.submit(self._search, dna_queries, "protein", "blastx", self.min_coverage)
            f_intact = pool.submit(self._search, protein_queries, "intact", "blastp", self.intact_coverage)
            u_dna_hits = f_dna.result()
            u_protein_hits = f_prot.result()
            u_intact_hits = f_intact.result()

        results = []
        for i, req in enumerate(expected):
            uid = dna_to_uid.get(dna[i])
            intact: set[str] = set()
            for p in orfs[i]:
                hits_p = u_intact_hits.get(prot_to_pid[p])
                if hits_p:
                    intact.update(hits_p)
            detected = (
                (u_dna_hits.get(uid, set()) if uid else set())
                | (u_protein_hits.get(uid, set()) if uid else set())
                | intact
            )
            detected |= {t for t, peptides in self.tags.items()
                         if any(peptide in p for peptide in peptides for p in orfs[i])}
            results.append(reward_components(req, detected, intact, **self.weights))
        keys = ("f1", "precision", "recall", "intact_orf_recall", "peptide_tag_recall",
                "extra_components", "extra_ori_amr")
        self.last_diagnostics = {k: sum(r[k] for r in results) / len(results)
                                 for k in keys} if results else {}
        for prefix, metric in (("<AMR_", "intact_orf_recall"),
                               ("<TAG_", "peptide_tag_recall")):
            applicable = [r for r in results if any(t.startswith(prefix) for t in r["expected"])]
            self.last_diagnostics[metric] = (
                sum(r[metric] for r in applicable) / len(applicable) if applicable else 0.0)
            self.last_diagnostics[f"{metric}_n"] = len(applicable)
        return results

    def score_batch(self, prompts: list[str], sequences: list[str], **kwargs: Any) -> list[float]:
        return [r["reward"] for r in self.score_batch_detailed(prompts, sequences)]

    def score_sequence(self, prompt: str, sequence: str, **kwargs: Any) -> float:
        return self.score_batch([prompt], [sequence])[0]

    def score_sequence_detailed(self, prompt: str, sequence: str, **kwargs: Any) -> dict:
        return self.score_batch_detailed([prompt], [sequence])[0]
