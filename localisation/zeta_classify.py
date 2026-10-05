"""
zeta_classify.py

Given a single ECG in ZETA format (produced by
`ecg_preprocessing.ECGPreprocessor.to_zeta_input`), scores it against a
chosen subset of ZETA's expert-reviewed conditions and returns a
"possibility score" per condition -- ZETA's own zero-shot classification
mechanism (Eq. 1 in the paper: mean of per-observation P-vs-N softmax
scores, temperature 0.5).

This reuses ZETA's OWN objects and functions from `main.py` / `models/cmelt.py`
(`load_encoders`, `extract_language_features`, `get_diseases_probs`) rather
than reimplementing the M3AEModel forward pass -- getting the pooling /
projection / CLS-token order subtly wrong would silently produce meaningless
scores with no error raised, so reusing the real objects is safer than
reimplementing them from the paper description.

The one thing this file does NOT reuse verbatim is `extract_ecg_features`,
because that function's internal permute is conditioned on a dataset-name
STRING ("ptbxl" vs anything else) that only produces the correct tensor
orientation when paired with a specific upstream permute done inside
`inference()`'s DataLoader loop. Calling it directly on a single new ECG
(no DataLoader involved) makes it easy to end up with the wrong orientation
silently. Instead, the ECG-encoding steps are inlined here exactly as
`extract_ecg_features` performs them (same objects, same order), but fed
input in the one unambiguous orientation Conv1d actually requires:
channel-first (12, T) -- which is exactly what
`ECGPreprocessor.to_zeta_input` already returns, so no permute is needed
at all for this single-ECG case.

Assumes the caller has already inserted the ZETA repo root onto sys.path
before importing this module (see `run_pipeline.py`, which does this based
on a command-line argument rather than a hardcoded path) and that the
process's working directory is the ZETA repo root, since ZETA's own
`main.py` loads `checkpoints/best.pt` via a path relative to CWD, not to
the repo itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from main import load_encoders, extract_language_features, get_diseases_probs  # noqa: E402


@dataclass
class ObservationScore:
    """
    A single positive or negative observation sentence and its score.

    `index` is the sentence's position in the ORIGINAL P or N list for its
    condition (i.e. `observations.json`'s "P"/"N" arrays), captured before
    any score-based sorting. This is deliberately kept because some
    conditions' observation lists contain duplicate sentences (e.g. RBBB's
    "P" list repeats "abnormal q wave morphology") -- matching an
    observation by text alone would be ambiguous, so anything that needs to
    re-identify this exact observation later (e.g. re-scoring it against a
    cropped ECG) should use `index`, not `text`.
    """
    text: str
    score: float
    index: int


@dataclass
class ConditionEvidence:
    """
    Full evidence for one condition on one ECG: the aggregate possibility
    score plus every individual observation's score, so you can see *which*
    specific sentence drove the prediction rather than just the mean.
    """
    condition: str
    score: float                              # mean of positive_observations' scores
    positive_observations: List[ObservationScore]   # sorted, highest score first
    negative_observations: List[ObservationScore]   # sorted, highest score first

    @property
    def top_observation(self) -> ObservationScore:
        """The single highest-scoring positive observation -- the concrete
        'why' text to hand off to anyECG-chat's localization prompt."""
        return self.positive_observations[0]


class ZetaZeroShotClassifier:
    """
    Wraps ZETA's released checkpoint + expert-reviewed observation bank to
    classify a single (already-preprocessed) ECG against a chosen set of
    conditions.

    Loading the encoders and embedding every observation sentence happens
    ONCE at construction (neither depends on any specific ECG) -- repeated
    `classify()` calls only pay for the ECG forward pass.
    """

    def __init__(
        self,
        observations_path: str = "configs/observations.json",
        conditions: Optional[Sequence[str]] = None,
    ):
        with open(observations_path, "r") as f:
            all_observations = json.load(f)

        # Restrict to a chosen subset, e.g. ["VPC", "LBBB", "RBBB"] for your
        # combined-pipeline overlap set -- or leave None for all 86 conditions.
        missing = [c for c in (conditions or []) if c not in all_observations]
        if missing:
            raise KeyError(f"Condition(s) not found in observations.json: {missing}")
        self.conditions: List[str] = list(conditions) if conditions else list(all_observations.keys())

        potential_labels = [
            [
                [m.lower() for m in all_observations[c]["P"]],
                [m.lower() for m in all_observations[c]["N"]],
            ]
            for c in self.conditions
        ]

        # Keep the original (lowercased) sentence text alongside its future
        # embedding, in the SAME order `extract_language_features` will
        # process it in ([P_list, N_list] per condition) -- this is what
        # lets `classify_with_evidence` report which exact sentence a score
        # belongs to, not just an aggregate number.
        self.observation_texts: Dict[str, Tuple[List[str], List[str]]] = {
            c: (labels[0], labels[1]) for c, labels in zip(self.conditions, potential_labels)
        }

        print("Loading ZETA encoders (checkpoints/best.pt)...")
        (
            self.model, self.ecg_encoder, self.language_encoder,
            self.unimodal_ecg_pooler, self.unimodal_language_pooler,
            self.multi_modal_ecg_proj, self.multi_modal_language_proj,
            self.class_embedding,
        ) = load_encoders()

        print(f"Embedding observation bank for {len(self.conditions)} condition(s)...")
        # {condition: [[P_feature_vec, ...], [N_feature_vec, ...]]}
        self.language_features: Dict[str, list] = extract_language_features(
            self.language_encoder,
            potential_labels,
            self.unimodal_language_pooler,
            self.multi_modal_language_proj,
            all_labels=self.conditions,
        )

    def _encode_ecg(self, zeta_ecg: np.ndarray) -> torch.Tensor:
        """
        Runs the exact ECG-encoding steps `extract_ecg_features` performs
        (same objects, same order: get_embeddings -> prepend CLS ->
        get_output -> multi_modal_ecg_proj -> unimodal_ecg_pooler), but
        takes channel-first (12, T) input directly with NO permute --
        avoiding that function's dataset-name-conditional permute, which
        is only correct when paired with a specific upstream DataLoader
        permute this single-ECG path doesn't have.
        """
        ecg_batch = torch.tensor(zeta_ecg, dtype=torch.float32).unsqueeze(0).cuda()  # (1, 12, T)

        with torch.no_grad():
            uni_modal_ecg_feats, ecg_padding_mask = self.ecg_encoder.get_embeddings(
                ecg_batch, padding_mask=None
            )
            cls_emb = self.class_embedding.repeat((len(uni_modal_ecg_feats), 1, 1))
            uni_modal_ecg_feats = torch.cat([cls_emb, uni_modal_ecg_feats], dim=1)
            uni_modal_ecg_feats = self.ecg_encoder.get_output(uni_modal_ecg_feats, ecg_padding_mask)
            out = self.multi_modal_ecg_proj(uni_modal_ecg_feats)
            ecg_feature = self.unimodal_ecg_pooler(out)

        return F.normalize(ecg_feature.squeeze(0), dim=0)  # (hidden_dim,)

    def classify(self, zeta_ecg: np.ndarray) -> Dict[str, float]:
        """
        zeta_ecg: a single ECG in ZETA format, shape (12, T), as produced by
        `ECGPreprocessor.to_zeta_input(...)`.

        Returns {condition_name: possibility_score in [0, 1]} for every
        condition this classifier was constructed with. Higher = more
        evidence the condition is present.
        """
        ecg_feature = self._encode_ecg(zeta_ecg)

        scores: Dict[str, float] = {}
        for condition in self.conditions:
            p_feats, n_feats = self.language_features[condition]
            sims_p = [
                (ecg_feature @ F.normalize(torch.tensor(v).cuda(), dim=0).reshape(1, -1).T)[0]
                for v in p_feats
            ]
            sims_n = [
                (ecg_feature @ F.normalize(torch.tensor(v).cuda(), dim=0).reshape(1, -1).T)[0]
                for v in n_feats
            ]
            feature_p, _ = get_diseases_probs([sims_p, sims_n])
            scores[condition] = float(torch.mean(torch.tensor(feature_p)).item())

        return scores

    def predict(self, zeta_ecg: np.ndarray, threshold: float = 0.5) -> List[str]:
        """Convenience wrapper: condition names whose score exceeds `threshold`."""
        return [c for c, s in self.classify(zeta_ecg).items() if s > threshold]

    def classify_with_evidence(self, zeta_ecg: np.ndarray) -> Dict[str, ConditionEvidence]:
        """
        Same computation as `classify()`, but keeps every individual
        observation's score instead of collapsing straight to a mean --
        i.e. it does not discard the per-observation detail that
        `get_diseases_probs` already computes internally.

        This is the method to use for the anyECG-chat handoff: rather than
        just knowing "PVC scored 0.61 overall", you get the SPECIFIC
        sentence that scored highest (e.g. "broad qrs complex morphology,
        irregular rhythm because of pvc") to pass as the "why" alongside
        anyECG-chat's localization "where".

        Returns {condition_name: ConditionEvidence}.
        """
        ecg_feature = self._encode_ecg(zeta_ecg)

        evidence: Dict[str, ConditionEvidence] = {}
        for condition in self.conditions:
            p_feats, n_feats = self.language_features[condition]
            p_texts, n_texts = self.observation_texts[condition]

            sims_p = [
                (ecg_feature @ F.normalize(torch.tensor(v).cuda(), dim=0).reshape(1, -1).T)[0]
                for v in p_feats
            ]
            sims_n = [
                (ecg_feature @ F.normalize(torch.tensor(v).cuda(), dim=0).reshape(1, -1).T)[0]
                for v in n_feats
            ]
            # get_diseases_probs pairs sims_p[i] with sims_n[i] positionally
            # and returns one softmax score per pair -- exactly the
            # per-observation detail we want to keep here rather than
            # immediately averaging.
            feature_p_list, feature_n_list = get_diseases_probs([sims_p, sims_n])

            pos_scores = sorted(
                (ObservationScore(text=t, score=float(s), index=i)
                 for i, (t, s) in enumerate(zip(p_texts, feature_p_list))),
                key=lambda o: -o.score,
            )
            neg_scores = sorted(
                (ObservationScore(text=t, score=float(s), index=i)
                 for i, (t, s) in enumerate(zip(n_texts, feature_n_list))),
                key=lambda o: -o.score,
            )

            evidence[condition] = ConditionEvidence(
                condition=condition,
                score=float(torch.mean(torch.tensor(feature_p_list)).item()),
                positive_observations=pos_scores,
                negative_observations=neg_scores,
            )

        return evidence

    def top_finding(self, zeta_ecg: np.ndarray) -> ConditionEvidence:
        """
        Across every condition this classifier was constructed with, returns
        the ConditionEvidence for whichever one has the highest aggregate
        score -- i.e. ZETA's single best-guess diagnosis for this ECG, with
        full observation-level evidence attached. `.top_finding(ecg).condition`
        and `.top_finding(ecg).top_observation.text` are the two strings you
        need to construct the anyECG-chat localization prompt.
        """
        evidence = self.classify_with_evidence(zeta_ecg)
        return max(evidence.values(), key=lambda e: e.score)

    def score_observation_at_index(
        self, zeta_ecg: np.ndarray, condition: str, index: int
    ) -> ObservationScore:
        """
        Re-scores ONE specific positive observation (identified by its
        `index` from a previous `ObservationScore`, not by text -- see the
        note on `ObservationScore.index` about duplicate sentences) against
        a NEW ecg array, using the same positive-vs-negative pairing
        `get_diseases_probs` uses internally (index i's positive sentence
        is paired against index i's negative sentence).

        This is the method the crop/control-window consistency check uses:
        it asks "does THIS cropped region still support the SAME specific
        claim ZETA originally made on the full ECG?" rather than
        re-running full classification on the crop, which would answer a
        different question (whether the crop independently re-discovers
        the condition from scratch).
        """
        ecg_feature = self._encode_ecg(zeta_ecg)
        p_feats, n_feats = self.language_features[condition]
        p_texts, _ = self.observation_texts[condition]

        sim_p = (ecg_feature @ F.normalize(torch.tensor(p_feats[index]).cuda(), dim=0).reshape(1, -1).T)[0]
        sim_n = (ecg_feature @ F.normalize(torch.tensor(n_feats[index]).cuda(), dim=0).reshape(1, -1).T)[0]

        feature_p_list, _ = get_diseases_probs([[sim_p], [sim_n]])
        return ObservationScore(text=p_texts[index], score=float(feature_p_list[0]), index=index)


if __name__ == "__main__":
    from ecg_preprocessing import ECGPreprocessor

    pre = ECGPreprocessor()
    zeta_ecg = pre.to_zeta_input("path/to/some/record")  # no file extension, per wfdb convention

    clf = ZetaZeroShotClassifier(conditions=["VPC", "LBBB", "RBBB", "NORM"])

    scores = clf.classify(zeta_ecg)
    for condition, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"{condition}: {score:.3f}")

    print("\n--- with evidence ---")
    best = clf.top_finding(zeta_ecg)
    print(f"Top condition: {best.condition} (score {best.score:.3f})")
    print(f"Top observation ('why'): \"{best.top_observation.text}\" "
          f"(index {best.top_observation.index}, score {best.top_observation.score:.3f})")
    print("All positive observations, ranked:")
    for obs in best.positive_observations:
        print(f"  {obs.score:.3f}  {obs.text}")