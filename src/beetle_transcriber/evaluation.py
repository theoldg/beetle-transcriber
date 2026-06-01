from collections import defaultdict
from dataclasses import dataclass
from typing import Self

import numpy as np
import scipy.optimize

from beetle_transcriber.midi import Note


@dataclass
class Match:
    true_note: Note
    detected_note: Note


@dataclass
class MatchingResult:
    matches: list[Match]
    false_positives: list[Note]
    false_negatives: list[Note]

    @classmethod
    def sum(cls, results: list[Self]) -> Self:
        return cls(
            matches=[m for r in results for m in r.matches],
            false_positives=[n for r in results for n in r.false_positives],
            false_negatives=[n for r in results for n in r.false_negatives],
        )

    def apply_tolerance(self, tolerance_ms: float) -> Self:
        new_matches = []
        new_fp = self.false_positives.copy()
        new_fn = self.false_negatives.copy()
        for match in self.matches:
            time_diff = match.true_note.start_time - match.detected_note.start_time
            if np.abs(time_diff) * 1000 < tolerance_ms:
                new_matches.append(match)
            else:
                new_fn.append(match.true_note)
                new_fp.append(match.detected_note)
        return MatchingResult(new_matches, new_fp, new_fn)

    @property
    def recall(self):
        denom = len(self.matches) + len(self.false_negatives)
        if denom == 0:
            return float('nan')
        return len(self.matches) / denom

    @property
    def precision(self):
        denom = len(self.matches) + len(self.false_positives)
        if denom == 0:
            return float('nan')
        return len(self.matches) / denom

    @property
    def f1_score(self):
        denom = self.precision + self.recall
        if denom == 0:
            return 0
        return 2 * self.precision * self.recall / denom


def _hungarian_match_single_freq(
    true_notes: list[Note],
    detected_notes: list[Note],
) -> MatchingResult:
    true_times = np.array([note.start_time for note in true_notes])
    detected_times = np.array([note.start_time for note in detected_notes])
    # Shape: (n_detected, n_true).
    cost_matrix = np.abs(true_times[np.newaxis] - detected_times[:, np.newaxis])
    detected_indices = set(range(len(detected_notes)))
    true_indices = set(range(len(true_notes)))
    result = MatchingResult([], [], [])
    row_indices, col_indices = scipy.optimize.linear_sum_assignment(cost_matrix)
    for detected_i, true_i in zip(row_indices, col_indices):
        true_indices.remove(int(true_i))
        detected_indices.remove(int(detected_i))
        result.matches.append(
            Match(
                true_note=true_notes[true_i],
                detected_note=detected_notes[detected_i],
            )
        )
    for i in true_indices:
        result.false_negatives.append(true_notes[i])
    for i in detected_indices:
        result.false_positives.append(detected_notes[i])
    return result


def match_notes(
    true_notes: list[Note],
    detected_notes: list[Note],
) -> MatchingResult:
    true_by_freq = defaultdict(list)
    detected_by_freq = defaultdict(list)
    for note in true_notes:
        true_by_freq[note.note].append(note)
    for note in detected_notes:
        detected_by_freq[note.note].append(note)
    all_freqs = set(true_by_freq) | set(detected_by_freq)
    results = [
        _hungarian_match_single_freq(
            true_notes=true_by_freq.get(f, []),
            detected_notes=detected_by_freq.get(f, []),
        )
        for f in all_freqs
    ]
    return MatchingResult.sum(results)
