import argparse
import json
import re
import time
from collections import Counter
from fractions import Fraction

from tqdm import tqdm

try:
    from rapidfuzz.fuzz import ratio as similarity_ratio
except ImportError:
    try:
        from fuzzywuzzy.fuzz import ratio as similarity_ratio
    except ImportError:
        from difflib import SequenceMatcher

        def similarity_ratio(first, second):
            """Return a percentage similarity using only the standard library."""
            return 100 * SequenceMatcher(None, first, second).ratio()

REQUIRED_INDEX_FIELDS = ('arg_count', 'return_type', 'loc', 'full_loc')
EXCLUDED_RAG_LOC_MIN = 1
DEFAULT_EXCLUDED_RAG_LOC_MAX = 1
MAX_LOC_RELATIVE_DIFFERENCE = 0.3
MIN_SIMILARITY_SCORE = 90
IGNORED_CODE_TEXT_RE = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\r\n]*',
    flags=re.DOTALL,
)
NUMERIC_LITERAL_RE = re.compile(
    r'''(?<![A-Za-z0-9_$?@.])
        (?:
            0x[0-9a-f]+
            | [0-9][0-9a-f]*h
            | 0b[01]+
            | 0[0-7]+
            | [0-9]+
        )
        (?:u(?:ll?)?|ll?u?|u?i(?:8|16|32|64))?
        (?![A-Za-z0-9_$?@.])''',
    flags=re.IGNORECASE | re.VERBOSE,
)
NUMERIC_SUFFIX_RE = re.compile(
    r'(?:u(?:ll?)?|ll?u?|u?i(?:8|16|32|64))$', flags=re.IGNORECASE
)


def get_existing_index(item):
    """Read the current index; return None when required fields are missing."""
    index = item.get('index')
    if not isinstance(index, dict):
        return None
    if any(field not in index for field in REQUIRED_INDEX_FIELDS):
        return None
    return index


def is_rag_target_eligible(index, excluded_loc_max=DEFAULT_EXCLUDED_RAG_LOC_MAX):
    """Disable RAG when LOC is within the inclusive excluded range."""
    loc = index['loc']
    return not (
        isinstance(loc, (int, float))
        and EXCLUDED_RAG_LOC_MIN <= loc <= excluded_loc_max
    )


def extract_numeric_constants(code):
    """Extract normalized numeric constants, ignoring identifiers and text."""
    if not isinstance(code, str):
        return None

    code_without_ignored_text = IGNORED_CODE_TEXT_RE.sub('', code)
    constants = set()
    for match in NUMERIC_LITERAL_RE.finditer(code_without_ignored_text):
        literal = NUMERIC_SUFFIX_RE.sub('', match.group(0))
        if literal.lower().startswith('0x'):
            base = 16
        elif literal.lower().startswith('0b'):
            base = 2
        elif literal.lower().endswith('h'):
            literal = literal[:-1]
            base = 16
        elif len(literal) > 1 and literal.startswith('0'):
            base = 8
        else:
            base = 10
        constants.add(int(literal, base))
    return frozenset(constants)


def numeric_constant_jaccard(constants_a, constants_b):
    """Compute the Jaccard similarity of two numeric-constant sets."""
    if constants_a is None or constants_b is None:
        return None
    union = constants_a | constants_b
    if not union:
        return None
    return Fraction(len(constants_a & constants_b), len(union))


def relative_difference_within_limit(value_a, value_b):
    """Return whether two non-negative size features are within the limit."""
    if not isinstance(value_a, (int, float)) or not isinstance(
        value_b, (int, float)
    ):
        return False
    denominator = max(value_a, value_b, 1)
    return abs(value_a - value_b) / denominator <= MAX_LOC_RELATIVE_DIFFERENCE


def passes_size_filters(index_a, index_b):
    """Require both body LOC and auxiliary full LOC to be similar."""
    return relative_difference_within_limit(
        index_a['loc'], index_b['loc']
    ) and relative_difference_within_limit(
        index_a['full_loc'], index_b['full_loc']
    )


def score_candidate(item_a, item_b, index_b):
    """Return code similarity after size filtering, or None when filtered out."""
    index_a = get_existing_index(item_a)
    if index_a is None or not passes_size_filters(index_a, index_b):
        return None

    norm_code_a = item_a.get('rag_norm_code')
    norm_code_b = item_b.get('rag_norm_code')
    if not isinstance(norm_code_a, str) or not isinstance(norm_code_b, str):
        return None
    return similarity_ratio(norm_code_a, norm_code_b)


def build_index_for_data_a(data_a):
    """Index candidates with reliable source by (arg_count, return_type)."""
    candidate_index = {}
    for item in data_a:
        if not item.get('unstripped_code'):
            continue
        features = get_existing_index(item)
        if features is None:
            continue
        key = (features['arg_count'], features['return_type'])
        candidate_index.setdefault(key, []).append(item)
    return candidate_index


def retrieval_diagnostics(
    status,
    *,
    best_score=None,
    second_best_score=None,
    qualified_candidate_count=0,
    size_filtered_candidate_count=0,
    top_similarity_candidate_count=0,
    selection_method=None,
    best_numeric_constant_jaccard=None,
    second_best_numeric_constant_jaccard=None,
    top_jaccard_candidate_count=0,
):
    """Build retrieval diagnostics to store in each output record."""
    return {
        'status': status,
        'best_score': best_score,
        'second_best_score': second_best_score,
        'qualified_candidate_count': qualified_candidate_count,
        'size_filtered_candidate_count': size_filtered_candidate_count,
        'top_similarity_candidate_count': top_similarity_candidate_count,
        'selection_method': selection_method,
        'best_numeric_constant_jaccard': best_numeric_constant_jaccard,
        'second_best_numeric_constant_jaccard': second_best_numeric_constant_jaccard,
        'top_jaccard_candidate_count': top_jaccard_candidate_count,
    }


def resolve_similarity_tie_with_numeric_constants(item_b, top_candidates):
    """Break similarity ties using numeric-constant Jaccard similarity."""
    target_constants = extract_numeric_constants(item_b.get('code'))
    unavailable_result = {
        'selected_candidate': None,
        'best_numeric_constant_jaccard': None,
        'second_best_numeric_constant_jaccard': None,
        'top_jaccard_candidate_count': len(top_candidates),
    }
    # Without target constants there is no evidence for tie breaking.
    if not target_constants:
        return unavailable_result

    scored_candidates = []
    for candidate in top_candidates:
        candidate_constants = extract_numeric_constants(candidate.get('code'))
        if candidate_constants is None:
            return unavailable_result
        score = numeric_constant_jaccard(target_constants, candidate_constants)
        # The target set is non-empty, so a union and score must exist here.
        if score is None:
            return unavailable_result
        scored_candidates.append((score, candidate))

    scored_candidates.sort(key=lambda pair: pair[0], reverse=True)
    best_jaccard = scored_candidates[0][0]
    second_best_jaccard = (
        scored_candidates[1][0] if len(scored_candidates) > 1 else None
    )
    jaccard_winners = [
        candidate
        for score, candidate in scored_candidates
        if score == best_jaccard
    ]
    selected_candidate = jaccard_winners[0] if len(jaccard_winners) == 1 else None
    return {
        'selected_candidate': selected_candidate,
        'best_numeric_constant_jaccard': float(best_jaccard),
        'second_best_numeric_constant_jaccard': (
            float(second_best_jaccard)
            if second_best_jaccard is not None
            else None
        ),
        'top_jaccard_candidate_count': len(jaccard_winners),
    }


def select_best_candidate(item_b, candidates, progress=None):
    """Select the global top candidate; reject unresolved top-score ties."""
    index_b = get_existing_index(item_b)
    best_score = None
    second_best_score = None
    top_candidates = []
    qualified_candidate_count = 0
    size_filtered_candidate_count = 0

    for item_a in candidates:
        score = score_candidate(item_a, item_b, index_b)
        if progress is not None:
            progress.update(1)
        if score is None:
            continue

        size_filtered_candidate_count += 1
        if score > MIN_SIMILARITY_SCORE:
            qualified_candidate_count += 1

        if best_score is None or score > best_score:
            second_best_score = best_score
            best_score = score
            top_candidates = [item_a]
        elif score == best_score:
            second_best_score = best_score
            top_candidates.append(item_a)
        elif second_best_score is None or score > second_best_score:
            second_best_score = score

    diagnostic_kwargs = {
        'best_score': best_score,
        'second_best_score': second_best_score,
        'qualified_candidate_count': qualified_candidate_count,
        'size_filtered_candidate_count': size_filtered_candidate_count,
        'top_similarity_candidate_count': len(top_candidates),
    }
    if best_score is None or best_score <= MIN_SIMILARITY_SCORE:
        return None, retrieval_diagnostics('no_match', **diagnostic_kwargs)

    if len(top_candidates) == 1:
        return top_candidates[0], retrieval_diagnostics(
            'matched', selection_method='similarity', **diagnostic_kwargs
        )

    # Original candidate labels are unavailable at inference time. For ties,
    # use only observable numeric-constant Jaccard similarity.
    jaccard_result = resolve_similarity_tie_with_numeric_constants(
        item_b, top_candidates
    )
    jaccard_diagnostics = {
        key: value
        for key, value in jaccard_result.items()
        if key != 'selected_candidate'
    }
    selected_candidate = jaccard_result['selected_candidate']
    if selected_candidate is not None:
        return selected_candidate, retrieval_diagnostics(
            'matched',
            selection_method='numeric_constant_jaccard',
            **diagnostic_kwargs,
            **jaccard_diagnostics,
        )

    if len(top_candidates) > 1:
        unresolved_method = (
            'numeric_constant_jaccard_unavailable'
            if jaccard_result['best_numeric_constant_jaccard'] is None
            else 'unresolved_numeric_constant_jaccard'
        )
        return None, retrieval_diagnostics(
            'ambiguous',
            selection_method=unresolved_method,
            **diagnostic_kwargs,
            **jaccard_diagnostics,
        )

    raise AssertionError('Unexpected empty top-candidate set after threshold check.')


def attach_rag(item_b, item_a):
    """Attach a candidate that passed ranking and ambiguity checks."""
    item_b['rag'] = {
        'unstripped_code': item_a['unstripped_code'],
        'proj': item_a.get('proj'),
        'bin': item_a.get('bin'),
        'addr': item_a.get('addr'),
    }


def compare_code_files(
    file_a_path,
    file_b_path,
    output_path,
    excluded_loc_max=DEFAULT_EXCLUDED_RAG_LOC_MAX,
):
    with open(file_a_path, 'r', encoding='utf8') as file_a:
        data_a = json.load(file_a)
    with open(file_b_path, 'r', encoding='utf8') as file_b:
        data_b = json.load(file_b)

    if data_a and not any(get_existing_index(item) is not None for item in data_a):
        raise ValueError(
            'RAG candidate data lacks the complete index (full_loc is required); '
            'regenerate the index with test_data/quchong/suoyin.py first.'
        )
    if data_b and not any(get_existing_index(item) is not None for item in data_b):
        raise ValueError(
            'Test data lacks the complete index (full_loc is required); '
            'regenerate the index with test_data/quchong/suoyin.py first.'
        )

    print('[INFO] Building index for file A...')
    index_a = build_index_for_data_a(data_a)
    print(f'[INFO] Index built. It contains {len(index_a)} unique signatures.')

    total_tasks = 0
    for item_b in data_b:
        features = get_existing_index(item_b)
        if features is None or not is_rag_target_eligible(
            features, excluded_loc_max
        ):
            continue
        key = (features['arg_count'], features['return_type'])
        total_tasks += len(index_a.get(key, []))
    print(f'[INFO] Estimated total comparisons after indexing: {total_tasks}')

    status_counts = Counter()
    with tqdm(total=total_tasks, desc='Processing Comparisons', ncols=100) as pbar:
        for item_b in data_b:
            # Input may contain previous RAG results. Clear them first so only
            # reliable matches from this run are retained.
            item_b.pop('rag', None)
            features = get_existing_index(item_b)
            if features is None:
                item_b['rag_retrieval'] = retrieval_diagnostics('missing_index')
                status_counts['missing_index'] += 1
                continue
            if not is_rag_target_eligible(features, excluded_loc_max):
                item_b['rag_retrieval'] = retrieval_diagnostics('short_loc')
                status_counts['short_loc'] += 1
                continue

            key = (features['arg_count'], features['return_type'])
            candidates = index_a.get(key, [])
            selected_candidate, diagnostics = select_best_candidate(
                item_b, candidates, progress=pbar
            )

            item_b['rag_retrieval'] = diagnostics
            status_counts[diagnostics['status']] += 1
            if selected_candidate is not None:
                attach_rag(item_b, selected_candidate)

    print(f'[DEBUG] Final length of output data: {len(data_b)}')
    for status in ('matched', 'ambiguous', 'no_match', 'short_loc', 'missing_index'):
        print(f'[INFO] {status}: {status_counts[status]}')

    with open(output_path, 'w', encoding='utf8') as file_out:
        json.dump(data_b, file_out, indent=4, ensure_ascii=False)
    print(f'[+] Reliable RAG retrieval complete; saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'rag_base',
        help='RAG candidate JSON file (for example ./data/train_candidates.json)',
    )
    parser.add_argument(
        'test',
        help='Test JSON file (for example ./data/test_rag.json)',
    )
    parser.add_argument(
        'output_path',
        help='Output JSON file (for example ./data/test_rag_output.json)',
    )
    parser.add_argument(
        '--exclude-loc-max',
        type=int,
        default=DEFAULT_EXCLUDED_RAG_LOC_MAX,
        help=(
            'Maximum body LOC excluded from RAG; 1 skips only LOC=1, '
            '5 skips LOC=1..5, and 0 disables LOC exclusion (default: 1).'
        ),
    )
    args = parser.parse_args()
    if args.exclude_loc_max < 0:
        parser.error('--exclude-loc-max must be at least 0.')

    start_time = time.perf_counter()
    try:
        compare_code_files(
            args.rag_base,
            args.test,
            args.output_path,
            excluded_loc_max=args.exclude_loc_max,
        )
    finally:
        elapsed_seconds = time.perf_counter() - start_time
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            f'[INFO] Total script runtime: '
            f'{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}'
        )
