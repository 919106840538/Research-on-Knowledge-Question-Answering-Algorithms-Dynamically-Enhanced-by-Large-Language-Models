import math
import re
from collections import Counter
from difflib import SequenceMatcher


def _tokenize(text):
    if text is None:
        return []
    return [tok for tok in re.split(r'[^a-zA-Z0-9_]+', str(text).lower().replace('_', ' ')) if tok]


def _string_similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def _jaccard(a_tokens, b_tokens):
    a_set, b_set = set(a_tokens), set(b_tokens)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def _cosine_counter(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0.0
    a_counter, b_counter = Counter(a_tokens), Counter(b_tokens)
    shared = set(a_counter) & set(b_counter)
    numerator = sum(a_counter[t] * b_counter[t] for t in shared)
    a_norm = math.sqrt(sum(v * v for v in a_counter.values()))
    b_norm = math.sqrt(sum(v * v for v in b_counter.values()))
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return numerator / (a_norm * b_norm)


def build_candidate_pool(entities, info=None, KG=None):
    pool = []
    if isinstance(entities, tuple):
        pool.append(entities[0])
    elif isinstance(entities, list):
        for item in entities:
            pool.append(item[0] if isinstance(item, tuple) else item)
    if info is not None and hasattr(info, 'mid_dict'):
        pool.extend(list(info.mid_dict.keys()))
    if KG:
        try:
            pool.extend(list(KG.keys()))
        except Exception:
            pass
    dedup, seen = [], set()
    for item in pool:
        key = str(item).strip().lower()
        if key and key not in seen:
            seen.add(key)
            dedup.append(item)
    return dedup


def build_local_context(candidate, adapter=None, max_relations=12):
    if adapter is None:
        return {'relations': [], 'context_text': str(candidate)}
    try:
        relations = adapter.get_relations(candidate)[:max_relations]
    except Exception:
        relations = []
    relation_text = ' '.join(str(r).replace('.', ' ').replace('_', ' ') for r in relations)
    context_text = f"{candidate} {relation_text}".strip()
    return {'relations': relations, 'context_text': context_text}


def score_candidate(question, mention, candidate, relation_hints=None, adapter=None):
    relation_hints = relation_hints or []
    q_tokens = _tokenize(question)
    m_tokens = _tokenize(mention)
    local_context = build_local_context(candidate, adapter=adapter)
    c_tokens = _tokenize(candidate)
    ctx_tokens = _tokenize(local_context['context_text'])

    lexical = _string_similarity(mention, candidate)
    mention_overlap = _jaccard(m_tokens, c_tokens)
    question_context_overlap = _jaccard(q_tokens, ctx_tokens)
    semantic_alignment = _cosine_counter(q_tokens, ctx_tokens)

    relation_score = 0.0
    if relation_hints:
        lowered_rels = [str(r).lower() for r in local_context['relations']]
        for hint in relation_hints:
            hint_lower = str(hint).lower()
            if any(hint_lower in rel or rel in hint_lower for rel in lowered_rels):
                relation_score += 1.0
        relation_score /= max(len(relation_hints), 1)

    score = (
        0.30 * lexical
        + 0.20 * mention_overlap
        + 0.20 * question_context_overlap
        + 0.20 * semantic_alignment
        + 0.10 * relation_score
    )
    return round(score, 6), local_context


def rank_entity_candidates(question, mention, candidate_pool, relation_hints=None, adapter=None, top_k=5):
    scored = []
    for candidate in candidate_pool:
        score, local_context = score_candidate(question, mention, candidate, relation_hints=relation_hints, adapter=adapter)
        scored.append({
            'entity': candidate,
            'score': score,
            'local_context': local_context['context_text'],
            'relations': local_context['relations'],
        })
    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored[:top_k]


def choose_start_entity(question, parsed_question, entities, info=None, KG=None, adapter=None, top_k=5):
    mention = parsed_question.get('main_entity', '') or ''
    relation_hints = parsed_question.get('relation_hints', [])
    candidate_pool = build_candidate_pool(entities, info=info, KG=KG)
    ranked = rank_entity_candidates(question, mention, candidate_pool, relation_hints=relation_hints, adapter=adapter, top_k=top_k)
    if ranked:
        return ranked[0]['entity'], ranked
    fallback = candidate_pool[0] if candidate_pool else mention
    return fallback, [{'entity': fallback, 'score': 0.0, 'local_context': str(fallback), 'relations': []}]
