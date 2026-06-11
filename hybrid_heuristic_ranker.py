import math
import re
from collections import Counter
from functools import lru_cache

LEVEL = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
DEFAULT_WEIGHTS = {'sem': 0.3, 'str': 0.4, 'cov': 0.1, 'conf': 0.1, 'eff': 0.1}
SEM_WEIGHTS = {'llm': 0.35, 'dense': 0.35, 'lex': 0.30}
STR_WEIGHTS = {'type': 0.40, 'direction': 0.35, 'diversity': 0.25}


@lru_cache(maxsize=2)
def load_sentence_bert(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception:
        return None


def parse_weight_string(s):
    if not s or isinstance(s, dict):
        return s
    parts = [x.strip() for x in str(s).replace(';', ',').split(',') if x.strip()]
    if all('=' in x for x in parts):
        return {k.strip(): float(v.strip()) for k, v in (p.split('=', 1) for p in parts)}
    vals = [float(x) for x in parts]
    if len(vals) != 5:
        raise ValueError('c4_weights must contain sem,str,cov,conf,eff')
    return dict(zip(['sem', 'str', 'cov', 'conf', 'eff'], vals))


class HybridHeuristicRanker:
    def __init__(self, weights=None, gamma1=0.35, gamma2=0.35):
        self.weights = self._norm(weights or DEFAULT_WEIGHTS, DEFAULT_WEIGHTS)
        self.sem_weights = self._norm(SEM_WEIGHTS, SEM_WEIGHTS)
        self.str_weights = self._norm(STR_WEIGHTS, STR_WEIGHTS)
        self.gamma1, self.gamma2 = gamma1, gamma2

    def rank(self, items, state, initial_budget=None, ablation='none', method='hybrid'):
        if not items:
            return []
        if method == 'base':
            return self.base_rank(items, state)
        if method in {'bm25', 'dense', 'sentence_bert', 'transe', 'tog'}:
            return self.baseline_rank(items, state, method)
        initial_budget = initial_budget or getattr(state, 'max_depth', 1) or 1
        corpus = [self.action_text(x.get('action', x)) for x in items]
        raw = [self.raw_scores(x, state, initial_budget, corpus) for x in items]
        normed = self.normalize(raw, ['sem', 'str', 'cov', 'conf', 'cost'])
        out = []
        for item, r, n in zip(items, raw, normed):
            comps = {'sem': n['sem'], 'str': n['str'], 'cov': n['cov'], 'conf': n['conf'], 'eff': self.eff(n['cost'], state, initial_budget)}
            comps = self.ablate(comps, ablation)
            score = sum(self.weights[k] * comps[k] for k in DEFAULT_WEIGHTS)
            y = dict(item)
            y.update({'hybrid_score': round(score, 6), 'hybrid_components': self.round_obj(comps), 'hybrid_raw_components': self.round_obj(r), 'ranking_method': method})
            out.append(y)
        out.sort(key=lambda x: (x.get('hybrid_score', 0.0), x.get('action', {}).get('soft_match_score', 0.0), self.confidence(x.get('critic', {}))), reverse=True)
        for i, x in enumerate(out, 1):
            x['rank'] = i
        return out

    def raw_scores(self, item, state, initial_budget, corpus):
        action, critic = item.get('action', item), item.get('critic', {})
        sem, semp = self.semantic(action, state, critic, corpus)
        st, stp = self.structure(action, state, critic)
        cov, covp = self.coverage(critic)
        conf, confp = self.confidence(critic, True)
        cost, costp = self.cost(action, state)
        return {'sem': sem, 'str': st, 'cov': cov, 'conf': conf, 'cost': cost, 'sem_parts': semp, 'str_parts': stp, 'cov_parts': covp, 'conf_parts': confp, 'cost_parts': costp, 'budget_used_ratio': self.used_ratio(state, initial_budget)}

    def baseline_rank(self, items, state, method):
        corpus = [self.action_text(x.get('action', x)) for x in items]
        out = []
        for item in items:
            action, critic = item.get('action', item), item.get('critic', {})
            if method == 'bm25':
                score = self.bm25(getattr(state, 'question', ''), self.action_text(action), corpus)
                parts = {'bm25': score}
            elif method in {'dense', 'sentence_bert'}:
                score, parts = self.sentence_bert(action, state)
            elif method == 'transe':
                score, parts = self.transe(action, state)
            elif method == 'tog':
                score, parts = self.tog_pruning(action, state, critic)
            else:
                score, parts = 0.0, {}
            y = dict(item)
            y.update({'hybrid_score': round(score, 6), 'hybrid_components': self.round_obj(parts), 'hybrid_raw_components': self.round_obj(parts), 'ranking_method': method})
            out.append(y)
        out.sort(key=lambda x: (x.get('hybrid_score', 0.0), x.get('action', {}).get('soft_match_score', 0.0), self.confidence(x.get('critic', {}))), reverse=True)
        for i, x in enumerate(out, 1):
            x['rank'] = i
        return out

    def sentence_bert(self, action, state):
        query, document = getattr(state, 'question', ''), self.action_text(action)
        model_name = getattr(state, 'settings', {}).get('sentence_bert_model', 'sentence-transformers/all-MiniLM-L6-v2')
        model = load_sentence_bert(model_name)
        if model is not None:
            q_emb, d_emb = model.encode([query, document], normalize_embeddings=True)
            cosine = float(sum(float(a) * float(b) for a, b in zip(q_emb, d_emb)))
            score = max(0.0, min(1.0, (cosine + 1.0) / 2.0))
            return score, {'sentence_bert_cosine': score, 'fallback_hash_dense': 0.0, 'model': model_name}
        score = self.dense(query, document)
        return score, {'sentence_bert_cosine': score, 'fallback_hash_dense': 1.0, 'model': model_name}

    def transe(self, action, state):
        h = self.embedding(action, state, 'source_entity')
        r = self.embedding(action, state, 'relation')
        t = self.embedding(action, state, 'target_entity')
        distance = math.sqrt(sum((h[i] + r[i] - t[i]) ** 2 for i in range(len(h))))
        score = 1.0 / (1.0 + distance)
        return score, {'transe_distance': distance, 'transe_score': score}

    def tog_pruning(self, action, state, critic):
        question = getattr(state, 'question', '')
        rel = str(action.get('relation', ''))
        target = str(action.get('target_entity', ''))
        path = self.path_text(getattr(state, 'path', None))
        sig = critic.get('diagnostic_signals', {})
        relation_score = max(self.dense(question, rel), LEVEL.get(sig.get('direction_consistency', 'low'), 0.0))
        entity_score = max(self.dense(question, target), LEVEL.get(sig.get('constraint_matching', 'low'), 0.0))
        path_score = self.dense(question, ' '.join([path, rel, target]))
        score = 0.55 * relation_score + 0.30 * entity_score + 0.15 * path_score
        return score, {'relation_pruning': relation_score, 'entity_pruning': entity_score, 'path_context': path_score}

    def base_rank(self, items, state):
        def key(x):
            c, a = x.get('critic', {}), x.get('action', x)
            s = c.get('diagnostic_signals', {})
            cs = 0.35 * LEVEL.get(s.get('direction_consistency', 'low'), 0) + 0.25 * LEVEL.get(s.get('incremental_completeness', 'low'), 0) + 0.25 * LEVEL.get(s.get('constraint_matching', 'low'), 0) + 0.15 * LEVEL.get(s.get('conflict_status', 'low'), 0)
            return (LEVEL.get(c.get('overall_diagnosis', 'low'), 0), cs, self.constraint_hint(a, getattr(state, 'soft_constraints', [])), a.get('soft_match_score', 0.0))
        out = [dict(x) for x in items]
        out.sort(key=key, reverse=True)
        for i, x in enumerate(out, 1):
            x.update({'rank': i, 'ranking_method': 'base'})
        return out

    def semantic(self, action, state, critic, corpus):
        q, text = getattr(state, 'question', ''), self.action_text(action)
        lex, dense = self.bm25(q, text, corpus), self.sentence_bert(action, state)[0]
        sig = critic.get('diagnostic_signals', {})
        soft = float(action.get('soft_match_score', 0.0)); soft = soft / (1 + abs(soft)) if soft else 0.0
        llm = 0.35 * LEVEL.get(sig.get('direction_consistency', 'low'), 0) + 0.25 * LEVEL.get(sig.get('constraint_matching', 'low'), 0) + 0.25 * soft + 0.15 * self.constraint_hint(action, getattr(state, 'soft_constraints', []))
        return self.sem_weights['llm'] * llm + self.sem_weights['dense'] * dense + self.sem_weights['lex'] * lex, {'llm': llm, 'dense': dense, 'lex': lex}

    def structure(self, action, state, critic):
        type_cons = self.type_consistency(action, state)
        direct = self.direction_consistency(action, state, critic)
        div = self.structural_diversity(action, state)
        return self.str_weights['type'] * type_cons + self.str_weights['direction'] * direct + self.str_weights['diversity'] * div, {'type_consistency': type_cons, 'direction_consistency': direct, 'diversity': div}

    def coverage(self, critic):
        before_set = self.gap_set(critic.get('previous_gaps'))
        after_set = self.gap_set(critic.get('remaining_gaps'))
        if before_set is not None:
            before, after = len(before_set), len(after_set or set())
            gain = max(0.0, (before - after) / (before + 1e-8)) if before > 0 else 1.0
            return gain, {'gap_reduction': gain, 'before_gap_count': before, 'after_gap_count': after}
        gap = critic.get('main_gap_type', 'none')
        before = 0 if gap == 'none' else 1
        after = 0 if gap == 'none' and critic.get('overall_diagnosis') == 'high' else before
        gain = max(0.0, (before - after) / (before + 1e-8)) if before > 0 else 1.0
        return gain, {'gap_reduction': gain, 'before_gap_count': before, 'after_gap_count': after}

    def confidence(self, critic, parts=False):
        sig = critic.get('diagnostic_signals', {})
        exp = critic.get('confidence_score') if isinstance(critic.get('confidence_score'), (int, float)) else None
        exp = max(0, min(1, float(exp))) if exp is not None else None
        reliability = LEVEL.get(critic.get('overall_diagnosis', 'low'), 0)
        self_consistency = LEVEL.get(sig.get('direction_consistency', 'low'), 0)
        factual_consistency = LEVEL.get(sig.get('conflict_status', 'low'), 0)
        constraint_consistency = LEVEL.get(sig.get('constraint_matching', 'low'), 0)
        hallucination_risk = self.hallucination_risk(critic)
        proxy = max(0.0, 0.35 * reliability + 0.25 * self_consistency + 0.30 * factual_consistency + 0.10 * constraint_consistency - hallucination_risk)
        score = 0.5 * exp + 0.5 * proxy if exp is not None else proxy
        detail = {'explicit': exp if exp is not None else -1.0, 'reliability': reliability, 'self_consistency': self_consistency, 'factual_consistency': factual_consistency, 'constraint_consistency': constraint_consistency, 'hallucination_risk': hallucination_risk, 'proxy': proxy}
        return (score, detail) if parts else score

    def cost(self, action, state):
        rel, tar = str(action.get('relation', '')), str(action.get('target_entity', ''))
        visited = getattr(getattr(state, 'memory', None), 'visited_entities', []) or []
        explicit = min(1, (float(action.get('graph_query_cost', 1)) + float(action.get('llm_cost', 1))) / 6)
        text = min(1, (len(tar) + len(rel)) / 120)
        depth = min(1, getattr(getattr(state, 'path', None), 'depth', 0) / max(1, getattr(state, 'max_depth', 1)))
        fanout = 1 if tar in visited else min(1, max(0, len(tar) - 20) / 60)
        return 0.3 * explicit + 0.25 * text + 0.25 * depth + 0.2 * fanout, {'explicit': explicit, 'text': text, 'depth': depth, 'fanout_proxy': fanout}

    def type_consistency(self, action, state):
        rel, tar = str(action.get('relation', '')).lower(), str(action.get('target_entity', '')).lower()
        source_type = self.meta_value(action, ['source_type', 'domain', 'relation_domain'])
        target_type = self.meta_value(action, ['target_type', 'range', 'relation_range'])
        constraints = list(getattr(state, 'hard_constraints', []) or []) + list(getattr(state, 'soft_constraints', []) or [])
        score = 0.65 if (source_type or target_type) else 0.5
        for c in constraints:
            slot, val = getattr(c, 'slot', ''), str(getattr(c, 'value', '')).lower()
            if not val:
                continue
            if slot in {'entity_type', 'answer_role'}:
                if val in tar or val in str(target_type).lower() or str(target_type).lower() in val:
                    score += 0.25 * float(getattr(c, 'weight', 1))
                elif val in rel or val in str(source_type).lower():
                    score += 0.10 * float(getattr(c, 'weight', 1))
                else:
                    score -= 0.10 * float(getattr(c, 'weight', 1))
            elif slot in {'attribute', 'time', 'relation'} and self.constraint_hit_text(val, rel, tar):
                score += 0.12 * float(getattr(c, 'weight', 1))
        return max(0, min(1, score))

    def direction_consistency(self, action, state, critic):
        rel = str(action.get('relation', '')).lower()
        source = str(action.get('source_entity', '')).lower()
        target = str(action.get('target_entity', '')).lower()
        question = str(getattr(state, 'question', '')).lower()
        score = LEVEL.get(critic.get('diagnostic_signals', {}).get('direction_consistency', 'low'), 0)
        if rel.startswith('~') or rel.startswith('inverse_') or rel.endswith('_inverse'):
            score -= 0.15
        if source and target and source == target:
            score -= 0.30
        if any(k in question for k in ['born', 'birth']) and any(k in rel for k in ['birth', 'born']):
            score += 0.15
        if any(k in question for k in ['when', 'year', 'date']) and any(k in rel for k in ['time', 'date', 'year']):
            score += 0.15
        return max(0, min(1, score))

    def structural_diversity(self, action, state):
        rel, target = str(action.get('relation', '')).lower(), action.get('target_entity')
        triples = getattr(getattr(state, 'path', None), 'triples', []) or []
        visited = set(getattr(getattr(state, 'memory', None), 'visited_entities', []) or [])
        previous_relations = [str(t[1]).lower() for t in triples if len(t) >= 2]
        repeated_node = 1.0 if target in visited else 0.0
        repeated_relation = 1.0 if rel in previous_relations else 0.0
        immediate_backtrack = 1.0 if len(triples) >= 2 and target == triples[-2][0] else 0.0
        return max(0, min(1, 0.45 * (1 - repeated_node) + 0.25 * (1 - repeated_relation) + 0.20 * (1 - repeated_node) + 0.10 * (1 - immediate_backtrack)))

    def constraint_hint(self, action, constraints):
        rel, tar = str(action.get('relation', '')).lower(), str(action.get('target_entity', '')).lower()
        total = hit = 0.0
        for c in constraints or []:
            if getattr(c, 'op', 'prefer') != 'prefer':
                continue
            w, v = float(getattr(c, 'weight', 1)), str(getattr(c, 'value', '')).lower()
            total += w
            if v and (v in rel or v in tar or rel in v or tar in v):
                hit += w
        return hit / total if total else 0

    def bm25(self, query, document, corpus=None):
        q, d = self.tokens(query), self.tokens(document)
        if not q or not d:
            return 0
        docs = [self.tokens(x) for x in (corpus or [document])]
        avgdl, tf, score = sum(len(x) for x in docs) / max(1, len(docs)), Counter(d), 0.0
        for tok in q:
            df, freq = sum(1 for doc in docs if tok in doc), tf.get(tok, 0)
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            denom = freq + 1.5 * (0.25 + 0.75 * len(d) / max(avgdl, 1e-8))
            score += idf * (freq * 2.5 / denom) if denom else 0
        return score / (score + 1) if score > 0 else 0

    def dense(self, query, document):
        qv, dv = self.vec(query), self.vec(document)
        num = sum(v * dv.get(k, 0) for k, v in qv.items())
        den = math.sqrt(sum(v * v for v in qv.values())) * math.sqrt(sum(v * v for v in dv.values()))
        return max(0, num / den) if den else 0

    def eff(self, norm_cost, state, initial_budget):
        return max(0, 1 - (self.gamma1 + self.gamma2 * self.used_ratio(state, initial_budget)) * norm_cost)

    def action_text(self, action):
        return ' '.join(str(action.get(k, '')) for k in ['source_entity', 'relation', 'target_entity', 'rationale'])

    def path_text(self, path):
        triples = getattr(path, 'triples', []) or []
        return ' '.join(' '.join(map(str, tri[:3])) for tri in triples if isinstance(tri, (list, tuple)))

    def embedding(self, action, state, key, dims=128):
        direct = action.get(f'{key}_embedding')
        if isinstance(direct, (list, tuple)):
            return [float(x) for x in direct]
        table = getattr(state, 'settings', {}).get('transe_embeddings', {}) if hasattr(state, 'settings') else {}
        label = str(action.get(key, ''))
        if isinstance(table, dict) and isinstance(table.get(label), (list, tuple)):
            return [float(x) for x in table[label]]
        counts = self.vec(label, dims)
        den = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        return [counts.get(i, 0.0) / den for i in range(dims)]

    def gap_set(self, gaps):
        if gaps is None:
            return None
        if isinstance(gaps, dict):
            return {str(k).strip().lower() for k, v in gaps.items() if v}
        if isinstance(gaps, (list, tuple, set)):
            return {str(x).strip().lower() for x in gaps if str(x).strip()}
        text = str(gaps).strip().lower()
        if text in {'', 'none', '[]'}:
            return set()
        return {text}

    def hallucination_risk(self, critic):
        text = str(critic.get('gap_description', '')).lower()
        lexical = 0.25 if any(k in text for k in ['conflict', 'hallucination', 'wrong', 'irrelevant', 'contradict', 'unsupported', 'uncertain']) else 0.0
        conflict = critic.get('diagnostic_signals', {}).get('conflict_status', 'low')
        return max(lexical, {'high': 0.0, 'medium': 0.10, 'low': 0.25}.get(conflict, 0.25))

    def meta_value(self, action, keys):
        for key in keys:
            if action.get(key):
                return action.get(key)
        meta = action.get('meta') if isinstance(action.get('meta'), dict) else {}
        for key in keys:
            if meta.get(key):
                return meta.get(key)
        return ''

    def constraint_hit_text(self, value, relation, target):
        value = str(value).lower().strip()
        relation = str(relation).lower()
        target = str(target).lower()
        return bool(value) and (value in relation or value in target or relation in value or target in value)

    def tokens(self, text):
        return [x for x in re.split(r'[^0-9a-zA-Z_]+', str(text).lower().replace('.', ' ')) if x]

    def vec(self, text, dims=256):
        out = Counter()
        for t in self.tokens(text):
            out[hash(t) % dims] += 1
        return out

    def gap_count(self, gaps, default=0):
        if gaps is None:
            return default
        if isinstance(gaps, (list, tuple, set, dict)):
            return len(gaps)
        return 0 if str(gaps).strip().lower() in {'', 'none', '[]'} else 1

    def used_ratio(self, state, initial_budget):
        initial_budget = max(1, initial_budget or 1)
        return max(0, min(1, (initial_budget - getattr(state, 'budget', initial_budget)) / initial_budget))

    def normalize(self, rows, dims):
        vals = {d: [r[d] for r in rows] for d in dims}
        return [{d: 0.0 if math.isclose(min(vals[d]), max(vals[d])) else (r[d] - min(vals[d])) / (max(vals[d]) - min(vals[d])) for d in dims} for r in rows]

    def _norm(self, weights, defaults):
        vals = {k: max(0, float(weights.get(k, 0))) for k in defaults}
        total = sum(vals.values())
        return dict(defaults) if total <= 0 else {k: v / total for k, v in vals.items()}

    def ablate(self, scores, ablation):
        key = {'no_sem': 'sem', 'no_str': 'str', 'no_cov': 'cov', 'no_conf': 'conf', 'no_eff': 'eff'}.get(ablation)
        if key:
            scores = dict(scores); scores[key] = 0.0
        return scores

    def round_obj(self, obj):
        if isinstance(obj, float):
            return round(obj, 6)
        if isinstance(obj, dict):
            return {k: self.round_obj(v) for k, v in obj.items()}
        return obj


def rank_candidates_with_hybrid_heuristic(items, state, initial_budget=None, ablation='none', weights=None, method='hybrid'):
    return HybridHeuristicRanker(parse_weight_string(weights)).rank(items, state, initial_budget, ablation, method)
