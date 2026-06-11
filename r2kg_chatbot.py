import os, sys, json, ast, argparse
from dataclasses import dataclass, field, asdict

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

import utils
from entity_linking import choose_start_entity as choose_start_entity_with_ranking
try:
    from model import LLMBot
except ImportError:
    LLMBot = None
from agent_prompts import QUESTION_PARSING_PROMPT, EXPLORER_PROMPT, CRITIC_PROMPT
from hybrid_heuristic_ranker import rank_candidates_with_hybrid_heuristic

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
load_dotenv()
if openai is not None:
    openai.api_key = os.getenv('OPENAI_KEY')
client = OpenAI(api_key=os.getenv('OPENAI_KEY')) if OpenAI is not None and os.getenv('OPENAI_KEY') else None
LEVELS = {'high', 'medium', 'low'}
GAPS = {'relation', 'entity_type', 'time', 'attribute', 'answer_role', 'none'}

class OpenAIBot:
    def __init__(self, engine, client, temp, top_p):
        self.engine, self.client, self.temperature, self.top_p = engine, client, temp, top_p
        self.conversation = [{"role": "system", "content": "You are a helpful assistant."}]
    def generate_response(self, prompt):
        self.conversation.append({"role": "user", "content": prompt})
        try:
            r = self.client.chat.completions.create(model=self.engine, messages=self.conversation, temperature=self.temperature, top_p=self.top_p)
            c = r.choices[0].message.content.strip()
            self.conversation.append({"role": "assistant", "content": c})
            return c
        except openai.APIError:
            return ""

@dataclass
class ConstraintItem:
    slot: str
    op: str
    value: str
    weight: float = 1.0
    source: str = 'unknown'
    age: int = 0
    success_count: int = 0
    failure_count: int = 0

@dataclass
class PathState:
    entities: list = field(default_factory=list)
    triples: list = field(default_factory=list)

    @property
    def depth(self):
        return len(self.triples)

@dataclass
class MemoryState:
    visited_entities: list = field(default_factory=list)
    banned_branches: list = field(default_factory=list)
    failed_direction_summaries: list = field(default_factory=list)
    branch_diagnostics: list = field(default_factory=list)
    invalid_relations: list = field(default_factory=list)

@dataclass
class State:
    question: str
    start_entity: str
    path: PathState
    hard_constraints: list
    soft_constraints: list
    memory: MemoryState
    budget: int
    max_depth: int
    parsing: dict = field(default_factory=dict)
    candidate_entities: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    consecutive_stalls: int = 0
    explored_steps: int = 0
    evaluated_candidates: int = 0

@dataclass
class GammaOutput:
    hard_add: list = field(default_factory=list)
    soft_add: list = field(default_factory=list)
    hard_remove: list = field(default_factory=list)
    soft_remove: list = field(default_factory=list)
    hard_replace: list = field(default_factory=list)
    notes: list = field(default_factory=list)

def build_bot(operator):
    if operator[0] in {'gpt-4o-mini', 'gpt-4o'}:
        if client is None or openai is None:
            raise RuntimeError('OpenAI runtime is unavailable. Please install dependencies and set OPENAI_KEY.')
        if operator[0] == 'gpt-4o-mini':
            return OpenAIBot('gpt-4o-mini-2024-07-18', client, operator[1], operator[2])
        return OpenAIBot('gpt-4o-2024-08-06', client, operator[1], operator[2])
    if LLMBot is None:
        raise RuntimeError('LLMBot backend is unavailable because model dependencies could not be imported.')
    return LLMBot(operator[0], operator[1], operator[2], 2000)

def jblock(text):
    text = (text or '').strip()
    if text.startswith('```'):
        for p in [x.strip() for x in text.split('```') if x.strip()]:
            if p.startswith('json'): p = p[4:].strip()
            if p[:1] in ['{', '[']: text = p; break
    s = [i for i in [text.find('{'), text.find('[')] if i != -1]
    if not s: raise ValueError('no json')
    payload = text[min(s):max(text.rfind('}'), text.rfind(']')) + 1]
    try: return json.loads(payload)
    except Exception: return ast.literal_eval(payload)

def parse_question_output(raw):
    try: d = jblock(raw)
    except Exception: d = {}
    return {'main_entity': d.get('main_entity', '') if isinstance(d, dict) else '', 'auxiliary_entities': d.get('auxiliary_entities', []) if isinstance(d, dict) else [], 'relation_hints': d.get('relation_hints', []) if isinstance(d, dict) else [], 'constraints': d.get('constraints', []) if isinstance(d, dict) else []}

def parse_explorer_output(raw):
    try: d = jblock(raw)
    except Exception: return []
    return [x for x in d if isinstance(x, dict) and {'action_id', 'relation', 'target_entity', 'rationale'}.issubset(x.keys())] if isinstance(d, list) else []

def parse_critic_output(raw):
    base = {'overall_diagnosis': 'low', 'diagnostic_signals': {'direction_consistency': 'low', 'incremental_completeness': 'low', 'constraint_matching': 'low', 'conflict_status': 'low'}, 'main_gap_type': 'none', 'gap_description': ''}
    try: d = jblock(raw)
    except Exception: return base
    if not isinstance(d, dict): return base
    if d.get('overall_diagnosis') in LEVELS: base['overall_diagnosis'] = d['overall_diagnosis']
    sig = d.get('diagnostic_signals', {}) if isinstance(d.get('diagnostic_signals'), dict) else {}
    for k in base['diagnostic_signals']:
        if sig.get(k) in LEVELS: base['diagnostic_signals'][k] = sig[k]
    if d.get('main_gap_type') in GAPS: base['main_gap_type'] = d['main_gap_type']
    base['gap_description'] = d.get('gap_description', '')
    return base

def deduplicate_constraints(items):
    dedup = []
    seen = set()
    for item in items:
        key = (item.slot, item.op, str(item.value).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup


def _infer_constraint_slot(text):
    text_lower = str(text).lower()
    if any(k in text_lower for k in ['before', 'after', 'during', 'year', 'date', 'time']):
        return 'time'
    if any(k in text_lower for k in ['wife', 'husband', 'mother', 'father', 'child', 'son', 'daughter', 'president', 'director', 'author']):
        return 'answer_role'
    if any(k in text_lower for k in ['type', 'kind of', 'category', 'class of']):
        return 'entity_type'
    if any(k in text_lower for k in ['population', 'language', 'color', 'height', 'birth', 'genre']):
        return 'attribute'
    return 'constraint'


def choose_start_entity(parsed, entities):
    main = parsed.get('main_entity', '')
    cands = [entities[0]] if isinstance(entities, tuple) else [x[0] if isinstance(x, tuple) else x for x in (entities or [])]
    if main and cands:
        m = utils.match_and_replace_single(main, cands)
        if m in cands:
            return m
    return entities[0] if isinstance(entities, tuple) else (cands[0] if cands else main)


def build_initial_constraints(parsed):
    hard = [ConstraintItem('auxiliary_entity', 'must', str(x), 1.0, 'question_parse') for x in parsed.get('auxiliary_entities', [])]
    for c in parsed.get('constraints', []):
        slot = _infer_constraint_slot(c)
        hard.append(ConstraintItem(slot, 'must', str(c), 1.0, 'question_parse'))
    soft = [ConstraintItem('relation', 'prefer', str(x), 0.8, 'question_parse') for x in parsed.get('relation_hints', [])]
    for c in parsed.get('constraints', []):
        slot = _infer_constraint_slot(c)
        if slot in {'time', 'attribute', 'answer_role', 'entity_type'}:
            soft.append(ConstraintItem(slot, 'prefer', str(c), 0.75, 'question_parse'))
    return deduplicate_constraints(hard), deduplicate_constraints(soft)

class GraphAdapter:
    def __init__(self, dataset, info=None, KG=None): self.dataset, self.info, self.KG = dataset, info, KG or {}
    def get_relations(self, entity):
        if self.dataset == 'CRONQUESTIONS': return list({g[1] for g in self.KG.get(entity, [])})
        if self.dataset == 'FactKG':
            import FactKG.dbpedia_sparql as db
            return list(dict.fromkeys(db.getRelationsFromEntity(entity) + db.getRelationsFromEntity('"' + entity + '"')))
        if self.dataset in {'WebQSP', 'CWQ'}:
            db = __import__(f'{self.dataset}.freebase_sparql', fromlist=['x'])
            mid = getattr(self.info, 'mid_dict', {}).get(entity)
            if mid is None: return []
            rels = []
            for rel in db.getRelationsFromEntity(mid):
                k = rel.split('/')[-1]; rels.append(k)
                if hasattr(self.info, 'rel_dict') and k not in self.info.rel_dict: self.info.rel_dict[k] = rel
            return list(dict.fromkeys(rels))
        if self.dataset == 'MetaQA':
            import MetaQA.movie_sparql as db
            return db.getRelationsFromEntity(utils.preprocess_ent(entity), noInverse=True)
        return []
    def expand(self, entity, relation):
        relation = utils.retrieval_relation_parse_answer(relation)
        if self.dataset == 'CRONQUESTIONS': return [g for g in self.KG.get(entity, []) if g[1] == relation]
        if self.dataset == 'FactKG':
            import FactKG.dbpedia_sparql as db
            ent = entity if len(db.getRelationsFromEntity(entity)) >= len(db.getRelationsFromEntity('"' + entity + '"')) else '"' + entity + '"'
            tails = db.getEntityFromEntRel(ent, relation)
            tails += db.getEntityFromEntRel(ent, relation[1:] if relation.startswith('~') else '~' + relation)
            return [[ent, relation, t] for t in tails]
        if self.dataset in {'WebQSP', 'CWQ'}:
            db = __import__(f'{self.dataset}.freebase_sparql', fromlist=['x'])
            mid = getattr(self.info, 'mid_dict', {}).get(entity)
            if mid is None: return []
            out = []
            for tail in db.getEntityFromEntRel(mid, relation):
                name = db.mid2name(tail)
                if hasattr(self.info, 'mid_dict'): self.info.mid_dict[name] = tail
                out.append([entity, relation, name])
            return out
        if self.dataset == 'MetaQA':
            import MetaQA.movie_sparql as db
            src = utils.preprocess_ent(entity)
            tails = db.getEntityFromEntRel(src, relation)
            tails += db.getEntityFromEntRel(src, relation[1:] if relation.startswith('~') else '~' + relation)
            return [[src, relation, t] for t in tails]
        return []
    def get_actions(self, entity):
        out, seen = [], set()
        for rel in self.get_relations(entity):
            for tri in self.expand(entity, rel):
                k = tuple(tri[:3])
                if k in seen: continue
                seen.add(k)
                out.append({'action_id': f'a_{len(out)}', 'source_entity': entity, 'relation': tri[1], 'target_entity': tri[2], 'triple': tri})
        return out

def _constraint_hit(value, relation, target):
    value = str(value).lower().strip()
    relation = str(relation).lower()
    target = str(target).lower()
    return bool(value) and (value in relation or value in target or relation in value or target in value)


def _soft_constraint_match_score(action, constraint):
    rel = str(action['relation']).lower()
    tar = str(action['target_entity']).lower()
    value = str(constraint.value).lower()
    base = 1.0 if _constraint_hit(value, rel, tar) else 0.0
    if constraint.slot == 'time' and any(k in rel for k in ['time', 'date', 'year']):
        base = max(base, 0.8)
    elif constraint.slot == 'attribute' and any(k in rel for k in ['type', 'name', 'language', 'genre', 'population', 'birth']):
        base = max(base, 0.7)
    elif constraint.slot == 'answer_role' and any(k in rel for k in ['role', 'position', 'office', 'spouse', 'parent', 'child']):
        base = max(base, 0.7)
    elif constraint.slot == 'relation' and _constraint_hit(value, rel, tar):
        base = max(base, 0.9)
    return base


def _hard_constraint_violated(action, state):
    rel = str(action['relation']).lower()
    tar = str(action['target_entity']).lower()
    for c in state.hard_constraints:
        value = str(c.value).lower()
        if c.slot == 'entity_type' and not _constraint_hit(value, rel, tar):
            return True, f'entity_type:{c.value}'
        if c.slot == 'time' and not any(k in rel for k in ['time', 'date', 'year']):
            return True, f'time:{c.value}'
        if c.slot == 'attribute' and not any(k in rel for k in ['name', 'type', 'genre', 'language', 'population', 'birth']):
            return True, f'attribute:{c.value}'
        if c.slot == 'answer_role' and not any(k in rel for k in ['role', 'position', 'spouse', 'parent', 'child', 'president', 'director', 'author']):
            return True, f'answer_role:{c.value}'
        if c.slot == 'auxiliary_entity' and state.path.depth > 0 and not _constraint_hit(value, rel, tar):
            return True, f'auxiliary_entity:{c.value}'
        if c.slot == 'constraint' and not _constraint_hit(value, rel, tar):
            return True, f'constraint:{c.value}'
    return False, ''


def construct_action_space(adapter, state, entity):
    neighborhood = adapter.get_actions(entity)
    banned = {(x.get('source_entity'), x.get('relation'), x.get('target_entity')) for x in state.memory.banned_branches}
    feasible, rejected = [], []
    for action in neighborhood:
        key = (action['source_entity'], action['relation'], action['target_entity'])
        if action['target_entity'] in state.memory.visited_entities:
            rejected.append({'action': action, 'reason': 'visited'})
            continue
        if key in banned:
            rejected.append({'action': action, 'reason': 'banned'})
            continue
        violated, reason = _hard_constraint_violated(action, state)
        if violated:
            rejected.append({'action': action, 'reason': reason})
            continue
        action['soft_match_score'] = round(sum(_soft_constraint_match_score(action, c) * c.weight for c in state.soft_constraints if c.op == 'prefer'), 4)
        feasible.append(action)
    feasible.sort(key=lambda x: x.get('soft_match_score', 0.0), reverse=True)
    return neighborhood, feasible[:16], rejected

def branch_reaches_answer(adapter, state, action, gold_answers, max_depth):
    if not gold_answers:
        return False
    targets = {str(x).lower() for x in gold_answers}
    start = action.get('target_entity')
    if str(start).lower() in targets:
        return True
    queue = [(start, 0)]
    seen = {start}
    while queue:
        entity, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        for nxt in adapter.get_actions(entity):
            target = nxt.get('target_entity')
            if str(target).lower() in targets:
                return True
            if target not in seen:
                seen.add(target)
                queue.append((target, depth + 1))
    return False

def path_dict(path): return {'entities': path.entities, 'triples': path.triples, 'tail_entity': path.entities[-1] if path.entities else None, 'depth': len(path.triples)}
def cdict(state): return {'hard_constraints': [asdict(x) for x in state.hard_constraints], 'soft_constraints': [asdict(x) for x in state.soft_constraints]}

def run_explorer(bot, state, actions):
    raw = bot.generate_response(EXPLORER_PROMPT.format(Q=state.question, P_t=json.dumps(path_dict(state.path), ensure_ascii=False, indent=2), C_t=json.dumps(cdict(state), ensure_ascii=False, indent=2), M_t=json.dumps(asdict(state.memory), ensure_ascii=False, indent=2), A_t=json.dumps(actions, ensure_ascii=False, indent=2)))
    amap = {a['action_id']: a for a in actions}; out = []
    for x in parse_explorer_output(raw):
        a = amap.get(x['action_id'])
        if a and x['relation'] == a['relation'] and x['target_entity'] == a['target_entity']: out.append(a)
    seen, uniq = set(), []
    for a in out:
        k = (a['action_id'], a['relation'], a['target_entity'])
        if k not in seen: seen.add(k); uniq.append(a)
    return uniq[:8]

def run_critic(bot, state, action):
    cand = {'entities': state.path.entities + [action['target_entity']], 'triples': state.path.triples + [action['triple']], 'tail_entity': action['target_entity'], 'depth': len(state.path.triples) + 1}
    raw = bot.generate_response(CRITIC_PROMPT.format(Q=state.question, P_t=json.dumps(path_dict(state.path), ensure_ascii=False, indent=2), candidate_path=json.dumps(cand, ensure_ascii=False, indent=2), C_t=json.dumps(cdict(state), ensure_ascii=False, indent=2), M_t=json.dumps(asdict(state.memory), ensure_ascii=False, indent=2)))
    return parse_critic_output(raw)

def score_soft_constraints(action, state):
    score = 0.0
    rel = str(action['relation']).lower()
    tar = str(action['target_entity']).lower()
    for c in state.soft_constraints:
        value = str(c.value).lower()
        hit = value and (value in rel or value in tar or rel in value or tar in value)
        if c.op == 'prefer' and hit:
            score += c.weight
        elif c.op == 'avoid' and hit:
            score -= c.weight
    return score


def score_critic_signals(critic):
    rank = {'high': 2.0, 'medium': 1.0, 'low': 0.0}
    signals = critic.get('diagnostic_signals', {})
    return (
        0.35 * rank.get(signals.get('direction_consistency', 'low'), 0.0)
        + 0.25 * rank.get(signals.get('incremental_completeness', 'low'), 0.0)
        + 0.25 * rank.get(signals.get('constraint_matching', 'low'), 0.0)
        + 0.15 * rank.get(signals.get('conflict_status', 'low'), 0.0)
    )


def best_candidate(items, state, initial_budget=None):
    method = state.settings.get('c4_ranker', 'hybrid')
    if state.settings.get('ablation') != 'base_rank' and method != 'base':
        ranked = rank_candidates_with_hybrid_heuristic(
            items,
            state,
            initial_budget=initial_budget,
            ablation=state.settings.get('ablation', 'none'),
            weights=state.settings.get('c4_weights'),
            method=method,
        )
        return ranked[0] if ranked else None

    rank = {'high': 2, 'medium': 1, 'low': 0}
    def score(x):
        return (
            rank.get(x['critic']['overall_diagnosis'], 0),
            score_critic_signals(x['critic']),
            score_soft_constraints(x['action'], state),
            x['action'].get('soft_match_score', 0.0),
        )
    items.sort(key=score, reverse=True)
    return items[0] if items else None

def gamma_constraints(question, path, gap_analysis, current_constraints):
    del current_constraints
    question_text = str(question).lower()
    depth = path.depth if hasattr(path, 'depth') else len(getattr(path, 'triples', []))
    gap = gap_analysis['main_gap_type']
    desc = str(gap_analysis.get('gap_description', '')).strip()
    overall = gap_analysis.get('overall_diagnosis', 'low')
    signals = gap_analysis.get('diagnostic_signals', {})
    out = GammaOutput()
    if gap == 'none' and overall == 'high':
        return out
    if gap == 'relation' and desc:
        out.soft_add.append(ConstraintItem('relation', 'prefer', desc, 0.9, 'gamma'))
    elif gap == 'entity_type' and desc:
        target = 'must' if signals.get('constraint_matching') == 'low' else 'prefer'
        item = ConstraintItem('entity_type', target, desc, 0.9, 'gamma')
        if target == 'must':
            out.hard_add.append(item)
        else:
            out.soft_add.append(item)
    elif gap == 'time':
        value = desc or 'time constraint'
        out.hard_add.append(ConstraintItem('time', 'must', value, 0.85, 'gamma'))
        out.soft_add.append(ConstraintItem('relation', 'prefer', 'time', 0.7, 'gamma'))
    elif gap == 'attribute':
        value = desc or 'attribute match'
        out.soft_add.append(ConstraintItem('attribute', 'prefer', value, 0.85, 'gamma'))
    elif gap == 'answer_role':
        value = desc or 'answer role'
        out.hard_add.append(ConstraintItem('answer_role', 'must', value, 0.8, 'gamma'))
    if 'when' in question_text and gap != 'time':
        out.soft_add.append(ConstraintItem('time', 'prefer', 'temporal answer', 0.75, 'gamma'))
    if depth >= 2 and signals.get('incremental_completeness') == 'low':
        out.soft_add.append(ConstraintItem('relation', 'avoid', desc or 'unproductive branch', 0.7, 'gamma'))
    if overall == 'low' or signals.get('conflict_status') == 'low':
        out.soft_add.append(ConstraintItem('branch', 'avoid', desc or 'conflicting branch', 0.95, 'gamma'))
        out.notes.append('ban_direction')
    if signals.get('constraint_matching') == 'low':
        out.notes.append('tighten_constraints')
    if 'insufficient' in desc.lower() or 'too restrictive' in desc.lower():
        out.notes.append('relax_hard_constraint')
    if signals.get('direction_consistency') == 'low':
        out.notes.append('penalize_relation')
    return out


def _constraint_key(item):
    return (item.slot, item.op, str(item.value).strip().lower())


def _remove_constraints(constraints, items_to_remove):
    remove_keys = {_constraint_key(item) for item in items_to_remove}
    return [item for item in constraints if _constraint_key(item) not in remove_keys]


def apply_constraint_deltas(state, gamma_output):
    state.hard_constraints = _remove_constraints(state.hard_constraints, gamma_output.hard_remove)
    state.soft_constraints = _remove_constraints(state.soft_constraints, gamma_output.soft_remove)
    if gamma_output.hard_replace:
        for old_item, new_item in gamma_output.hard_replace:
            state.hard_constraints = _remove_constraints(state.hard_constraints, [old_item])
            state.hard_constraints.append(new_item)
    state.hard_constraints.extend(gamma_output.hard_add)
    state.soft_constraints.extend(gamma_output.soft_add)
    state.hard_constraints = deduplicate_constraints(state.hard_constraints)
    state.soft_constraints = deduplicate_constraints(state.soft_constraints)


def update_constraint_histories(state, success=False):
    for item in state.hard_constraints + state.soft_constraints:
        item.age += 1
        if success:
            item.success_count += 1
        else:
            item.failure_count += 1


def decay_soft_constraints(state):
    updated = []
    for item in state.soft_constraints:
        decay = 0.08 + 0.05 * min(item.failure_count, 3)
        new_weight = round(item.weight - decay, 2)
        if item.op == 'avoid':
            new_weight = round(max(item.weight - 0.05, 0.35), 2)
        if new_weight >= 0.3:
            updated.append(ConstraintItem(item.slot, item.op, item.value, new_weight, item.source, item.age, item.success_count, item.failure_count))
    state.soft_constraints = deduplicate_constraints(updated)


def release_conflicting_hard_constraints(state):
    kept = []
    for item in state.hard_constraints:
        if item.failure_count >= 2 and item.success_count == 0:
            continue
        kept.append(item)
    state.hard_constraints = deduplicate_constraints(kept)


def relax_constraints_from_gamma(state, gamma_output):
    if 'relax_hard_constraint' in gamma_output.notes:
        release_conflicting_hard_constraints(state)


def record_branch_feedback(state, action, critic):
    state.memory.branch_diagnostics.append({'source_entity': action['source_entity'], 'relation': action['relation'], 'target_entity': action['target_entity'], 'critic': critic})
    if critic.get('diagnostic_signals', {}).get('direction_consistency') == 'low':
        state.memory.invalid_relations.append(action['relation'])


def decide_control_action(state, action, critic):
    del state, action
    signals = critic.get('diagnostic_signals', {})
    direction = signals.get('direction_consistency', 'low')
    completeness = signals.get('incremental_completeness', 'low')
    matching = signals.get('constraint_matching', 'low')
    conflict = signals.get('conflict_status', 'low')
    if conflict == 'low':
        return 'reject_conflict'
    if direction == 'low':
        return 'reject_direction'
    if matching == 'low' and critic.get('main_gap_type') != 'none':
        return 'repair_constraints'
    if completeness == 'low' and direction in {'high', 'medium'} and conflict in {'high', 'medium'}:
        return 'accept_but_expand'
    if critic.get('overall_diagnosis') == 'low':
        return 'reject_low'
    return 'accept'


def finalize_prediction(state):
    ans = []
    for tri in state.path.triples[-3:]:
        if tri[2] != state.start_entity and str(tri[2]) not in ans:
            ans.append(str(tri[2]))
    return ans or None


def should_stop(state, iter_limit):
    if state.budget <= 0:
        return True
    if state.explored_steps >= min(iter_limit, state.max_depth):
        return True
    if state.consecutive_stalls >= 4:
        return True
    return False


def write_c4_ranking_log(state, step, ranked_items):
    path = state.settings.get('c4_log_path')
    if not path:
        return
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    record = {
        'question': state.question,
        'step': step,
        'tail_entity': state.path.entities[-1] if state.path.entities else None,
        'remaining_depth': max(0, state.max_depth - step - 1),
        'ranker': state.settings.get('c4_ranker', 'hybrid'),
        'ablation': state.settings.get('ablation', 'none'),
        'ranked_candidates': ranked_items,
    }
    with open(path, 'a', encoding='utf-8') as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + '\n')

def build_runtime_report():
    return {
        'openai_installed': openai is not None,
        'dotenv_installed': callable(load_dotenv),
        'openai_key_set': bool(os.getenv('OPENAI_KEY')),
    }


def reasoning(operator, supervisor, claim, iter_limit, initial_prompt, sub_prompt, entities, f, KG=None, info=None, dataset=None, budget=40, ablation='none', c4_ranker='hybrid', c4_weights=None, c4_log_path=None, c5_search=False, c5_max_iterations=20, c5_k0=4, c5_k_min=1, c5_k_max=8, c5_sim_depth=3, c5_strategy='c5', gold_answers=None, c5_ablation='none'):
    del supervisor, initial_prompt, sub_prompt
    max_depth = min(iter_limit, 8)
    initial_budget = budget
    bot = build_bot(operator)
    adapter = GraphAdapter(dataset, info=info, KG=KG)
    parsed = parse_question_output(bot.generate_response(QUESTION_PARSING_PROMPT.format(Q=claim)))
    start, ranked_entities = choose_start_entity_with_ranking(claim, parsed, entities, info=info, KG=KG, adapter=adapter, top_k=5)
    hard, soft = build_initial_constraints(parsed)
    state = State(claim, start, PathState([start], []), hard, soft, MemoryState([start], [], [], [], []), initial_budget, max_depth, parsed, ranked_entities, {'ablation': ablation, 'c4_ranker': c4_ranker, 'c4_weights': c4_weights, 'c4_log_path': c4_log_path, 'c5_search': c5_search, 'c5_ablation': c5_ablation, 'c5_strategy': c5_strategy})
    f.write(f"Runtime report: {json.dumps(build_runtime_report(), ensure_ascii=False)}\n")
    f.write(f"\nQuestion parsing: {json.dumps(parsed, ensure_ascii=False)}\nStart entity: {start}\nRanked start candidates: {json.dumps(ranked_entities, ensure_ascii=False)}\n")
    f.write(f"Control config: max_depth={state.max_depth}, budget={state.budget}\n")
    if c5_search:
        try:
            from adaptive_graph_search import AdaptiveGraphSearchController, C5SearchConfig
            c5_cfg = C5SearchConfig(max_iterations=c5_max_iterations, max_depth=max_depth, k0=c5_k0, k_min=c5_k_min, k_max=c5_k_max, sim_depth=c5_sim_depth, strategy=c5_strategy)
            if c5_ablation == 'no_dynamic_width':
                c5_cfg.ablate_dynamic_width = True
            elif c5_ablation == 'no_dynamic_depth':
                c5_cfg.ablate_dynamic_depth = True
            elif c5_ablation == 'no_failure_rescore':
                c5_cfg.ablate_failure_rescore = True
            elif c5_ablation == 'no_buffer_reactivation':
                c5_cfg.ablate_buffer_reactivation = True
            callbacks = {
                'construct_action_space': construct_action_space,
                'run_explorer': run_explorer,
                'run_critic': run_critic,
                'rank_candidates_with_hybrid_heuristic': rank_candidates_with_hybrid_heuristic,
                'gamma_constraints': gamma_constraints,
                'cdict': cdict,
                'apply_constraint_deltas': apply_constraint_deltas,
                'update_constraint_histories': update_constraint_histories,
                'relax_constraints_from_gamma': relax_constraints_from_gamma,
                'decay_soft_constraints': decay_soft_constraints,
                'release_conflicting_hard_constraints': release_conflicting_hard_constraints,
                'record_branch_feedback': record_branch_feedback,
                'decide_control_action': decide_control_action,
                'finalize_prediction': finalize_prediction,
            }
            if gold_answers:
                callbacks['branch_is_correct'] = lambda s, a: branch_reaches_answer(adapter, s, a, gold_answers, max(0, max_depth - s.path.depth - 1))
            controller = AdaptiveGraphSearchController(bot, adapter, state, initial_budget, callbacks, c5_cfg, f)
            return controller.search()
        except Exception as e:
            f.write(f"C5 fallback to original reasoning due to error: {e}\n")

    for step in range(state.max_depth):
        tail = state.path.entities[-1] if state.path.entities else None
        if should_stop(state, iter_limit) or tail is None:
            break
        state.explored_steps += 1
        raw_actions, feasible_actions, rejected_actions = construct_action_space(adapter, state, tail)
        f.write(f"\n******************** Step:{step} ********************\nTail entity: {tail}\nNeighborhood size: {len(raw_actions)}\nFeasible action space size: {len(feasible_actions)}\nRejected actions: {len(rejected_actions)}\n")
        if not feasible_actions:
            state.memory.failed_direction_summaries.append(f'No legal actions from {tail}')
            state.consecutive_stalls += 1
            update_constraint_histories(state, success=False)
            decay_soft_constraints(state)
            release_conflicting_hard_constraints(state)
            break
        actions = run_explorer(bot, state, feasible_actions)
        f.write(f"Explorer selected {len(actions)} candidates\n")
        if not actions:
            state.memory.failed_direction_summaries.append(f'Explorer returned no valid candidates from {tail}')
            state.consecutive_stalls += 1
            update_constraint_histories(state, success=False)
            decay_soft_constraints(state)
            release_conflicting_hard_constraints(state)
            break
        scored = []
        for action in actions:
            if state.settings.get('ablation') == 'no_critic':
                critic = {'overall_diagnosis': 'high', 'diagnostic_signals': {'direction_consistency': 'high', 'incremental_completeness': 'medium', 'constraint_matching': 'medium', 'conflict_status': 'high'}, 'main_gap_type': 'none', 'gap_description': ''}
            else:
                critic = run_critic(bot, state, action)
            state.evaluated_candidates += 1
            scored.append({'action': action, 'critic': critic})
            f.write(f"Candidate: {json.dumps(action, ensure_ascii=False)}\nCritic: {json.dumps(critic, ensure_ascii=False)}\n")
            if state.budget - state.evaluated_candidates <= 0:
                break
        ranked_scored = rank_candidates_with_hybrid_heuristic(
            scored,
            state,
            initial_budget=initial_budget,
            ablation=state.settings.get('ablation', 'none'),
            weights=state.settings.get('c4_weights'),
            method='base' if state.settings.get('ablation') == 'base_rank' else state.settings.get('c4_ranker', 'hybrid'),
        )
        write_c4_ranking_log(state, step, ranked_scored)
        best = ranked_scored[0] if ranked_scored else None
        if not best:
            state.consecutive_stalls += 1
            update_constraint_histories(state, success=False)
            decay_soft_constraints(state)
            break
        action, critic = best['action'], best['critic']
        if 'hybrid_score' in best:
            f.write(f"Hybrid score: {best['hybrid_score']} components={json.dumps(best.get('hybrid_components', {}), ensure_ascii=False)}\n")
        record_branch_feedback(state, action, critic)
        gamma_output = GammaOutput() if state.settings.get('ablation') == 'no_gamma' else gamma_constraints(state.question, state.path, critic, cdict(state))
        decision = decide_control_action(state, action, critic)
        f.write(f"Control decision: {decision}\n")
        if decision in {'reject_conflict', 'reject_direction', 'reject_low'}:
            state.memory.banned_branches.append({'source_entity': action['source_entity'], 'relation': action['relation'], 'target_entity': action['target_entity']})
            state.memory.failed_direction_summaries.append(critic.get('gap_description', decision))
            if state.settings.get('ablation') != 'no_gamma':
                apply_constraint_deltas(state, gamma_output)
            state.consecutive_stalls += 1
            update_constraint_histories(state, success=False)
            if state.settings.get('ablation') != 'no_adaptive':
                relax_constraints_from_gamma(state, gamma_output)
                decay_soft_constraints(state)
                release_conflicting_hard_constraints(state)
            state.budget -= 1
            continue
        if decision == 'repair_constraints':
            if state.settings.get('ablation') != 'no_gamma':
                apply_constraint_deltas(state, gamma_output)
            update_constraint_histories(state, success=False)
            if state.settings.get('ablation') != 'no_adaptive':
                relax_constraints_from_gamma(state, gamma_output)
            state.consecutive_stalls += 1
            state.budget -= 1
            continue
        state.path.entities.append(action['target_entity'])
        state.path.triples.append(action['triple'])
        if action['target_entity'] not in state.memory.visited_entities:
            state.memory.visited_entities.append(action['target_entity'])
        if state.settings.get('ablation') != 'no_gamma':
            apply_constraint_deltas(state, gamma_output)
        update_constraint_histories(state, success=True)
        if decision == 'accept_but_expand':
            decay_soft_constraints(state)
        state.consecutive_stalls = 0
        state.budget -= 1
        if critic['main_gap_type'] == 'none' and critic['diagnostic_signals'].get('conflict_status') != 'low':
            pred = finalize_prediction(state)
            if pred is not None:
                f.write(f"Final prediction: {pred}\n")
                return pred, step
    pred = finalize_prediction(state) or 'Abstain'
    f.write(f"Fallback prediction: {pred}\n")
    return pred, state.explored_steps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='WebQSP'); parser.add_argument('--operator', type=str, default='gpt-4o-mini')
    parser.add_argument('--supervisor', type=str, default='gpt-4o'); parser.add_argument('--prompt', type=str, default='pr_1')
    parser.add_argument('--iter_limit', type=int, default=15); parser.add_argument('--budget', type=int, default=40)
    parser.add_argument('--temperature', type=float, default=0.95)
    parser.add_argument('--top_p', type=float, default=0.95); parser.add_argument('--ablation', type=str, default='none')
    parser.add_argument('--c4_ranker', type=str, default='hybrid', choices=['hybrid', 'base', 'bm25', 'dense', 'sentence_bert', 'tog', 'transe'])
    parser.add_argument('--c4_weights', type=str, default='')
    parser.add_argument('--c4_log_path', type=str, default='')
    parser.add_argument('--c5_search', action='store_true')
    parser.add_argument('--c5_max_iterations', type=int, default=20)
    parser.add_argument('--c5_k0', type=int, default=4)
    parser.add_argument('--c5_k_min', type=int, default=1)
    parser.add_argument('--c5_k_max', type=int, default=8)
    parser.add_argument('--c5_sim_depth', type=int, default=3)
    parser.add_argument('--c5_strategy', type=str, default='c5', choices=['c5', 'greedy', 'beam', 'random_rollout'])
    parser.add_argument('--c5_ablation', type=str, default='none', choices=['none', 'no_dynamic_width', 'no_dynamic_depth', 'no_failure_rescore', 'no_buffer_reactivation'])
    parser.add_argument('-paraphrase', action='store_true'); parser.add_argument('-single_agent', action='store_true')
    args = parser.parse_args()
    if args.dataset == 'CRONQUESTIONS': import CRONQ.utils as cronq; cronq.main(args)
    elif args.dataset == 'FactKG': import FactKG.utils as factkg; factkg.main(args)
    elif args.dataset == 'WebQSP': import WebQSP.utils as webqsp; webqsp.main(args)
    elif args.dataset == 'CWQ': import CWQ.utils as cwq; cwq.main(args)
    elif args.dataset == 'MetaQA': import MetaQA.utils as metaqa; metaqa.main(args)
