import copy, json, math, random
from dataclasses import dataclass, field

LEVEL = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
DEFAULT_WEIGHTS = {'sem': 0.3, 'str': 0.4, 'cov': 0.1, 'conf': 0.1, 'eff': 0.1}

@dataclass
class C5SearchConfig:
    max_iterations: int = 20
    strategy: str = 'c5'
    beam_width: int = 4
    max_depth: int = 8
    c_uct: float = 1.2
    omega_h: float = 0.6
    k_min: int = 1
    k_max: int = 8
    k0: int = 4
    alpha_uncertainty: float = 2.0
    beta_budget: float = 2.0
    kappa_gap: float = 2.0
    sim_depth: int = 3
    tau_done: float = 0.85
    mu_gap: float = 0.4
    mu_cost: float = 0.2
    dead_penalty: float = -0.3
    depth_window: int = 2
    epsilon_s: float = 0.05
    epsilon_g: float = 1.0
    epsilon_buffer: float = 0.05
    v_ban: float = -0.2
    n_ban: int = 2
    b_crit: int = 3
    ablate_dynamic_width: bool = False
    ablate_dynamic_depth: bool = False
    ablate_failure_rescore: bool = False
    ablate_buffer_reactivation: bool = False

@dataclass
class ActionStats:
    visits: int = 0
    mean_value: float = 0.0
    banned: bool = False

@dataclass
class SearchNode:
    node_id: str
    state: object
    parent_id: str | None = None
    parent_action_key: tuple | None = None
    keep: list = field(default_factory=list)
    buffer: list = field(default_factory=list)
    scored: list = field(default_factory=list)
    children: dict = field(default_factory=dict)
    tried: set = field(default_factory=set)
    visits: int = 0
    expanded: bool = False
    failure: str | None = None
    depth_gain_window: list = field(default_factory=list)

@dataclass
class SearchMetrics:
    expanded_nodes: int = 0
    selected_steps: int = 0
    total_candidates: int = 0
    total_kept: int = 0
    total_buffered: int = 0
    total_depth: int = 0
    rollout_calls: int = 0
    rollout_steps: int = 0
    rollout_reward_sum: float = 0.0
    reward_sum: float = 0.0
    pcr_sum: float = 0.0
    cpr_hits: int = 0
    cpr_total: int = 0
    buffer_reactivations: int = 0
    backtracks: int = 0
    failed_nodes: int = 0
    recovery_success: int = 0
    recovery_failure: int = 0

class AdaptiveGraphSearchController:
    def __init__(self, bot, adapter, state, initial_budget, callbacks, config=None, log_file=None):
        self.bot, self.adapter, self.cb = bot, adapter, callbacks
        self.initial_budget = max(1, initial_budget)
        self.cfg, self.log_file = config or C5SearchConfig(), log_file
        self.nodes, self.stats, self.counter = {}, {}, 0
        self.metrics = SearchMetrics()
        self.best_state, self.best_score = None, float('-inf')
        root = self._nid(); self.root_id = root
        self.nodes[root] = SearchNode(root, copy.deepcopy(state))

    def _nid(self):
        self.counter += 1; return f'c5_n_{self.counter}'
    def _log(self, obj):
        if self.log_file: self.log_file.write(json.dumps({'c5': obj}, ensure_ascii=False) + '\n')
    def _tail(self, s):
        ents = getattr(getattr(s, 'path', None), 'entities', []) or []
        return ents[-1] if ents else None
    def _key(self, a):
        return (str(a.get('source_entity','')), str(a.get('relation','')), str(a.get('target_entity','')))
    def _rank_args(self, s):
        st = getattr(s, 'settings', {})
        method = 'base' if st.get('ablation') == 'base_rank' else st.get('c4_ranker', 'hybrid')
        return st.get('ablation', 'none'), st.get('c4_weights'), method
    def _score_critic(self, c):
        sig = c.get('diagnostic_signals', {}) if isinstance(c, dict) else {}
        return 0.35*LEVEL.get(c.get('overall_diagnosis','low'),0)+0.25*LEVEL.get(sig.get('incremental_completeness','low'),0)+0.25*LEVEL.get(sig.get('constraint_matching','low'),0)+0.15*LEVEL.get(sig.get('conflict_status','low'),0)
    def _depth(self, s):
        return int(getattr(getattr(s, 'path', None), 'depth', 0) or 0)
    def _gap_count(self, c):
        gaps = c.get('remaining_gaps') if isinstance(c, dict) else None
        if isinstance(gaps, (list, tuple, set, dict)): return len(gaps)
        return 0 if isinstance(c, dict) and c.get('main_gap_type', 'none') == 'none' else 1
    def _logic_gaps(self, c):
        gaps = c.get('remaining_gaps') if isinstance(c, dict) else None
        if isinstance(gaps, dict): return {str(k) for k, v in gaps.items() if v}
        if isinstance(gaps, (list, tuple, set)): return {str(x) for x in gaps if str(x).strip()}
        gap = c.get('main_gap_type', 'none') if isinstance(c, dict) else 'none'
        desc = c.get('gap_description', '') if isinstance(c, dict) else ''
        return set() if gap == 'none' else {str(gap) + ':' + str(desc)}
    def _path_score(self, state, critic):
        depth_penalty = 0.02 * self._depth(state)
        return max(0.0, min(1.0, self._score_critic(critic) - depth_penalty))
    def _summary(self):
        pcr = self.metrics.pcr_sum / max(1, self.metrics.expanded_nodes)
        cpr = self.metrics.cpr_hits / self.metrics.cpr_total if self.metrics.cpr_total else None
        avg_depth = self.metrics.total_depth / max(1, self.metrics.selected_steps)
        avg_rollout = self.metrics.rollout_reward_sum / max(1, self.metrics.rollout_calls)
        return {'event':'summary','expanded_nodes':self.metrics.expanded_nodes,'selected_steps':self.metrics.selected_steps,'avg_pcr':round(pcr,6),'avg_cpr':None if cpr is None else round(cpr,6),'avg_depth':round(avg_depth,6),'total_candidates':self.metrics.total_candidates,'total_kept':self.metrics.total_kept,'total_buffered':self.metrics.total_buffered,'rollout_calls':self.metrics.rollout_calls,'rollout_steps':self.metrics.rollout_steps,'avg_rollout_reward':round(avg_rollout,6),'backtracks':self.metrics.backtracks,'buffer_reactivations':self.metrics.buffer_reactivations,'failed_nodes':self.metrics.failed_nodes,'recovery_success':self.metrics.recovery_success,'recovery_failure':self.metrics.recovery_failure,'strategy':self.cfg.strategy,'oracle_cpr_available':self.cb.get('branch_is_correct') is not None}
    def _depth_stall(self, n):
        if self.cfg.ablate_dynamic_depth: return False
        w = n.depth_gain_window[-self.cfg.depth_window:]
        if len(w) < self.cfg.depth_window: return False
        avg_gain = sum(x['score_gain'] for x in w) / len(w)
        gap_gain = sum(x['gap_gain'] for x in w)
        return avg_gain < self.cfg.epsilon_s and gap_gain < self.cfg.epsilon_g
    def _reward(self, item, rollout_reward=0.0):
        c = item.get('critic', {}); comp = item.get('hybrid_components', {})
        gap = 0.2 if c.get('main_gap_type') == 'none' else 0.0
        bad = -0.35 if c.get('diagnostic_signals', {}).get('conflict_status') == 'low' else 0.0
        return 0.55*self._score_critic(c)+0.25*float(item.get('hybrid_score',0) or 0)+0.1*float(comp.get('conf',0) or 0)+0.1*float(comp.get('cov',0) or 0)+gap+bad+rollout_reward
    def _width(self, ranked, s):
        if not ranked: return 0
        if self.cfg.ablate_dynamic_width: return min(self.cfg.k0, len(ranked))
        conf = [float(x.get('hybrid_components',{}).get('conf',0.5) or 0.5) for x in ranked]
        u = sum(1-c for c in conf)/max(1,len(conf))
        d = float(ranked[0].get('hybrid_score',0))-float(ranked[1].get('hybrid_score',0)) if len(ranked)>1 else 0
        used = 1 - float(getattr(s,'budget',self.initial_budget))/self.initial_budget
        k = math.ceil(self.cfg.k0+self.cfg.alpha_uncertainty*u-self.cfg.beta_budget*used-self.cfg.kappa_gap*d)
        return max(self.cfg.k_min, min(self.cfg.k_max, k, len(ranked)))
    def _buffer(self, ranked, k):
        if self.cfg.ablate_buffer_reactivation or k <= 0 or k >= len(ranked): return []
        b = float(ranked[k-1].get('hybrid_score',0))
        return [x for x in ranked[k:] if float(x.get('hybrid_score',0)) >= b-self.cfg.epsilon_buffer]
    def _norm(self, w):
        t = sum(max(0,float(v)) for v in w.values())
        return {k:max(0,float(v))/t for k,v in w.items()} if t else DEFAULT_WEIGHTS
    def _adjust(self, failure, weights):
        w = dict(weights or DEFAULT_WEIGHTS)
        if failure in {'empty','stall'}: w['sem']=w.get('sem',0)+.1; w['cov']=w.get('cov',0)+.1
        elif failure == 'conflict': w['str']=w.get('str',0)+.1; w['conf']=w.get('conf',0)+.1
        elif failure == 'budget': w['eff']=w.get('eff',0)+.15
        return self._norm(w)
    def _expand(self, n):
        s, tail = n.state, self._tail(n.state)
        if tail is None: n.failure = 'empty'; n.expanded = True; self.metrics.failed_nodes += 1; return
        raw, feas, rej = self.cb['construct_action_space'](self.adapter, s, tail)
        if not feas:
            n.failure = 'empty'; n.expanded = True; self.metrics.failed_nodes += 1; self._log({'event':'empty','node':n.node_id,'tail':tail,'rejected':len(rej)}); return
        acts = self.cb['run_explorer'](self.bot, s, feas) or feas[:8]
        scored = [{'action':a, 'critic':self.cb['run_critic'](self.bot, s, a)} for a in acts]
        ab, w, m = self._rank_args(s)
        ranked = self.cb['rank_candidates_with_hybrid_heuristic'](scored, s, self.initial_budget, ab, w, m)
        k = self._width(ranked, s); n.scored, n.keep, n.buffer = ranked, ranked[:k], self._buffer(ranked, k)
        self.metrics.expanded_nodes += 1; self.metrics.total_candidates += len(raw); self.metrics.total_kept += len(n.keep); self.metrics.total_buffered += len(n.buffer)
        if raw: self.metrics.pcr_sum += 1 - len(n.keep) / len(raw)
        n.failure = None if n.keep else 'empty'
        n.expanded = True
        self._log({'event':'expand','node':n.node_id,'tail':tail,'raw':len(raw),'feasible':len(feas),'keep':len(n.keep),'buffer':len(n.buffer)})
    def _rescore(self, n):
        if self.cfg.ablate_failure_rescore or not n.scored: return
        ab, w, m = self._rank_args(n.state); w = self._adjust(n.failure, w); n.state.settings['c4_weights'] = w
        ranked = self.cb['rank_candidates_with_hybrid_heuristic'](n.scored, n.state, self.initial_budget, ab, w, m)
        k = self._width(ranked, n.state); n.scored, n.keep, n.buffer = ranked, ranked[:k], self._buffer(ranked, k)
    def _select(self, n):
        pool = n.keep + ([] if self.cfg.ablate_buffer_reactivation else n.buffer)
        best, val = None, float('-inf')
        for it in pool:
            k = self._key(it['action'])
            if k in n.tried and k not in n.children: continue
            st = self.stats.setdefault((n.node_id,k), ActionStats())
            if st.banned: continue
            u = self.cfg.c_uct*math.sqrt(math.log(max(2,n.visits+1))/(st.visits+1))
            score = st.mean_value + u + self.cfg.omega_h*float(it.get('hybrid_score',0) or 0)
            if score > val: best, val = it, score
        return best
    def _best_buffer_item(self, n):
        if self.cfg.ablate_buffer_reactivation: return None
        cands = [x for x in n.buffer if self._key(x['action']) not in n.tried]
        if not cands: return None
        cands.sort(key=lambda x: float(x.get('hybrid_score', 0) or 0), reverse=True)
        return cands[0]
    def _backup(self, n, key, reward):
        st = self.stats.setdefault((n.node_id,key), ActionStats()); st.visits += 1
        st.mean_value += (reward-st.mean_value)/st.visits
        st.banned = st.mean_value < self.cfg.v_ban and st.visits >= self.cfg.n_ban
        n.visits += 1; self.metrics.reward_sum += reward; self._log({'event':'backup','node':n.node_id,'action':key,'reward':round(reward,6),'value':round(st.mean_value,6),'visits':st.visits})
    def _reject(self, n, a, c, decision):
        s = n.state; g = self.cb['gamma_constraints'](s.question, s.path, c, self.cb['cdict'](s))
        self.cb['apply_constraint_deltas'](s, g)
        if decision != 'repair_constraints': s.memory.banned_branches.append({'source_entity':a['source_entity'],'relation':a['relation'],'target_entity':a['target_entity']})
        self.cb['relax_constraints_from_gamma'](s, g); self.cb['decay_soft_constraints'](s); self.cb['release_conflicting_hard_constraints'](s)
        self.cb['update_constraint_histories'](s, success=False); s.consecutive_stalls += 1; s.budget = max(0, s.budget-1)
    def _child_state(self, s, a, c):
        child = copy.deepcopy(s); child.path.entities.append(a['target_entity']); child.path.triples.append(a['triple'])
        if a['target_entity'] not in child.memory.visited_entities: child.memory.visited_entities.append(a['target_entity'])
        self.cb['record_branch_feedback'](child, a, c)
        g = self.cb['gamma_constraints'](child.question, child.path, c, self.cb['cdict'](child))
        self.cb['apply_constraint_deltas'](child, g); self.cb['update_constraint_histories'](child, success=True)
        child.consecutive_stalls = 0; child.budget = max(0, child.budget-1); return child
    def _rollout(self, state, base_critic):
        local = copy.deepcopy(state); prev_score = self._score_critic(base_critic)
        total, steps, dead = 0.0, 0, False
        for _ in range(self.cfg.sim_depth):
            tail = self._tail(local)
            if tail is None or getattr(local, 'budget', 0) <= 0: dead = True; break
            raw, feas, _ = self.cb['construct_action_space'](self.adapter, local, tail)
            if not feas: dead = True; break
            acts = self.cb['run_explorer'](self.bot, local, feas) or feas[:4]
            scored = [{'action':a, 'critic':self.cb['run_critic'](self.bot, local, a)} for a in acts]
            ab, w, m = self._rank_args(local)
            ranked = self.cb['rank_candidates_with_hybrid_heuristic'](scored, local, self.initial_budget, ab, w, m)
            if not ranked: dead = True; break
            best = ranked[0]; critic = best['critic']
            gain = self._score_critic(critic) - prev_score
            gap_gain = max(0.0, self._gap_count(base_critic) - self._gap_count(critic)) / max(1, self._gap_count(base_critic) + 1)
            total += gain + self.cfg.mu_gap * gap_gain - self.cfg.mu_cost * ((steps + 1) / max(1, self.cfg.sim_depth))
            local = self._child_state(local, best['action'], critic); prev_score = self._score_critic(critic); steps += 1
            if critic.get('main_gap_type') == 'none' and critic.get('diagnostic_signals',{}).get('conflict_status') != 'low': break
        if dead: total += self.cfg.dead_penalty
        self.metrics.rollout_calls += 1; self.metrics.rollout_steps += steps; self.metrics.rollout_reward_sum += total
        return total, steps, dead
    def _backtrack(self, n):
        cur = n
        while cur.parent_id:
            self.metrics.backtracks += 1
            p = self.nodes.get(cur.parent_id)
            if p and (self._select(p) is not None or self._best_buffer_item(p) is not None):
                if self._best_buffer_item(p) is not None:
                    self.metrics.buffer_reactivations += 1
                self.metrics.recovery_success += 1
                self._log({'event':'recovery_success','from':n.node_id,'to':p.node_id})
                return p
            cur = p
        self.metrics.recovery_failure += 1
        self._log({'event':'recovery_failure','from':n.node_id})
        return None
    def _make_child(self, parent, item):
        action, critic = item['action'], item['critic']
        key = self._key(action)
        child_state = self._child_state(parent.state, action, critic)
        cid = self._nid(); child = SearchNode(cid, child_state, parent.node_id, key)
        self.nodes[cid] = child; parent.children[key] = cid
        return child
    def _tree_policy(self):
        node = self.nodes[self.root_id]
        path = []
        while node.expanded and node.keep:
            if self._depth_stall(node): node.failure = 'stall'; self._rescore(node)
            item = self._select(node) or self._best_buffer_item(node)
            if item is None: break
            key = self._key(item['action']); path.append((node, item))
            if key in node.children:
                node = self.nodes[node.children[key]]
            else:
                node.tried.add(key)
                return node, item, path
            if self._depth(node.state) >= self.cfg.max_depth: break
        return node, None, path
    def _run_baseline(self):
        if self.cfg.strategy == 'beam': return self._run_beam()
        cur = self.nodes[self.root_id]
        steps = 0
        while steps < min(self.cfg.max_iterations, self.cfg.max_depth) and getattr(cur.state, 'budget', 0) > 0:
            self._expand(cur)
            pool = cur.scored if cur.scored else cur.keep
            if not pool: break
            item = random.choice(pool) if self.cfg.strategy == 'random_rollout' else pool[0]
            a, c = item['action'], item['critic']; key = self._key(a); cur.tried.add(key)
            reward = self._reward(item, 0.0); self._backup(cur, key, reward)
            if self.cb['decide_control_action'](cur.state, a, c) in {'reject_conflict','reject_direction','reject_low','repair_constraints'}:
                self._reject(cur, a, c, 'reject_low'); steps += 1; continue
            cur = self._make_child(cur, item)
            pred = self.cb['finalize_prediction'](cur.state)
            if pred is not None and c.get('main_gap_type') == 'none': self._log(self._summary()); return pred, steps
            steps += 1
        self._log(self._summary()); return self.cb['finalize_prediction'](cur.state) or 'Abstain', steps
    def _run_beam(self):
        beam = [self.nodes[self.root_id]]; best = None; steps = 0
        while steps < min(self.cfg.max_iterations, self.cfg.max_depth) and beam:
            nxt = []
            for node in beam:
                self._expand(node)
                for item in node.scored[:self.cfg.beam_width]:
                    a, c = item['action'], item['critic']
                    if self.cb['decide_control_action'](node.state, a, c) in {'reject_conflict','reject_direction','reject_low','repair_constraints'}: continue
                    child = self._make_child(node, item); nxt.append((float(item.get('hybrid_score',0) or 0), child, c))
            nxt.sort(key=lambda x: x[0], reverse=True)
            beam = [x[1] for x in nxt[:self.cfg.beam_width]]
            if beam: best = beam[0]
            for _, child, critic in nxt[:self.cfg.beam_width]:
                pred = self.cb['finalize_prediction'](child.state)
                if pred is not None and critic.get('main_gap_type') == 'none': self._log(self._summary()); return pred, steps
            steps += 1
        self._log(self._summary()); return self.cb['finalize_prediction']((best or self.nodes[self.root_id]).state) or 'Abstain', steps
    def search(self):
        if self.cfg.strategy in {'greedy', 'beam', 'random_rollout'}:
            return self._run_baseline()
        steps = 0
        max_steps = min(self.cfg.max_iterations, self.cfg.max_depth)
        current = self.nodes[self.root_id]
        while steps < max_steps:
            leaf, pending_item, path = self._tree_policy()
            current = leaf
            if getattr(leaf.state, 'budget', 0) <= self.cfg.b_crit:
                leaf.failure = 'budget'; self._rescore(leaf); self._log({'event':'budget_failure','node':leaf.node_id,'budget':getattr(leaf.state,'budget',0)})
            if not leaf.expanded:
                self._expand(leaf)
                if leaf.failure: self._rescore(leaf)
            item = pending_item or self._select(leaf) or self._best_buffer_item(leaf)
            if item is None:
                parent = self._backtrack(leaf)
                if parent is None: break
                current = parent; steps += 1; continue
            action, critic = item['action'], item['critic']; key = self._key(action); leaf.tried.add(key)
            decision = self.cb['decide_control_action'](leaf.state, action, critic)
            rollout_reward, _, _ = self._rollout(leaf.state, critic)
            reward = self._reward(item, rollout_reward)
            self.metrics.selected_steps += 1; self.metrics.total_depth += self._depth(leaf.state)
            if item in leaf.keep:
                self.metrics.cpr_total += 1
                if self.cb.get('branch_is_correct') is not None: self.metrics.cpr_hits += 1 if self.cb['branch_is_correct'](leaf.state, action) else 0
                elif reward > 0: self.metrics.cpr_hits += 1
            if decision in {'reject_conflict','reject_direction','reject_low','repair_constraints'}:
                self._reject(leaf, action, critic, decision); self._backup(leaf, key, reward); steps += 1; continue
            child = self._make_child(leaf, item); self._backup(leaf, key, reward)
            score = self._path_score(child.state, critic)
            leaf.depth_gain_window.append({'score_gain': score, 'gap_gain': max(0.0, 1.0 - self._gap_count(critic))})
            if len(leaf.depth_gain_window) > self.cfg.depth_window: leaf.depth_gain_window = leaf.depth_gain_window[-self.cfg.depth_window:]
            if score > self.best_score: self.best_score, self.best_state = score, copy.deepcopy(child.state)
            pred = self.cb['finalize_prediction'](child.state)
            if pred is not None and critic.get('main_gap_type') == 'none' and critic.get('diagnostic_signals',{}).get('conflict_status') != 'low':
                self._log({'event':'final','steps':steps+1,'prediction':pred}); self._log(self._summary()); return pred, steps
            current = child; steps += 1
        final = self.best_state or current.state
        self._log(self._summary())
        return self.cb['finalize_prediction'](final) or 'Abstain', steps
