#!/usr/bin/env python3
"""
Evaluation script for CURRENT ButterflyClusterViz 2 project state.
Parameters taken directly from MainScene.unity.
Outputs JSON results for report generation.
"""
import csv, math, os, sys, json
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "Assets", "StreamingAssets", "Data",
                        "unity_pruned_density_tree_butterfly_3d.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "scripts", "eval_current_results.json")

# CURRENT Unity Inspector values from MainScene.unity
POSITION_SCALE = 5.0
PLANET_SCALE = 2.0
ADAPTIVE_SCALE_SENSITIVITY = 1.0
RADIUS_SCALE = 0.3
MIN_RADIUS = 0.3
DEPTH_RADIUS_FACTOR = 0.65
MIN_EFFECTIVE_RADIUS = 0.1
SIBLING_RADIUS_CAP_FACTOR = 0.6
TOP_LEVEL_MARGIN = 1.2
TOP_LEVEL_SEP_ITERS = 300
IMAGE_QUAD_SIZE = 0.3
MIN_IMAGE_QUAD_SIZE = 0.05

class Node:
    __slots__ = ["nid","pid","planet","size","pos","children","images","depth",
                 "parent","world_pos","sphere_radius","displayed_radius"]
    def __init__(self, nid, pid, planet, size, pos):
        self.nid, self.pid, self.planet = nid, pid, int(planet)
        self.size = int(size); self.pos = np.array(pos, dtype=np.float64)
        self.children, self.images = [], []; self.depth = 0
        self.parent = None; self.world_pos = None
        self.sphere_radius = 0.0; self.displayed_radius = 0.0

class Image:
    __slots__ = ["fname","pid","planet","pos","parent_node","before_pos","world_pos"]
    def __init__(self, fname, pid, planet, pos):
        self.fname, self.pid, self.planet = fname, pid, int(planet)
        self.pos = np.array(pos, dtype=np.float64)
        self.parent_node = None; self.before_pos = None; self.world_pos = None

def parse_csv(path):
    with open(path, newline="") as f: rows = list(csv.DictReader(f))
    nodes = {}
    for r in rows:
        if r["type"] != "node": continue
        nid = r["node_id"]
        if nid not in nodes:
            nodes[nid] = Node(nid, r["parent_id"], r["planet_id"], r["size"],
                              [float(r["x"]), float(r["y"]), float(r["z"])])
    planets = []
    for n in nodes.values():
        if n.pid == "root": planets.append(n)
        elif n.pid in nodes: n.parent = nodes[n.pid]; nodes[n.pid].children.append(n)
    planets.sort(key=lambda p: p.planet)
    def set_depth(nd, d):
        nd.depth = d
        for c in nd.children: set_depth(c, d+1)
    for p in planets: set_depth(p, 0)
    images = []
    for r in rows:
        if r["type"] != "image": continue
        img = Image(r.get("image_id",""), r["parent_id"], r["planet_id"],
                    [float(r["x"]), float(r["y"]), float(r["z"])])
        if img.pid in nodes: img.parent_node = nodes[img.pid]; nodes[img.pid].images.append(img)
        images.append(img)
    return nodes, planets, images

def compute_radius(size):
    return max(math.log2(size + 1) * RADIUS_SCALE, MIN_RADIUS)

def sibling_cap(positions, uncapped):
    n = len(positions); capped = np.array(uncapped, dtype=np.float64)
    if n < 2 or SIBLING_RADIUS_CAP_FACTOR <= 0: return capped
    D = cdist(positions, positions); np.fill_diagonal(D, np.inf)
    avg_nn = float(np.mean(np.min(D, axis=1)))
    mx = SIBLING_RADIUS_CAP_FACTOR * avg_nn / 2.0
    for i in range(n):
        if capped[i] > mx: capped[i] = max(mx, MIN_EFFECTIVE_RADIUS)
    return capped

def resolve_overlaps(positions, radii, margin, max_iters):
    n = len(positions); res = np.array(positions, dtype=np.float64)
    for it in range(max_iters):
        any_ov = False
        for i in range(n):
            for j in range(i+1, n):
                diff = res[j] - res[i]; dist = np.linalg.norm(diff)
                md = radii[i] + radii[j] + margin
                if dist < md:
                    any_ov = True
                    if dist > 1e-4:
                        d = diff/dist; p = (md-dist)/2; res[i] -= d*p; res[j] += d*p
                    else:
                        res[j][0] += md/2; res[i][0] -= md/2
        if not any_ov: break
    return res

def get_planet_ancestor(node):
    c = node
    while c.parent is not None: c = c.parent
    return c

def simulate(nodes, planets, images):
    n = len(planets)
    positions = np.array([p.pos * POSITION_SCALE for p in planets])
    uncapped = np.array([max(compute_radius(p.size), MIN_EFFECTIVE_RADIUS) for p in planets])
    
    # Compute per-planet adaptive scales
    eff_scales = {}
    if n >= 2 and ADAPTIVE_SCALE_SENSITIVITY > 0:
        nn_dists = np.zeros(n)
        for i in range(n):
            min_d = float('inf')
            for j in range(n):
                if i == j: continue
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_d: min_d = d
            nn_dists[i] = min_d
        sorted_nn = np.sort(nn_dists)
        median_nn = float(np.median(sorted_nn))
        for i in range(n):
            density_factor = min(1.0, nn_dists[i] / median_nn)
            adapted = PLANET_SCALE * density_factor
            eff_scales[planets[i].nid] = PLANET_SCALE * (1 - ADAPTIVE_SCALE_SENSITIVITY) + adapted * ADAPTIVE_SCALE_SENSITIVITY
    else:
        for p in planets: eff_scales[p.nid] = PLANET_SCALE
    
    # Top-level overlap resolution
    capped = sibling_cap(positions, uncapped)
    final_pos = resolve_overlaps(positions, capped, TOP_LEVEL_MARGIN, TOP_LEVEL_SEP_ITERS)
    
    for i, p in enumerate(planets):
        p.world_pos = final_pos[i]
        p.sphere_radius = uncapped[i]
        p.displayed_radius = capped[i]
    
    def expand(node):
        planet = get_planet_ancestor(node)
        pwp, prp = planet.world_pos, planet.pos * POSITION_SCALE
        es = eff_scales[planet.nid]
        if node.children:
            cp = []; cu = []
            for c in node.children:
                cp.append(pwp + (c.pos * POSITION_SCALE - prp) * es)
                cu.append(max(compute_radius(c.size) * (DEPTH_RADIUS_FACTOR ** c.depth), MIN_EFFECTIVE_RADIUS))
            cp_arr = np.array(cp); cu_arr = np.array(cu)
            cc = sibling_cap(cp_arr, cu_arr)
            pr = node.displayed_radius
            if pr > 0:
                for k in range(len(node.children)):
                    if cc[k] > pr: cc[k] = max(pr, MIN_EFFECTIVE_RADIUS)
            for k, c in enumerate(node.children):
                c.world_pos = cp_arr[k]; c.sphere_radius = cu_arr[k]; c.displayed_radius = cc[k]
        for im in node.images:
            im.world_pos = pwp + (im.pos * POSITION_SCALE - prp) * es
        for c in node.children: expand(c)
    
    for p in planets: expand(p)

def evaluate_all(nodes, planets, images):
    valid = [i for i in images if i.world_pos is not None]
    N = len(valid)
    print(f"  Valid images: {N}")
    
    bp = np.array([i.before_pos for i in valid])
    ap = np.array([i.world_pos for i in valid])
    plids = np.array([i.planet for i in valid])
    pids = np.array([i.pid for i in valid])
    
    print("  Computing distance matrices...")
    Db = cdist(bp, bp); Da = cdist(ap, ap)
    print("  Sorting...")
    sb = np.argsort(Db, axis=1); sa = np.argsort(Da, axis=1)
    
    results = {"n_images": N, "n_nodes": len(nodes), "n_planets": len(planets)}
    
    # ── 1. WITHIN-PLANET FIDELITY ──
    print("  Within-planet fidelity...")
    planet_groups = defaultdict(list)
    for idx, img in enumerate(valid):
        planet_groups[img.planet].append(idx)
    
    wp = {}
    for pid in sorted(planet_groups.keys()):
        idxs = planet_groups[pid]
        n_p = len(idxs)
        if n_p < 2:
            wp[str(pid)] = {"n": n_p, "k5_mean": 1.0, "k5_min": 1.0, "k5_zero": 0,
                            "k10_mean": 1.0, "k10_min": 1.0, "k10_zero": 0}
            continue
        bp_p = bp[idxs]; ap_p = ap[idxs]
        Db_p = cdist(bp_p, bp_p); Da_p = cdist(ap_p, ap_p)
        sb_p = np.argsort(Db_p, axis=1); sa_p = np.argsort(Da_p, axis=1)
        entry = {"n": n_p}
        for k in [5, 10]:
            ek = min(k, n_p - 1)
            if ek < 1:
                entry[f"k{k}_mean"] = 1.0; entry[f"k{k}_min"] = 1.0; entry[f"k{k}_zero"] = 0
                continue
            ovs = []
            for i in range(n_p):
                kb = set(sb_p[i, 1:ek+1]); ka = set(sa_p[i, 1:ek+1])
                ovs.append(len(kb & ka) / ek)
            entry[f"k{k}_mean"] = round(float(np.mean(ovs)), 6)
            entry[f"k{k}_min"] = round(float(np.min(ovs)), 6)
            entry[f"k{k}_zero"] = int(sum(1 for o in ovs if o == 0.0))
        wp[str(pid)] = entry
    results["within_planet"] = wp
    
    perfect_k5 = sum(1 for v in wp.values() if v["k5_mean"] == 1.0)
    perfect_k10 = sum(1 for v in wp.values() if v["k10_mean"] == 1.0)
    results["within_planet_summary"] = {
        "planets_evaluated": len(wp),
        "perfect_k5": perfect_k5, "perfect_k10": perfect_k10,
        "overall_mean_k5": round(float(np.mean([v["k5_mean"] for v in wp.values()])), 6),
        "overall_mean_k10": round(float(np.mean([v["k10_mean"] for v in wp.values()])), 6),
    }
    
    # ── 2. GLOBAL FIDELITY ──
    print("  Global fidelity...")
    gf = {"n": N}
    all_overlaps = {}
    for k in [5, 10]:
        ovs = []
        for i in range(N):
            kb = set(sb[i, 1:k+1]); ka = set(sa[i, 1:k+1])
            ovs.append(len(kb & ka) / k)
        gf[f"k{k}"] = {
            "mean": round(float(np.mean(ovs)), 6),
            "median": round(float(np.median(ovs)), 6),
            "min": round(float(np.min(ovs)), 6),
            "zero_count": int(sum(1 for o in ovs if o == 0.0)),
            "zero_pct": round(100.0 * sum(1 for o in ovs if o == 0.0) / N, 2),
        }
        all_overlaps[k] = ovs
    
    # Cross-leaf intrusion
    for k in [5, 10]:
        cb, ca, tot = 0, 0, 0
        for i in range(N):
            for j in sb[i, 1:k+1]:
                tot += 1
                if pids[i] != pids[j]: cb += 1
            for j in sa[i, 1:k+1]:
                if pids[i] != pids[j]: ca += 1
        gf[f"cl{k}"] = {"before": round(100*cb/tot, 2), "after": round(100*ca/tot, 2),
                         "delta": round(100*(ca-cb)/tot, 2)}
    
    # Cross-planet intrusion
    for k in [5, 10]:
        cb, ca, tot = 0, 0, 0
        for i in range(N):
            for j in sb[i, 1:k+1]:
                tot += 1
                if plids[i] != plids[j]: cb += 1
            for j in sa[i, 1:k+1]:
                if plids[i] != plids[j]: ca += 1
        gf[f"cp{k}"] = {"before": round(100*cb/tot, 2), "after": round(100*ca/tot, 2),
                         "delta": round(100*(ca-cb)/tot, 2)}
    
    # Intra/cross-leaf preservation
    k = 5
    ip, it_, cp_, ct_ = 0, 0, 0, 0
    for i in range(N):
        kb = set(sb[i, 1:k+1]); ka = set(sa[i, 1:k+1])
        for j in kb:
            if pids[i] == pids[j]:
                it_ += 1
                if j in ka: ip += 1
            else:
                ct_ += 1
                if j in ka: cp_ += 1
    gf["pres"] = {
        "intra_p": int(ip), "intra_t": int(it_),
        "intra_pct": round(100*ip/it_, 2) if it_ > 0 else 0,
        "cross_p": int(cp_), "cross_t": int(ct_),
        "cross_pct": round(100*cp_/ct_, 2) if ct_ > 0 else 0,
    }
    results["global_fidelity"] = gf
    
    # ── 3. READABILITY ──
    print("  Readability...")
    groups = defaultdict(list)
    for img in valid: groups[img.pid].append(img)
    t_op, t_tp, nn_lt_q, g_min = 0, 0, 0, float('inf')
    for pid, imgs in groups.items():
        pos = np.array([i.world_pos for i in imgs]); n = len(imgs)
        cen = pos.mean(axis=0)
        gr = max(float(np.max(np.linalg.norm(pos - cen, axis=1))), 0.1)
        aqs = max(MIN_IMAGE_QUAD_SIZE, min(IMAGE_QUAD_SIZE, gr*0.4/math.sqrt(n)))
        if n < 2: continue
        D = cdist(pos, pos); np.fill_diagonal(D, np.inf)
        mn = np.min(D, axis=1)
        lmin = float(mn.min())
        if lmin < g_min: g_min = lmin
        om = D < aqs; np.fill_diagonal(om, False)
        t_op += int(np.sum(om))//2; t_tp += n*(n-1)//2
        nn_lt_q += int(np.sum(mn < aqs))
    
    opr = 100*t_op/t_tp if t_tp else 0
    
    # Inside parent
    iip = 0
    for img in valid:
        pn = img.parent_node
        if pn and pn.world_pos is not None:
            if np.linalg.norm(img.world_pos - pn.world_pos) < pn.sphere_radius: iip += 1
    
    # Node child overlap
    nco, tnc = 0, 0
    for nd in nodes.values():
        if nd.parent and nd.world_pos is not None and nd.parent.world_pos is not None:
            tnc += 1
            if np.linalg.norm(nd.world_pos - nd.parent.world_pos) < nd.parent.sphere_radius + nd.sphere_radius: nco += 1
    
    # Global NN distances
    np.fill_diagonal(Da, np.inf)
    nnd = np.min(Da, axis=1)
    pcts = {}
    for p in [10, 25, 50, 75, 90]:
        pcts[f"P{p}"] = round(float(np.percentile(nnd, p)), 4)
    
    # Readability score
    iop = nn_lt_q/N if N else 0
    ppp = iip/N if N else 0
    nop = nco/tnc if tnc else 0
    med_nn = float(np.median(nnd))
    sp = max(0, 1 - med_nn/0.2)
    rs = max(0, min(1, 1 - 0.4*iop - 0.3*ppp - 0.2*nop - 0.1*sp))
    
    if rs >= 0.90: verdict = "Excellent"
    elif rs >= 0.75: verdict = "Acceptable"
    elif rs >= 0.55: verdict = "Moderate concern"
    else: verdict = "Poor"
    
    results["readability"] = {
        "t_op": t_op, "t_tp": t_tp, "opr_pct": round(opr, 4),
        "nn_lt_q": nn_lt_q, "nn_lt_q_pct": round(100*nn_lt_q/N, 2),
        "g_min": round(g_min, 4) if g_min < float('inf') else None,
        "iip": iip, "iip_pct": round(100*iip/N, 2),
        "nco": nco, "tnc": tnc, "nco_pct": round(100*nco/tnc, 2) if tnc else 0,
        "nn_min": round(float(nnd.min()), 4), "nn_mean": round(float(nnd.mean()), 4),
        "nn_med": round(med_nn, 4), "nn_max": round(float(nnd.max()), 4),
        "pcts": pcts, "rs": round(rs, 4), "verdict": verdict,
    }
    
    # ── 4. ADDITIONAL METRICS ──
    print("  Additional metrics...")
    
    # Stress (Kruskal)
    upper = np.triu_indices(N, k=1)
    db_u = Db[upper]; da_u = Da[upper]
    db_norm = db_u / db_u.max() if db_u.max() > 0 else db_u
    da_norm = da_u / da_u.max() if da_u.max() > 0 else da_u
    stress = float(np.sqrt(np.sum((db_norm - da_norm)**2) / np.sum(db_norm**2)))
    
    # Spearman rank correlation (sampled for speed)
    from scipy.stats import spearmanr
    rng = np.random.RandomState(42)
    sample_size = min(500, N)
    sample_idxs = rng.choice(N, sample_size, replace=False)
    rank_corrs = []
    for i in sample_idxs:
        rc, _ = spearmanr(Db[i], Da[i])
        if not np.isnan(rc): rank_corrs.append(rc)
    
    # Trustworthiness & Continuity (k=5)
    k = 5
    trust_sum = 0
    cont_sum = 0
    for i in range(N):
        nn_after = set(sa[i, 1:k+1])
        nn_before = set(sb[i, 1:k+1])
        # Trustworthiness: neighbors in after but not in before
        for j in nn_after - nn_before:
            rank_in_before = int(np.where(sb[i] == j)[0][0])
            trust_sum += rank_in_before - k
        # Continuity: neighbors in before but not in after
        for j in nn_before - nn_after:
            rank_in_after = int(np.where(sa[i] == j)[0][0])
            cont_sum += rank_in_after - k
    denom = N * k * (2*N - 3*k - 1)
    trustworthiness = 1 - (2 / denom) * trust_sum if denom > 0 else 1.0
    continuity = 1 - (2 / denom) * cont_sum if denom > 0 else 1.0
    
    # Planet separation stats
    planet_positions = np.array([p.world_pos for p in planets])
    Dp = cdist(planet_positions, planet_positions)
    np.fill_diagonal(Dp, np.inf)
    planet_nn = np.min(Dp, axis=1)
    
    results["additional"] = {
        "kruskal_stress": round(stress, 6),
        "spearman_rank_corr_mean": round(float(np.mean(rank_corrs)), 6),
        "spearman_rank_corr_median": round(float(np.median(rank_corrs)), 6),
        "trustworthiness_k5": round(trustworthiness, 6),
        "continuity_k5": round(continuity, 6),
        "planet_nn_min": round(float(planet_nn.min()), 4),
        "planet_nn_max": round(float(planet_nn.max()), 4),
        "planet_nn_mean": round(float(planet_nn.mean()), 4),
        "planet_nn_median": round(float(np.median(planet_nn)), 4),
        "nn_std": round(float(nnd.std()), 4),
        "nn_iqr": round(float(np.percentile(nnd, 75) - np.percentile(nnd, 25)), 4),
        "k20_overlap": None,  # computed below
    }
    
    # k=20 overlap
    ovs20 = []
    for i in range(N):
        kb = set(sb[i, 1:21]); ka = set(sa[i, 1:21])
        ovs20.append(len(kb & ka) / 20)
    results["additional"]["k20_overlap"] = round(float(np.mean(ovs20)), 6)
    results["additional"]["k20_zero"] = int(sum(1 for o in ovs20 if o == 0.0))
    
    results["meta"] = {
        "dataset": "Butterfly (Dataset 2)",
        "project": "ButterflyClusterViz 2",
        "method": "Candidate K + Density-Adaptive Scaling",
        "csv": "unity_pruned_density_tree_butterfly_3d.csv",
        "positionScale": POSITION_SCALE,
        "planetScale": PLANET_SCALE,
        "adaptiveScaleSensitivity": ADAPTIVE_SCALE_SENSITIVITY,
        "siblingRadiusCapFactor": SIBLING_RADIUS_CAP_FACTOR,
        "topLevelMargin": TOP_LEVEL_MARGIN,
        "topLevelSeparationIterations": TOP_LEVEL_SEP_ITERS,
        "imageQuadSize": IMAGE_QUAD_SIZE,
        "minImageQuadSize": MIN_IMAGE_QUAD_SIZE,
        "radiusScale": RADIUS_SCALE,
        "depthRadiusFactor": DEPTH_RADIUS_FACTOR,
    }
    
    return results

def main():
    print("=" * 60)
    print("ButterflyClusterViz 2 — Current State Evaluation")
    print("=" * 60)
    
    nodes, planets, images = parse_csv(CSV_PATH)
    print(f"Dataset: {len(nodes)} nodes, {len(images)} images, {len(planets)} planets")
    
    # Before positions (raw UMAP × positionScale)
    for img in images: img.before_pos = img.pos * POSITION_SCALE
    
    # After positions (Candidate K with adaptive scaling)
    print("Simulating Candidate K...")
    simulate(nodes, planets, images)
    
    print("Evaluating...")
    results = evaluate_all(nodes, planets, images)
    
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
