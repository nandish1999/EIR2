#!/usr/bin/env python3
"""
Parameter sweep for Dataset 2 optimization.
Tests combinations of planetScale and positionScale (the two parameters
that most affect global fidelity and readability).
"""
import csv, math, os, sys, json
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, "Assets", "StreamingAssets", "Data",
                        "unity_pruned_density_tree_butterfly_3d.csv")

# Fixed parameters
RADIUS_SCALE = 0.3
MIN_RADIUS = 0.3
DEPTH_RADIUS_FACTOR = 0.65
MIN_EFFECTIVE_RADIUS = 0.1
SIBLING_RADIUS_CAP_FACTOR = 0.6
TOP_LEVEL_MARGIN = 1.2
TOP_LEVEL_SEPARATION_ITERATIONS = 300

class Node:
    __slots__ = ["nid","pid","planet","size","pos","children","images","depth","parent","world_pos","sphere_radius","displayed_radius"]
    def __init__(self, nid, pid, planet, size, pos):
        self.nid, self.pid, self.planet = nid, pid, int(planet)
        self.size = int(size); self.pos = np.array(pos, dtype=np.float64)
        self.children, self.images = [], []; self.depth = 0
        self.parent = None; self.world_pos = None; self.sphere_radius = 0.0; self.displayed_radius = 0.0

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

def compute_sphere_radius(node):
    r = max(math.log2(node.size + 1) * RADIUS_SCALE, MIN_RADIUS)
    return max(r * (DEPTH_RADIUS_FACTOR ** node.depth), MIN_EFFECTIVE_RADIUS)

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

def simulate(planets, pos_scale, planet_scale):
    n = len(planets)
    positions = np.array([p.pos * pos_scale for p in planets])
    uncapped = np.array([compute_sphere_radius(p) for p in planets])
    capped = sibling_cap(positions, uncapped)
    final_pos = resolve_overlaps(positions, capped, TOP_LEVEL_MARGIN, TOP_LEVEL_SEPARATION_ITERATIONS)
    for i, p in enumerate(planets):
        p.world_pos = final_pos[i]; p.sphere_radius = uncapped[i]; p.displayed_radius = capped[i]
    def expand(node):
        planet = get_planet_ancestor(node)
        pwp, prp = planet.world_pos, planet.pos * pos_scale
        if node.children:
            cp = []; cu = []
            for c in node.children:
                cp.append(pwp + (c.pos * pos_scale - prp) * planet_scale)
                cu.append(compute_sphere_radius(c))
            cp_arr = np.array(cp); cu_arr = np.array(cu)
            cc = sibling_cap(cp_arr, cu_arr)
            pr = node.displayed_radius
            if pr > 0:
                for k in range(len(node.children)):
                    if cc[k] > pr: cc[k] = max(pr, MIN_EFFECTIVE_RADIUS)
            for k, c in enumerate(node.children):
                c.world_pos = cp_arr[k]; c.sphere_radius = cu_arr[k]; c.displayed_radius = cc[k]
        for im in node.images:
            im.world_pos = pwp + (im.pos * pos_scale - prp) * planet_scale
        for c in node.children: expand(c)
    for p in planets: expand(p)

def evaluate(images, nodes, image_quad_size):
    valid = [i for i in images if i.world_pos is not None]
    N = len(valid)
    bp = np.array([i.before_pos for i in valid])
    ap = np.array([i.world_pos for i in valid])
    plids = np.array([i.planet for i in valid])
    pids = np.array([i.pid for i in valid])
    Db = cdist(bp, bp); Da = cdist(ap, ap)
    sb = np.argsort(Db, axis=1); sa = np.argsort(Da, axis=1)
    # Global k=5
    ovs = []
    for i in range(N):
        kb = set(sb[i,1:6]); ka = set(sa[i,1:6])
        ovs.append(len(kb & ka) / 5)
    gk5 = float(np.mean(ovs))
    zero5 = sum(1 for o in ovs if o == 0.0)
    # Cross-planet intrusion k=5
    cb, ca, tot = 0, 0, 0
    for i in range(N):
        for j in sb[i,1:6]:
            tot += 1
            if plids[i] != plids[j]: cb += 1
        for j in sa[i,1:6]:
            if plids[i] != plids[j]: ca += 1
    cp_before = 100*cb/tot; cp_after = 100*ca/tot; cp_delta = cp_after - cp_before
    # Readability
    groups = defaultdict(list)
    for img in valid: groups[img.pid].append(img)
    t_op, t_tp, nn_lt_q = 0, 0, 0
    MIN_IQS = 0.05
    for pid, imgs in groups.items():
        pos = np.array([i.world_pos for i in imgs]); n = len(imgs)
        cen = pos.mean(axis=0)
        gr = max(np.max(np.linalg.norm(pos - cen, axis=1)), 0.1)
        aqs = max(MIN_IQS, min(image_quad_size, gr*0.4/math.sqrt(n)))
        if n < 2: continue
        D = cdist(pos, pos); np.fill_diagonal(D, np.inf)
        mn = np.min(D, axis=1)
        om = D < aqs; np.fill_diagonal(om, False)
        t_op += int(np.sum(om))//2; t_tp += n*(n-1)//2
        nn_lt_q += int(np.sum(mn < aqs))
    opr = 100*t_op/t_tp if t_tp else 0
    iop = nn_lt_q/N if N else 0
    # Inside parent
    iip = 0
    for img in valid:
        pn = img.parent_node
        if pn and pn.world_pos is not None:
            if np.linalg.norm(img.world_pos - pn.world_pos) < pn.sphere_radius: iip += 1
    ppp = iip/N if N else 0
    # Node child overlap
    nco, tnc = 0, 0
    for nd in nodes.values():
        if nd.parent and nd.world_pos is not None and nd.parent.world_pos is not None:
            tnc += 1
            if np.linalg.norm(nd.world_pos - nd.parent.world_pos) < nd.parent.sphere_radius + nd.sphere_radius: nco += 1
    nop = nco/tnc if tnc else 0
    ap2 = np.array([i.world_pos for i in valid])
    Da2 = cdist(ap2, ap2); np.fill_diagonal(Da2, np.inf)
    nnd = np.min(Da2, axis=1)
    med_nn = float(np.median(nnd))
    sp = max(0, 1 - med_nn/0.2)
    rs = max(0, min(1, 1 - 0.4*iop - 0.3*ppp - 0.2*nop - 0.1*sp))
    return {
        "gk5": round(gk5, 4), "zero5": zero5, "zero5_pct": round(100*zero5/N, 2),
        "cp_delta": round(cp_delta, 2),
        "opr": round(opr, 2), "nn_lt_q_pct": round(100*nn_lt_q/N, 1),
        "iip_pct": round(100*iip/N, 1), "nco_pct": round(100*nco/tnc, 1) if tnc else 0,
        "med_nn": round(med_nn, 4), "rs": round(rs, 4),
    }

def reset_nodes(nodes, images):
    for n in nodes.values(): n.world_pos = None; n.sphere_radius = 0; n.displayed_radius = 0
    for i in images: i.world_pos = None

def main():
    nodes, planets, images = parse_csv(CSV_PATH)
    print(f"Dataset: {len(nodes)} nodes, {len(images)} images, {len(planets)} planets")

    # Parameter ranges to test
    pos_scales = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    planet_scales = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    image_quad_sizes = [0.3, 0.5, 0.8]

    results = []
    total = len(pos_scales) * len(planet_scales) * len(image_quad_sizes)
    count = 0

    for iqs in image_quad_sizes:
        for ps in pos_scales:
            # Set before positions with this positionScale
            for img in images: img.before_pos = img.pos * ps
            for pls in planet_scales:
                count += 1
                reset_nodes(nodes, images)
                simulate(planets, ps, pls)
                r = evaluate(images, nodes, iqs)
                r["positionScale"] = ps
                r["planetScale"] = pls
                r["imageQuadSize"] = iqs
                results.append(r)
                print(f"[{count}/{total}] ps={ps} pls={pls} iqs={iqs} -> "
                      f"gk5={r['gk5']:.4f} rs={r['rs']:.4f} opr={r['opr']:.2f}% "
                      f"cp_delta={r['cp_delta']:+.1f}pp zero5={r['zero5_pct']:.1f}%")

    # Sort by composite score: prioritize global fidelity + readability
    for r in results:
        r["composite"] = round(0.5 * r["gk5"] + 0.35 * r["rs"] + 0.15 * max(0, 1 - r["opr"]/10), 4)

    results.sort(key=lambda x: -x["composite"])

    print("\n" + "="*80)
    print("TOP 15 PARAMETER COMBINATIONS (by composite score)")
    print("="*80)
    print(f"{'Rank':<5} {'posScale':<9} {'plnScale':<9} {'iqs':<5} {'gk5':<7} {'rs':<7} "
          f"{'opr%':<7} {'cpΔ':<8} {'zero5%':<8} {'nnltq%':<8} {'medNN':<8} {'comp':<7}")
    for i, r in enumerate(results[:15]):
        print(f"{i+1:<5} {r['positionScale']:<9} {r['planetScale']:<9} {r['imageQuadSize']:<5} "
              f"{r['gk5']:<7.4f} {r['rs']:<7.4f} {r['opr']:<7.2f} {r['cp_delta']:<+8.1f} "
              f"{r['zero5_pct']:<8.1f} {r['nn_lt_q_pct']:<8.1f} {r['med_nn']:<8.4f} {r['composite']:<7.4f}")

    # Save full results
    out_path = os.path.join(PROJECT_ROOT, "param_sweep_results.json")
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\nFull results: {out_path}")

if __name__ == "__main__":
    main()
