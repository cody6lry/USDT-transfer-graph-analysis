import json, math, re
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / 'data_sensitivity'
OUT = ROOT / 'results' / 'sampling_sensitivity'
OUT.mkdir(parents=True, exist_ok=True)
ZERO = '0x0000000000000000000000000000000000000000'

def gini(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if len(x) == 0 or x.sum() <= 0:
        return float('nan')
    x = np.sort(x)
    n = len(x)
    i = np.arange(1, n + 1)
    return float((2 * np.sum(i * x) / (n * x.sum())) - (n + 1) / n)

def top_share(x, frac=0.01):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    k = max(1, math.ceil(len(x) * frac))
    return float(np.partition(x, len(x) - k)[len(x) - k:].sum() / x.sum())

def event_key(df):
    h = df.get('hash', pd.Series('', index=df.index)).astype('string').fillna('').str.lower().str.strip()
    li = df.get('logIndex', pd.Series('', index=df.index)).astype('string').fillna('').str.strip()
    preferred = 'log|' + h + '|' + li
    fallback = (
        'fallback|' + h + '|' +
        df.get('blockNumber', pd.Series('', index=df.index)).astype('string').fillna('') + '|' +
        df['from'].astype('string').fillna('').str.lower() + '|' +
        df['to'].astype('string').fillna('').str.lower() + '|' +
        df['value'].astype('string').fillna('')
    )
    return preferred.where((h != '') & (li != ''), fallback)

def preprocess(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df['from'] = df['from'].astype('string').str.lower().str.strip()
    df['to'] = df['to'].astype('string').str.lower().str.strip()
    raw = pd.to_numeric(df['value'], errors='coerce')
    dec = pd.to_numeric(df['tokenDecimal'], errors='coerce')
    valid = df['from'].notna() & df['to'].notna() & raw.notna() & dec.notna()
    df = df.loc[valid].copy()
    df['event_key'] = event_key(df)
    df = df.drop_duplicates('event_key')
    raw = pd.to_numeric(df['value'], errors='coerce')
    dec = pd.to_numeric(df['tokenDecimal'], errors='coerce')
    df['value_usdt'] = raw / np.power(10.0, dec)
    positive = df.loc[df['value_usdt'] > 0].copy()
    graph_df = positive.loc[
        (positive['from'] != positive['to']) &
        (positive['from'] != ZERO) &
        (positive['to'] != ZERO)
    ].copy()
    return positive, graph_df

def graph_from(df):
    edges = df.groupby(['from','to'], sort=False, as_index=False).agg(weight=('value_usdt','sum'))
    G = nx.DiGraph()
    G.add_weighted_edges_from((r['from'], r['to'], float(r['weight'])) for _, r in edges.iterrows())
    return G

def largest_sizes(G):
    w = max((len(c) for c in nx.weakly_connected_components(G)), default=0)
    s = max((len(c) for c in nx.strongly_connected_components(G)), default=0)
    return w, s

def assort(G, x, y):
    try:
        return float(nx.degree_assortativity_coefficient(G, x=x, y=y))
    except Exception:
        return float('nan')

def degree_attack(G):
    n = G.number_of_nodes()
    k = round(0.001 * n)
    order = sorted(G.nodes(), key=lambda node: (-G.degree(node), node))
    H = G.copy(); H.remove_nodes_from(order[:k])
    w = max((len(c) for c in nx.weakly_connected_components(H)), default=0)
    return k, w / n

def seed_from_name(path):
    m = re.search(r'seed_(\d+)_n100', path.stem)
    return int(m.group(1)) if m else None

def main():
    rows = []
    files = sorted(INPUT_DIR.glob('usdt_2025_10_seed_*_n100.csv'))
    if not files:
        raise FileNotFoundError(f'No sensitivity CSV files in {INPUT_DIR}')

    for path in files:
        seed = seed_from_name(path)
        print('Analyzing', path.name)
        positive, graph_df = preprocess(path)
        values = positive['value_usdt'].to_numpy(float)
        G = graph_from(graph_df)
        n, m = G.number_of_nodes(), G.number_of_edges()
        w, s = largest_sizes(G)
        removed, attack_share = degree_attack(G)
        row = {
            'seed': seed,
            'raw_rows': len(pd.read_csv(path, usecols=['hash'])),
            'positive_events': len(positive),
            'graph_events': len(graph_df),
            'gini': gini(values),
            'top1_event_share': top_share(values),
            'nodes': n,
            'edges': m,
            'largest_wcc_share': w / n,
            'largest_scc_share': s / n,
            'assort_out_in': assort(G, 'out', 'in'),
            'assort_in_out': assort(G, 'in', 'out'),
            'assort_out_out': assort(G, 'out', 'out'),
            'assort_in_in': assort(G, 'in', 'in'),
            'degree_0_1_removed_nodes': removed,
            'degree_0_1_wcc_share': attack_share,
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    df = pd.DataFrame(rows).sort_values('seed')
    df.to_csv(OUT / 'seed_sensitivity_summary.csv', index=False, encoding='utf-8-sig')

    metrics = [
        'gini','top1_event_share','nodes','edges','largest_wcc_share',
        'largest_scc_share','assort_out_in','assort_in_out','assort_out_out',
        'assort_in_in','degree_0_1_wcc_share'
    ]
    ranges = []
    for metric in metrics:
        vals = pd.to_numeric(df[metric], errors='coerce').dropna()
        mean = vals.mean()
        ranges.append({
            'metric': metric,
            'mean': mean,
            'min': vals.min(),
            'max': vals.max(),
            'absolute_range': vals.max() - vals.min(),
            'relative_range': (vals.max() - vals.min()) / abs(mean) if mean != 0 else np.nan,
        })
    pd.DataFrame(ranges).to_csv(OUT / 'seed_sensitivity_ranges.csv', index=False, encoding='utf-8-sig')
    print('\n', df.to_string(index=False))
    print('\nSaved to:', OUT)

if __name__ == '__main__':
    main()
