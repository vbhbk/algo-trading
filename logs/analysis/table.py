import json, sys
d = {r['file']: r for r in json.load(open(sys.argv[1]))}
names = ['sample_v2','round1','round2_public','round3_public','round4_public','round1_a','round1_b','round2_a','round2_b','round3_a','round3_b','round4_a','round4_b','round4_a1','round4_a2','round4_b1','round4_b2','round4_b3','sample_v2_a','sample_v2_b']
short = lambda n: n.replace('_public','').replace('round','R').replace('sample_v2','S2')
def tab(title, cols):
    print(f'\n### {title}\n'); print('| file | ' + ' | '.join(c[0] for c in cols) + ' |'); print('|---' * (len(cols)+1) + '|')
    for n in names:
        if n not in d: continue
        r = d[n]; print(f'| {short(n)} | ' + ' | '.join(c[2].format(c[1](r)) for c in cols) + ' |')
g = lambda k: (lambda r: r[k]); m = lambda k: (lambda r: r['markout'][k])
tab('A. Цена и волатильность', [('T, s', g('T'), '{:.0f}'), ('range', g('range_total'), '{:.0f}'), ('|Δ10s| ср', g('absmove_10s_mean'), '{:.2f}'), ('|Δ60s| ср', g('absmove_60s_mean'), '{:.1f}'), ('|Δ10m| ср', g('absmove_10m_mean'), '{:.0f}'), ('|Δ10m| q95', g('absmove_10m_q95'), '{:.0f}'), ('σ 1s', g('rv_1s_per_sqrt_s'), '{:.2f}'), ('скачки≥5/1000s', g('jumps5_per_1000s'), '{:.1f}'), ('10m range мед', g('win10m_range_med'), '{:.0f}'), ('10m range max', g('win10m_range_max'), '{:.0f}'), ('σ1s 10m min–max', lambda r: (r['win10m_rv_min'], r['win10m_rv_max']), '{0[0]:.2f}–{0[1]:.2f}')])
tab('B. Спред, глубина, отмены', [('spread ср', g('spread_mean'), '{:.2f}'), ('spr=1', g('spr1'), '{:.1%}'), ('spr=2', g('spr2'), '{:.1%}'), ('spr≥3', g('spr3p'), '{:.1%}'), ('L1 объём мед', g('l1_vol_med'), '{:.0f}'), ('L1 объём ср', g('l1_vol_mean'), '{:.0f}'), ('отмены', g('cancel_share'), '{:.1%}'), ('событий/с', lambda r: r['n_events']/r['T'], '{:.0f}')])
tab('C. Агрессивный поток (IOC)', [('IOC/вставки', g('ioc_share'), '{:.1%}'), ('сделок/мин', g('trades_per_min'), '{:.0f}'), ('размер мед', g('trade_size_med'), '{:.0f}'), ('размер ср', g('trade_size_mean'), '{:.0f}'), ('≥2 уровней', g('multi_level_share'), '{:.1%}'), ('лотов/с', g('flow_lots_per_s'), '{:.0f}')])
tab('D. Автокорреляция приращений mid', [('ac 1s', g('ac_1s'), '{:+.3f}'), ('ac 5s', g('ac_5s'), '{:+.3f}'), ('ac 30s', g('ac_30s'), '{:+.3f}'), ('VR 30s/1s', g('var_ratio_30_1'), '{:.2f}')])
tab('E. Markout mid после сделки, тики в сторону агрессора (все / <30 / ≥300 лотов)', [(f'{h}s all', m(f'all_{h}s'), '{:+.2f}') for h in (0,1,5,30)] + [(f'{h}s <30', m(f'lt30_{h}s'), '{:+.2f}') for h in (1,5,30)] + [(f'{h}s 100-300', m(f'100_300_{h}s'), '{:+.2f}') for h in (1,5,30)] + [(f'{h}s ≥300', m(f'ge300_{h}s'), '{:+.2f}') for h in (1,5,30)])
tab('F. Предсказуемость mid (корреляция)', [('imb→1s', g('corr_imb_1s'), '{:+.3f}'), ('imb→5s', g('corr_imb_5s'), '{:+.3f}'), ('imb|spr1→1s', g('corr_imb_spr1_1s'), '{:+.3f}'), ('flow1s→1s', g('corr_flow1s_1s'), '{:+.3f}'), ('flow1s→5s', g('corr_flow1s_5s'), '{:+.3f}'), ('flow5s→1s', g('corr_flow5s_1s'), '{:+.3f}'), ('flow5s→5s', g('corr_flow5s_5s'), '{:+.3f}')])
