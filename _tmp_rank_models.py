import json
from pathlib import Path

scores = json.loads(Path('venv/Lib/site-packages/audio_separator/models-scores.json').read_text(encoding='utf-8'))
rows = []
for fname, info in scores.items():
    stems = info.get('stems') or []
    med = info.get('median_scores') or {}
    if 'vocals' in stems and ('instrumental' in stems or 'other' in stems):
        voc = med.get('vocals', {}).get('SDR')
        inst_key = 'instrumental' if 'instrumental' in med else ('other' if 'other' in med else None)
        inst = med.get(inst_key, {}).get('SDR') if inst_key else None
        rows.append((voc if voc is not None else -999, inst if inst is not None else -999, fname, info.get('model_name', ''), stems, inst_key))

rows.sort(reverse=True)
print('Top 30 by vocals SDR among local audio-separator scored models:')
for voc, inst, fname, mname, stems, inst_key in rows[:30]:
    print(f'{voc:6.3f} / {inst:6.3f} | {fname} | {mname} | stems={stems} ({inst_key})')

print('\nRoformer-only subset:')
ro = [r for r in rows if 'roformer' in r[2].lower() or 'roformer' in r[3].lower()]
for voc, inst, fname, mname, stems, inst_key in ro[:40]:
    print(f'{voc:6.3f} / {inst:6.3f} | {fname} | {mname} | stems={stems} ({inst_key})')
