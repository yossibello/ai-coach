import sqlite3
c = sqlite3.connect('aicoach_dev.db')
c.row_factory = sqlite3.Row
print("=== 8 most recent activities ===")
for r in c.execute("SELECT date, name, duration_seconds, avg_hr, avg_power, tss, review_status, quality_score, source FROM activities ORDER BY date DESC LIMIT 8"):
    d = dict(r)
    print(f"{d['date']}  {d['name'][:40]:40}  dur={d['duration_seconds']:>5}s  HR={d['avg_hr']}  P={d['avg_power']}  TSS={d['tss']}  status={d['review_status']}  qs={d['quality_score']}  src={d['source']}")

print("\n=== count by review_status ===")
for r in c.execute("SELECT review_status, COUNT(*) FROM activities GROUP BY review_status"):
    print(dict(r))
