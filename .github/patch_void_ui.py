from pathlib import Path

p = Path("tennis-quant/app/economic-validation.tsx")
s = p.read_text(encoding="utf-8")
repls = [
('''  settled: number;\n  pending: number;\n  wins: number;''','''  settled: number;\n  graded?: number;\n  pending: number;\n  wins: number;\n  voids?: number;'''),
('''  result: "WIN" | "LOSS" | null;''','''  result: "WIN" | "LOSS" | "VOID" | null;'''),
('''type HistoryFilter = "ALL" | "PENDING" | "WIN" | "LOSS";''','''type HistoryFilter = "ALL" | "PENDING" | "WIN" | "LOSS" | "VOID";'''),
('''  if (row.result === "LOSS") {\n    return { label: "PERDU", className: "historyStatus historyLoss" };\n  }\n\n  return { label: "EN ATTENTE", className: "historyStatus historyPending" };''','''  if (row.result === "LOSS") {\n    return { label: "PERDU", className: "historyStatus historyLoss" };\n  }\n\n  if (row.result === "VOID") {\n    return { label: "ANNULÉ", className: "historyStatus historyPending" };\n  }\n\n  return { label: "EN ATTENTE", className: "historyStatus historyPending" };'''),
('''      LOSS: history.filter((row) => row.result === "LOSS").length,\n    }),''','''      LOSS: history.filter((row) => row.result === "LOSS").length,\n      VOID: history.filter((row) => row.result === "VOID").length,\n    }),'''),
('''            ["LOSS", "Perdus"],\n          ] as const).map''','''            ["LOSS", "Perdus"],\n            ["VOID", "Annulés"],\n          ] as const).map'''),
('''          <small>{stats.settled} paris réglés</small>''','''          <small>{stats.graded ?? stats.settled} paris gradés · {stats.voids ?? 0} annulé(s)</small>'''),
]
for old, new in repls:
    if old not in s:
        raise SystemExit(f"missing EconomicValidation patch block: {old[:70]!r}")
    s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")

p = Path("tennis-quant/app/quality-economics.tsx")
s = p.read_text(encoding="utf-8")
for old, new in [
('''  settled: number;\n  pending: number;\n  wins: number;''','''  settled: number;\n  graded?: number;\n  pending: number;\n  wins: number;\n  voids?: number;'''),
('''                {row.name} · {row.settled} réglé(s) · ROI {pct(row.roi)} ·{" "}''','''                {row.name} · {row.graded ?? row.settled} gradé(s) · ROI {pct(row.roi)} ·{" "}'''),
]:
    if old not in s:
        raise SystemExit(f"missing QualityEconomics patch block: {old[:70]!r}")
    s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("Patched VOID UI semantics")
