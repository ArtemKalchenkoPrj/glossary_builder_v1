"""Evaluate intent_filter_psp_providers.pre_filter() on the labeled test dataset.

Loads dataset.xlsx (columns: text, label, group) and measures:
  - Recall    on A1+A2 (true positives — should PASS)
  - Precision on B1+B2+B3 (true negatives — should DROP)
  - Full breakdown per group
  - False negatives listed with matched terms and signals (for fixing)

Usage:
    python eval_prefilter.py --dataset dataset.xlsx --db provider_glossary.db
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

# Allow running from anywhere — add both the module folder and the project root
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))           # PSP_provider_glossary_builder/
sys.path.insert(0, str(_THIS.parent))    # project root

from intent_filter_psp_providers import (
    pre_filter,
    matched_terms,
    matched_signals,
    load_glossary_terms,
    SELLER_SIGNALS,
    DEFAULT_PROVIDER_GLOSSARY_DB,
)

console = Console()


def load_dataset(path: str) -> pd.DataFrame:
    if str(path).endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df["text"]  = df["text"].fillna("").astype(str).str.strip()
    df["label"] = (
        df["label"].astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    df["group"] = df["group"].astype(str).str.strip().str.upper()
    return df


@click.command()
@click.option("--dataset", "-d", default="dataset.xlsx",
              type=click.Path(exists=True),
              help="Master test dataset (text, label, group).")
@click.option("--db", "db_path",
              default=DEFAULT_PROVIDER_GLOSSARY_DB,
              type=click.Path(),
              help="Provider glossary SQLite DB.")
@click.option("--show-fn", is_flag=True, default=True,
              help="Print false negatives (missed true providers).")
@click.option("--show-fp", is_flag=True, default=False,
              help="Print false positives (negatives that passed).")
def main(dataset, db_path, show_fn, show_fp):
    """Pre-filter recall/precision eval on labeled test dataset."""

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_dataset(dataset)
    terms = load_glossary_terms(db_path)

    console.rule("[bold]PSP Provider Pre-filter Eval")
    console.print(f"Dataset  : [cyan]{dataset}[/]  ({len(df)} examples)")
    console.print(f"Glossary : [cyan]{db_path}[/]  ({len(terms)} surface terms)")
    console.print(f"Signals  : {len(SELLER_SIGNALS)} seller-intent signals")
    console.print(f"Logic    : AND\n")

    # ── Run pre_filter on every row ───────────────────────────────────────────
    df["pass"]           = df["text"].apply(lambda t: pre_filter(t, db_path))
    df["terms_hit"]      = df["text"].apply(lambda t: matched_terms(t, db_path))
    df["signals_hit"]    = df["text"].apply(lambda t: matched_signals(t))
    df["n_terms"]        = df["terms_hit"].apply(len)
    df["n_signals"]      = df["signals_hit"].apply(len)

    # ── Overall metrics ───────────────────────────────────────────────────────
    positives = df[df["label"] == True]
    negatives = df[df["label"] == False]

    tp = (positives["pass"] == True).sum()
    fn = (positives["pass"] == False).sum()
    fp = (negatives["pass"] == True).sum()
    tn = (negatives["pass"] == False).sum()

    recall    = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    console.print(f"[bold]Overall[/]")
    console.print(f"  Recall      (positives passed) : [{'green' if recall >= 0.7 else 'red'}]{recall:.1%}[/]  "
                  f"({tp} caught, {fn} missed)")
    console.print(f"  Specificity (negatives dropped): [{'green' if specificity >= 0.7 else 'red'}]{specificity:.1%}[/]  "
                  f"({tn} dropped, {fp} passed)")
    console.print(f"  Precision                      : {precision:.1%}")
    console.print()

    # ── Per-group breakdown ───────────────────────────────────────────────────
    table = Table(show_header=True, header_style="bold cyan", title="Per-group breakdown")
    table.add_column("Group",  width=7)
    table.add_column("Label",  width=8)
    table.add_column("Total",  justify="right", width=7)
    table.add_column("Passed", justify="right", width=7)
    table.add_column("Dropped",justify="right", width=8)
    table.add_column("Pass %", justify="right", width=8)

    for grp in ["A1", "A2", "B1", "B2", "B3"]:
        g = df[df["group"] == grp]
        if g.empty:
            continue
        lbl      = "positive" if g["label"].iloc[0] else "negative"
        total    = len(g)
        passed   = g["pass"].sum()
        dropped  = total - passed
        pct      = passed / total if total else 0
        # Color: green if positive+high or negative+low
        is_pos   = g["label"].iloc[0]
        color    = "green" if (is_pos and pct >= 0.7) or (not is_pos and pct <= 0.3) else "red"
        table.add_row(grp, lbl, str(total), str(passed), str(dropped),
                      f"[{color}]{pct:.0%}[/{color}]")

    console.print(table)
    console.print()

    # ── False Negatives — missed true providers ───────────────────────────────
    if show_fn:
        fns = df[(df["label"] == True) & (df["pass"] == False)]
        console.rule(f"[bold red]False Negatives — {len(fns)} missed providers[/]")
        if fns.empty:
            console.print("[green]None — all positives passed.[/]")
        else:
            for _, row in fns.iterrows():
                snippet = row["text"].replace("\n", " ")[:160]
                console.print(
                    f"[dim][{row['group']}][/]  "
                    f"terms=[yellow]{row['terms_hit'] or '—'}[/]  "
                    f"signals=[yellow]{row['signals_hit'] or '—'}[/]"
                )
                console.print(f"  {snippet}")
                console.print()

    # ── False Positives — negatives that passed ───────────────────────────────
    if show_fp:
        fps = df[(df["label"] == False) & (df["pass"] == True)]
        console.rule(f"[bold yellow]False Positives — {len(fps)} negatives passed[/]")
        if fps.empty:
            console.print("[green]None — all negatives dropped.[/]")
        else:
            for _, row in fps.iterrows():
                snippet = row["text"].replace("\n", " ")[:160]
                console.print(
                    f"[dim][{row['group']}][/]  "
                    f"terms=[cyan]{row['terms_hit']}[/]  "
                    f"signals=[cyan]{row['signals_hit']}[/]"
                )
                console.print(f"  {snippet}")
                console.print()

    # ── Missing signal analysis (FN only) ─────────────────────────────────────
    fns = df[(df["label"] == True) & (df["pass"] == False)]
    no_signal  = fns[fns["n_signals"] == 0]
    no_term    = fns[fns["n_terms"]   == 0]
    both_miss  = fns[(fns["n_signals"] == 0) & (fns["n_terms"] == 0)]

    console.rule("[bold]False Negative diagnosis[/]")
    console.print(f"  No signal only (term ok, signal missing) : {(fns['n_signals'] == 0) & (fns['n_terms'] > 0)}.sum()")
    console.print(f"  No signal (regardless of term)           : {len(no_signal)}")
    console.print(f"  No term   (regardless of signal)         : {len(no_term)}")
    console.print(f"  Both missing                             : {len(both_miss)}")
    console.print()
    console.print("[dim]Fix priority:[/]")
    console.print("  → Many 'no signal' FN  : add missing patterns to SELLER_SIGNALS")
    console.print("  → Many 'no term'  FN  : add missing terms to provider_glossary.db or GENERIC_TERMS_EXCLUDE")
    console.print("  → Many 'both'     FN  : message is too implicit — pre-filter can't catch it, LLM stage will")


if __name__ == "__main__":
    main()