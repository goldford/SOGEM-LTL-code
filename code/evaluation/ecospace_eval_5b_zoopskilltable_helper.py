from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import re

SCENARIO = "SC215"
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall", "All"]
TAB_P = "..//..//data//evaluation//"

def _parse_filename(path: Path) -> tuple[str, str, str]:
    """
    Expect names like:
        ecospace_zoop_performance_SC215_ZC_Spring.csv
        ecospace_zoop_performance_SC215_ZS_Winter.csv

    Returns
    -------
    scenario, pass_name, season
    """
    m = re.match(
        r"ecospace_zoop_performance_(?P<scenario>.+?)_(?P<pass>ZC|ZS)_(?P<season>All|Winter|Spring|Summer|Fall)\.csv$",
        path.name,
    )
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")
    return m.group("scenario"), m.group("pass"), m.group("season")


def _season_sort(df: pd.DataFrame, season_col: str = "season") -> pd.DataFrame:
    order_map = {s: i for i, s in enumerate(SEASON_ORDER)}
    return (
        df.assign(_season_order=df[season_col].map(order_map).fillna(999))
          .sort_values(["_season_order"] + [c for c in df.columns if c not in {season_col, "_season_order"}])
          .drop(columns="_season_order")
          .reset_index(drop=True)
    )


def _round_numeric(df: pd.DataFrame, decimals: int | None) -> pd.DataFrame:
    out = df.copy()
    if decimals is None:
        return out
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].round(decimals)
    return out


def _save_csv(df: pd.DataFrame, out_csv: str | Path, decimals: int | None) -> None:
    out_csv = Path(out_csv)
    if decimals is None:
        df.to_csv(out_csv, index=False)
    else:
        df.to_csv(out_csv, index=False, float_format=f"%.{decimals}f")


def build_manuscript_skill_table(
    scenario: str,
    in_dir: str | Path,
    *,
    out_csv: str | Path | None = None,
    include_all: bool = True,
    decimals: int | None = 2,
    total_name_in_source: str = "Total",
) -> pd.DataFrame:
    """
    Build manuscript table:
        season, N, r, RMSE, MAE, WSS

    Uses the TOTAL row from each seasonal ZC file only.
    """
    in_dir = Path(in_dir)
    files = sorted(in_dir.glob(f"ecospace_zoop_performance_{scenario}_ZC_*.csv"))

    rows = []
    for f in files:
        _, _, season = _parse_filename(f)
        if (not include_all) and season == "All":
            continue

        df = pd.read_csv(f)
        sub = df[df["Group"].astype(str) == total_name_in_source].copy()
        if sub.empty:
            continue

        sub.insert(0, "season", season)
        sub = sub[["season", "N", "r", "RMSE", "MAE", "WSS"]]
        rows.append(sub)

    if not rows:
        raise ValueError(
            f"No matching ZC seasonal files with Group == '{total_name_in_source}' found for scenario {scenario}."
        )

    out = pd.concat(rows, ignore_index=True)
    out = _season_sort(out, "season")
    out = _round_numeric(out, decimals)

    if out_csv is None:
        out_csv = in_dir / f"ecospace_forpub_zoop_skill_{scenario}_ZC.csv"

    _save_csv(out, out_csv, decimals)
    print(f"Saved manuscript table: {out_csv}")
    return out


def build_supplement_skill_table(
    scenario: str,
    in_dir: str | Path,
    *,
    out_csv: str | Path | None = None,
    include_all: bool = True,
    decimals: int | None = 2,
    passes: tuple[str, ...] = ("ZC", "ZS"),
    total_name_in_source: str = "Total",
    total_name_fmt: str = "{pass_name}-TOTAL",
) -> pd.DataFrame:
    """
    Build supplement table:
        season, group, N, r, RMSE, MAE, WSS

    Stacks ZC and ZS files and keeps individual groups.
    Renames source 'Total' rows to e.g. ZC-TOTAL / ZS-TOTAL.
    """
    in_dir = Path(in_dir)

    rows = []
    for pass_name in passes:
        files = sorted(in_dir.glob(f"ecospace_zoop_performance_{scenario}_{pass_name}_*.csv"))
        for f in files:
            _, _, season = _parse_filename(f)
            if (not include_all) and season == "All":
                continue

            df = pd.read_csv(f).copy()
            df.insert(0, "season", season)
            df["group"] = df["Group"].astype(str)
            df.loc[df["group"] == total_name_in_source, "group"] = total_name_fmt.format(pass_name=pass_name)
            df = df[["season", "group", "N", "r", "RMSE", "MAE", "WSS"]]
            rows.append(df)

    if not rows:
        raise ValueError(f"No matching ZC/ZS seasonal files found for scenario {scenario}.")

    out = pd.concat(rows, ignore_index=True)

    # Keep seasons in the desired order; within season, put totals first, then alphabetical
    season_order = {s: i for i, s in enumerate(SEASON_ORDER)}
    out["_season_order"] = out["season"].map(season_order).fillna(999)
    out["_group_order"] = out["group"].str.contains("TOTAL", na=False).map({True: 0, False: 1})
    out = (
        out.sort_values(["_season_order", "_group_order", "group"])
           .drop(columns=["_season_order", "_group_order"])
           .reset_index(drop=True)
    )

    out = _round_numeric(out, decimals)

    if out_csv is None:
        out_csv = in_dir / f"ecospace_forpub_zoop_skill_{scenario}_ALLGROUPS.csv"

    _save_csv(out, out_csv, decimals)
    print(f"Saved supplement table: {out_csv}")
    return out


def export_forpub_zoop_skill_tables(
    scenario: str,
    in_dir: str | Path,
    *,
    include_all: bool = True,
    decimals: int | None = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper:
      1) manuscript table from ZC total rows
      2) supplement table from ZC + ZS seasonal files

    Defaults now include:
      - include_all=True
      - decimals=2
    """
    ms = build_manuscript_skill_table(
        scenario=scenario,
        in_dir=in_dir,
        include_all=include_all,
        decimals=decimals,
    )
    supp = build_supplement_skill_table(
        scenario=scenario,
        in_dir=in_dir,
        include_all=include_all,
        decimals=decimals,
    )
    return ms, supp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build manuscript and supplement zooplankton skill tables from seasonal performance CSVs."
    )
    parser.add_argument(
        "--scenario",
        default=SCENARIO,
        help="Scenario name used in the filenames, e.g. SC215",
    )
    parser.add_argument(
        "--in-dir",
        default=TAB_P,
        help="Directory containing ecospace_zoop_performance_<scenario>_*.csv files",
    )
    parser.add_argument(
        "--exclude-all",
        action="store_true",
        help="Exclude the All-season summary rows/files",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimal places for numeric columns; use -1 for no rounding",
    )

    args = parser.parse_args()

    decimals = None if args.decimals < 0 else args.decimals

    export_forpub_zoop_skill_tables(
        scenario=args.scenario,
        in_dir=Path(args.in_dir),
        include_all=not args.exclude_all,
        decimals=decimals,
    )


if __name__ == "__main__":
    main()
