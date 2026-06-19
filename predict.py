"""predict.py — CLI for match predictions."""

import argparse
from model import predict_match


def print_prediction(s: dict) -> None:
    venue = "neutral" if s["neutral"] else f"{s['home_team']} home"
    print(f"\n{s['home_team']} ({s['home_elo']:.0f}) vs "
          f"{s['away_team']} ({s['away_elo']:.0f})   [{venue}]")
    print(f"  Expected goals:               {s['lambda_home']:.2f}  vs  {s['lambda_away']:.2f}")
    print(f"  Win / Draw / Loss:            "
          f"{s['p_home_win']:.1%}  /  {s['p_draw']:.1%}  /  {s['p_away_win']:.1%}")
    print(f"  Over 1.5 / 2.5 / 3.5 goals:   "
          f"{s['p_over_1.5']:.1%}  /  {s['p_over_2.5']:.1%}  /  {s['p_over_3.5']:.1%}")
    print(f"  Both teams to score:          {s['p_btts']:.1%}")
    print(f"  Clean sheet (home / away):    "
          f"{s['p_clean_sheet_home']:.1%}  /  {s['p_clean_sheet_away']:.1%}")
    print(f"  Most likely scorelines:")
    for h, a, p in s["top_scorelines"]:
        print(f"    {h}-{a}  {p:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home_team", nargs="?", default=None)
    ap.add_argument("away_team", nargs="?", default=None)
    ap.add_argument("--at-home", action="store_true",
                    help="Home team gets home advantage (default: neutral)")
    args = ap.parse_args()

    if args.home_team and args.away_team:
        print_prediction(predict_match(args.home_team, args.away_team,
                                        neutral=not args.at_home))
        return

    demos = [("Argentina", "Brazil"), ("France", "Germany"),
             ("Spain", "Portugal"), ("England", "Netherlands"),
             ("Morocco", "Japan"), ("Saudi Arabia", "Argentina")]
    for h, a in demos:
        try:
            print_prediction(predict_match(h, a, neutral=True))
        except KeyError as e:
            print(f"\nSkipped {h} vs {a}: {e}")


if __name__ == "__main__":
    main()