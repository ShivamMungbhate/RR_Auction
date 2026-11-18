from __future__ import annotations
import html, json, re
from datetime import datetime
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Who holds up under pressure? | SupeRR Selector S2", page_icon=":material/query_stats:",
                   layout="wide", initial_sidebar_state="expanded")

RESULTS_DIR = Path(__file__).parent / "results_final2"
T = dict(bg="#080D18", bg2="#0C1423", card="#101A2B", elev="#121E31", text="#F1F5FC", muted="#91A0B7",
         line="#26364F", grid="#1D2B43", pink="#FF5C9A", pink_lt="#FF9BC2", blue="#49689D", blue_lt="#8AA6D6")
ROLE = {"bat": dict(label="Batters", color=T["blue"], tints=[T["blue"], T["blue_lt"], "#C9D6EE"], light=T["blue_lt"]),
        "bowl": dict(label="Bowlers", color=T["pink"], tints=[T["pink"], T["pink_lt"], "#FFD3E3"], light=T["pink_lt"])}
FACTS = {"matches": 812, "deliveries": 192056, "wp_logloss": 0.534, "coin_logloss": 0.693,
         "wicket_cost": {"Powerplay": 12.35, "Middle": 7.44, "Death": 2.95}, "shrinkage_k": 120, "bootstrap_n": 500,
         "split_bat": 0.14, "split_bowl": 0.08, "score_matches": 399, "within2": "60%", "within5": "92%"}
META = {}
if (RESULTS_DIR / "meta.json").exists():
    try:
        META = json.loads((RESULTS_DIR / "meta.json").read_text())
        FACTS.update({k: v for k, v in META.items() if k in FACTS})
    except Exception:
        META = {}
NOT_AVAILABLE = {"Moeen Ali", "Ajinkya Rahane"}
ARCH_KIND = {"Strong either way": "ok", "Lifts under pressure": "ok", "Dips under pressure": "warn",
             "Struggles under pressure": "warn", "Steady": ""}
ARCH_EPS, ARCH_DIP = 2.0, -5.0
RR_FACTS = [("Purse", "about ₹27 crore"), ("Vacancies", "6"), ("Overseas slots", "3")]
RELEASED_INDIANS = ["Nehal Wadhera", "Deepak Hooda", "M Shahrukh Khan", "Nitish Rana", "Rahul Tripathi",
                    "Umran Malik", "Deepak Chahar", "Tushar Deshpande", "Akash Madhwal", "Avesh Khan"]
NEED_ORDER = ["Top-order batter", "Middle order / finisher", "All-rounder", "Overseas spin", "Overseas pace", "Indian pace"]
NEEDS = {"Reeza Hendricks": ["Top-order batter"], "Kusal Mendis": ["Top-order batter"], "Pathum Nissanka": ["Top-order batter"],
         "Shai Hope": ["Top-order batter"], "Daryl Mitchell": ["Middle order / finisher"], "Jimmy Neesham": ["Middle order / finisher"],
         "Kamindu Mendis": ["All-rounder"], "Kyle Mayers": ["All-rounder"], "Sikandar Raza": ["All-rounder"],
         "Roston Chase": ["All-rounder"],
         "Wanindu Hasaranga": ["All-rounder", "Overseas spin"], "Keshav Maharaj": ["Overseas spin"],
         "Maheesh Theekshana": ["Overseas spin"], "Allah Ghazanfar": ["Overseas spin"], "Richard Ngarava": ["Overseas pace"],
         "Lockie Ferguson": ["Overseas pace"], "Joshua Little": ["Overseas pace"], "Sean Abbott": ["Overseas pace"],
         "Umran Malik": ["Indian pace"]}

def _compat(fn):
    def wrapper(*a, **kw):
        if "use_container_width" in kw:
            kw.pop("use_container_width")

            if fn.__name__ == "image":
                return fn(*a, use_column_width=True, **kw)

            return fn(*a, use_container_width=True, **kw)

        return fn(*a, **kw)

    return wrapper

for _n in ("plotly_chart", "dataframe"):
    setattr(st, _n, _compat(getattr(st, _n)))

CSS = Template("""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp,.stApp p,.stApp li,.stApp label,.stApp td,.stApp th,.stApp input,.stApp textarea,.stApp button{font-family:'Inter',system-ui,sans-serif}
[data-testid="stIconMaterial"]{font-family:'Material Symbols Rounded'!important}
html,body{background:$bg}
.stApp{background:radial-gradient(900px 420px at 100% -6%,rgba(255,92,154,.08),transparent 62%),$bg;color:$text;overflow-x:hidden}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important}
header[data-testid="stHeader"]{background:rgba(8,13,24,.82);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border-bottom:1px solid $line}
header[data-testid="stHeader"]::after{content:"SupeRR Selector S2  |  Pressure ratings";position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);font:600 .85rem 'Space Grotesk',sans-serif;color:$text;letter-spacing:.05em;white-space:nowrap}
.block-container{padding:4.2rem 1rem 3rem;max-width:1180px}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:'Space Grotesk',sans-serif!important;color:$text!important;letter-spacing:-.01em}
.stApp h1{font-size:clamp(1.7rem,4vw,2.3rem)}
.stApp hr{border-color:$line}
[data-testid="stCaptionContainer"]{color:$muted}
section[data-testid="stSidebar"]{background:$bg2;border-right:1px solid $line}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]{border-radius:10px;margin:2px 0;padding:.5rem .7rem;color:$muted;transition:background .15s,color .15s}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover{background:$elev;color:$text}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"]{background:rgba(255,92,154,.12);color:$pink_lt;box-shadow:inset 3px 0 0 $pink}
.brand b{display:block;font:700 1.1rem 'Space Grotesk',sans-serif;color:$text}
.brand span{color:$muted;font-size:.78rem}
.hero{position:relative;overflow:hidden;background:radial-gradient(520px 200px at 92% 0%,rgba(255,92,154,.20),transparent 65%),linear-gradient(135deg,$elev 0%,$bg2 100%);border:1px solid $line;border-radius:20px;padding:2rem 2rem 1.8rem;margin-bottom:1rem;box-shadow:0 10px 30px rgba(0,0,0,.35)}
.hero h1{color:$text!important;font-size:clamp(1.8rem,5vw,2.6rem);margin:.6rem 0 .5rem;line-height:1.1}
.hero p{color:$muted!important;max-width:44rem;margin:0;font-size:1.02rem;line-height:1.6}
.tag{display:inline-block;background:rgba(255,92,154,.14);border:1px solid rgba(255,92,154,.5);color:$pink_lt;font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;padding:.28rem .75rem;border-radius:999px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.8rem;margin:.2rem 0 1.2rem;align-items:stretch}
.kpi{display:flex;flex-direction:column;justify-content:space-between;min-height:108px;background:linear-gradient(180deg,$elev 0%,$card 100%);border:1px solid $line;border-radius:16px;padding:.95rem 1.1rem;box-shadow:0 6px 18px rgba(0,0,0,.28)}
.kpi small{display:block;color:$muted;font-size:.72rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase}
.kpi b{display:block;font:700 1.7rem/1.25 'Space Grotesk',sans-serif;color:$text}
.kpi span{color:$muted;font-size:.8rem}
.card{background:$card;border:1px solid $line;border-radius:16px;padding:.3rem 1.1rem;margin-bottom:.8rem;box-shadow:0 6px 18px rgba(0,0,0,.25)}
.card.pad{padding:1rem 1.2rem}
.card h4{margin:.2rem 0 .5rem;font-size:1.05rem}
.card p,.card li{color:$text;line-height:1.6}
.rk{display:grid;grid-template-columns:1.8rem 2.4rem 1fr;gap:.25rem .7rem;align-items:center;padding:.7rem 0;border-bottom:1px solid $line}
.rk:last-child{border-bottom:0}
.rk-n{font:700 1.15rem 'Space Grotesk',sans-serif;color:$text;text-align:right}
.av{display:inline-flex;width:2.4rem;height:2.4rem;border-radius:50%;align-items:center;justify-content:center;font-weight:700;font-size:.8rem;flex:none;border:1px solid rgba(255,255,255,.12)}
.av.bat{background:$blue;color:$text}
.av.bowl{background:$pink;color:$bg}
.rk-name{min-width:0}
.rk-name b{display:block;color:$text;font-size:.97rem;overflow-wrap:anywhere}
.rk-name span,.rk-bar span,.muted{color:$muted;font-size:.78rem}
.rk-bar{grid-column:3}
.bar{height:8px;border-radius:99px;background:#1B2A42;overflow:hidden;margin-bottom:.2rem}
.bar i{display:block;height:100%;border-radius:99px}
.bar i.bat{background:linear-gradient(90deg,$blue,$blue_lt)}
.bar i.bowl{background:linear-gradient(90deg,$pink,$pink_lt)}
.chip{display:inline-block;border-radius:999px;padding:.2rem .65rem;font-size:.74rem;font-weight:600;margin:.15rem .3rem .15rem 0;border:1px solid $line;background:$elev;color:$text}
.chip.warn{background:rgba(255,92,154,.12);border-color:rgba(255,92,154,.5);color:$pink_lt}
.chip.ok{background:rgba(73,104,157,.25);border-color:$blue;color:#C9D6EE}
.chip.bat{background:$blue;color:$text;border-color:$blue}
.chip.bowl{background:$pink;color:$bg;border-color:$pink}
.prof{display:flex;gap:1rem;align-items:center}
.prof h3{margin:0}
.xs{overflow-x:auto}
.cmp{width:100%;border-collapse:collapse;font-size:.86rem}
.cmp th{text-align:right;color:$muted;font-weight:700;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;padding:.4rem .3rem;border-bottom:1px solid $line;white-space:nowrap}
.cmp td{padding:.45rem .3rem;border-bottom:1px solid $grid;text-align:right;color:$text;font-variant-numeric:tabular-nums}
.cmp tr:hover td{background:rgba(73,104,157,.12)}
.cmp th:first-child,.cmp td:first-child{text-align:left}
.note{background:rgba(73,104,157,.16);border:1px solid $line;border-left:4px solid $blue;color:$text;border-radius:10px;padding:.75rem 1rem;margin:.6rem 0;font-size:.9rem;line-height:1.55}
.note.warn{background:rgba(255,92,154,.10);border-left-color:$pink}
[data-testid="stPlotlyChart"]{background:$card;border:1px solid $line;border-radius:16px;padding:.4rem;margin-bottom:.8rem;box-shadow:0 6px 18px rgba(0,0,0,.25)}
[data-testid="stImage"] img{border-radius:12px;border:1px solid $line;filter:invert(1) hue-rotate(180deg) brightness(.95) contrast(.92)}
[data-testid="stDataFrame"]{border:1px solid $line;border-radius:14px;overflow:hidden}
[data-baseweb="input"],[data-baseweb="select"]>div,[data-baseweb="textarea"]{background:$card!important;border:1px solid $line!important;border-radius:10px!important}
[data-baseweb="input"]:focus-within,[data-baseweb="select"]>div:focus-within{border-color:$pink!important}
[data-baseweb="input"] input{color:$text;min-height:44px}
[data-baseweb="popover"] [role="listbox"],[data-baseweb="menu"]{background:$elev!important;border:1px solid $line;border-radius:12px}
[data-baseweb="popover"] [role="option"]:hover{background:rgba(255,92,154,.14)}
[data-baseweb="tab-list"]{border-bottom:1px solid $line;gap:.4rem}
[data-baseweb="tab"]{color:$muted;min-height:44px}
[data-baseweb="tab"][aria-selected="true"]{color:$pink_lt}
[data-baseweb="tab-highlight"]{background:$pink}
[data-baseweb="tab-border"]{background:$line}
[data-testid="stExpander"] details{background:$card;border:1px solid $line;border-radius:14px}
[data-testid="stExpander"] summary{min-height:44px}
[data-testid="stDownloadButton"] button,.stButton button{background:$elev;border:1px solid $line;color:$text;border-radius:10px;min-height:44px;transition:border-color .15s,background .15s}
[data-testid="stDownloadButton"] button:hover,.stButton button:hover{border-color:$pink;color:$pink_lt;background:$card}
[data-testid="stRadio"] [role="radiogroup"]{gap:.5rem}
[data-testid="stRadio"] [role="radiogroup"] label{background:$card;border:1px solid $line;border-radius:999px;padding:.3rem .9rem;min-height:40px;align-items:center}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked){border-color:$pink;background:rgba(255,92,154,.12)}
[data-testid="stCheckbox"] label{min-height:40px;align-items:center}
[data-testid="stAlert"]{border-radius:12px;border:1px solid $line}
@media(max-width:900px){.block-container{padding:4rem .8rem 2.5rem}.hero{padding:1.5rem 1.3rem}}
@media(max-width:640px){
header[data-testid="stHeader"]::after{display:none}
.kpis{grid-template-columns:repeat(2,1fr)}
.kpi{min-height:96px;padding:.8rem .9rem}
.kpi b{font-size:1.4rem}
.rk{grid-template-columns:1.4rem 2.2rem 1fr;gap:.2rem .55rem}
.av{width:2.2rem;height:2.2rem;font-size:.75rem}
.rk-name b{font-size:.92rem}
.prof{flex-direction:column;align-items:flex-start}
.cmp{font-size:.8rem}
.card{padding:.3rem .8rem}
.card.pad{padding:.9rem .9rem}
}
""")

def inject_css():
    st.markdown(f"<style>{CSS.substitute(T)}</style>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def _read(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)

def load(name: str):
    p = RESULTS_DIR / f"{name}.csv"
    return _read(str(p), p.stat().st_mtime) if p.exists() else None

def need(*names: str) -> bool:
    missing = [n + ".csv" for n in names if load(n) is None]
    if missing:
        st.error(f"Missing file(s) in `{RESULTS_DIR.name}/`: {', '.join(missing)}. Add them and reload.")
    return not missing

def last_updated() -> str:
    if META.get("last_updated"):
        return str(META["last_updated"])
    t = [p.stat().st_mtime for p in RESULTS_DIR.glob("*") if p.is_file()]
    return datetime.fromtimestamp(max(t)).strftime("%d %b %Y") if t else "n/a"

def get_pool(role: str):
    if not need(f"{role}_ratings", f"{role}_bootstrap"):
        return None
    r = load(f"{role}_ratings").rename(columns={"rank": "main_rank"}).copy()
    b = load(f"{role}_bootstrap").copy()
    for d in (r, b):
        d["player_id"] = d["player_id"].astype(float).astype(int)
    stale = sorted(set(r["name"]) & NOT_AVAILABLE)
    if stale:
        st.error(f"Results still contain unavailable player(s): {', '.join(stale)}. Re-run the pipeline with them excluded and replace the files in `{RESULTS_DIR.name}/`.")
    m = r.merge(b[["player_id", "p_top10", "p_top5", "mean_rank", "rank_p05", "rank_p95"]], on="player_id", how="left")
    m = m.sort_values(["mean_rank", "main_rank"]).reset_index(drop=True)
    m.insert(0, "final_rank", np.arange(1, len(m) + 1))
    for c in ("pn_ipl", "pn_major", "pn_assoc"):
        m[c] = m[c].fillna(0) if c in m else 0
    m["p_top10_pct"], m["p_top5_pct"] = m["p_top10"] * 100, m["p_top5"] * 100
    m["ipl_share_pct"] = m["ipl_share"] * 100 if "ipl_share" in m else np.nan
    m["rank_range"] = [f"{a:.0f}–{c:.0f}" if pd.notna(a) else "n/a" for a, c in zip(m["rank_p05"], m["rank_p95"])]
    return m

H = lambda s: " ".join(s.split())
esc = html.escape
def md(s): st.markdown(H(s), unsafe_allow_html=True)

def initials(n: str) -> str:
    t = n.split()
    return (t[0][0] + t[-1][0]).upper() if len(t) > 1 else n[:2].upper()

def avatar(name, color):
    cls = "bowl" if color == ROLE["bowl"]["color"] else "bat"
    return f'<span class="av {cls}">{esc(initials(name))}</span>'
def chip(text, kind=""): return f'<span class="chip {kind}">{esc(text)}</span>'

def kpis(items):
    md('<div class="kpis">' + "".join(f"<div class='kpi'><small>{esc(a)}</small><b>{esc(b)}</b><span>{esc(c)}</span></div>"
                                      for a, b, c in items) + "</div>")

def archetype(p, b):
    hp, hb = p > ARCH_EPS, b > ARCH_EPS
    if hp and hb:
        return ("Dips under pressure", "Above average, but clearly below their own usual level.") if p - b <= ARCH_DIP \
            else ("Strong either way", "Above average under pressure and in normal play.")
    if hp:
        return "Lifts under pressure", "Above average under pressure, not in normal play."
    if hb:
        return "Dips under pressure", "Above average in normal play, not under pressure."
    if p < -ARCH_EPS:
        return "Struggles under pressure", "Below average under pressure, not strong otherwise."
    return "Steady", "Close to average in both situations."

def sample_flags(r):
    out = []
    if r["pn_ipl"] == 0: out.append(("No IPL pressure balls", "warn"))
    if r["p_matches"] < 15: out.append(("Under 15 pressure matches", "warn"))
    tot = r["pn_ipl"] + r["pn_major"] + r["pn_assoc"]
    if tot and r["pn_assoc"] / tot > 0.5: out.append(("Over half of pressure balls vs associate teams", "warn"))
    return out or [("No sample warnings", "ok")]

def tie_note(pool, n=10):
    if len(pool) < n or pool["mean_rank"].isna().any():
        return ""
    m = pool.loc[n - 1, "mean_rank"]
    near = pool[(pool["mean_rank"] - m).abs() <= 1.0]
    lo, hi = int(near["final_rank"].min()), int(near["final_rank"].max())
    return (f"Places {lo}–{hi} are within one rank point of 10th place (mean rank {near['mean_rank'].min():.1f}–"
            f"{near['mean_rank'].max():.1f}), so their order is not meaningful and the top-10 cut is soft.") if hi > lo else ""

def ranked_list(pool, role, n=10):
    c, rows = ROLE[role]["color"], []
    for _, r in pool.head(n).iterrows():
        pct = 0 if pd.isna(r["p_top10"]) else r["p_top10"] * 100
        rows.append(f"""<div class="rk"><div class="rk-n">{int(r['final_rank'])}</div>{avatar(r['name'], c)}
          <div class="rk-name"><b>{esc(r['name'])}</b><span>Pressure {r['p_rating']:+.1f} | Usual {r['base_rating']:+.1f} (runs per 100 balls)</span></div>
          <div class="rk-bar"><div class="bar"><i class="{role}" style="width:{pct:.0f}%"></i></div>
          <span>{pct:.0f}% chance of top 10 | rank range {r['rank_range']}</span></div></div>""")
    return '<div class="card">' + "".join(rows) + "</div>"

def on_color(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return T["bg"] if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else T["text"]

def style_fig(fig, title, h=360):
    fig.update_layout(title=dict(text=title, font=dict(size=16, color=T["text"], family="Space Grotesk")), height=h,
                      font=dict(family="Inter", size=13, color=T["text"]), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=55, b=60), legend=dict(orientation="h", y=-0.22, font=dict(color=T["text"])),
                      hoverlabel=dict(bgcolor=T["elev"], bordercolor=T["line"], font=dict(color=T["text"], family="Inter")))
    fig.update_xaxes(gridcolor=T["grid"], zerolinecolor=T["line"], linecolor=T["line"], tickfont=dict(color=T["muted"]), title_font=dict(color=T["muted"]))
    fig.update_yaxes(gridcolor=T["grid"], zerolinecolor=T["line"], linecolor=T["line"], tickfont=dict(color=T["muted"]), title_font=dict(color=T["muted"]))
    fig.update_annotations(font=dict(color=T["muted"]))
    return fig

def show(fig): st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def show_img(name, caption):
    p = RESULTS_DIR / name
    if p.exists():
        st.image(str(p), caption=caption)
    else:
        st.info(f"Chart `{name}` not found.")

def footer():
    st.divider()

def table(df, key, filename, cfg=None, select=False):
    kw = dict(hide_index=True, column_config=cfg or {}, use_container_width=True)
    if select:
        kw.update(on_select="rerun", selection_mode="single-row", key=f"tbl_{key}")
    ev = st.dataframe(df, **kw)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), file_name=filename, mime="text/csv", key=f"dl_{key}")
    return ev

def page_overview():
    bat, bowl = get_pool("bat"), get_pool("bowl")
    md("""<div class="hero"><span class="tag">Provisional until auction registration closes</span>
      <h1>Who holds up under pressure?</h1>
      <p>We measure how IPL and T20I batters and bowlers perform in high-pressure moments compared with their own usual level,
      then rank the players Rajasthan Royals could actually bid for at the IPL 2027 auction (16 Dec 2026, Goa).</p></div>""")
    if bat is None or bowl is None:
        return
    kpis([("Players ranked", f"{len(bat) + len(bowl)}", f"{len(bat)} batters, {len(bowl)} bowlers"),
          ("Matches analysed", f"{FACTS['matches']:,}", "IPL and T20Is, 2022 onward"),
          ("Deliveries", f"{FACTS['deliveries']:,}", "ball-by-ball"),
          ("Split-half reliability", f"{FACTS['split_bat']:.2f} / {FACTS['split_bowl']:.2f}", "batters / bowlers: noisy, a shortlist")])
    a, b = st.columns(2)
    for col, pool, role in ((a, bat, "bat"), (b, bowl, "bowl")):
        with col:
            md(f"<h3 style='margin-bottom:.4rem'>Top 10 {ROLE[role]['label'].lower()}</h3>")
            md(ranked_list(pool, role))
            if tie_note(pool):
                md(f"<div class='note warn'>{esc(tie_note(pool))}</div>")
    st.caption(f"Order = mean rank over {FACTS['bootstrap_n']} match-level bootstrap resamples. Ratings are runs above expectation per 100 balls.")
    footer()

def archetype_card(r, role):
    label, why = archetype(r["p_rating"], r["base_rating"])
    team = f"Latest IPL team in the data (historical): {esc(str(r['latest_ipl_team']))}" if pd.notna(r.get("latest_ipl_team")) else ""
    chips = chip(ROLE[role]["label"][:-1], role) + chip(label, ARCH_KIND.get(label, "")) + "".join(chip(t, k) for t, k in sample_flags(r))
    md(f"""<div class="card pad"><div class="prof">{avatar(r['name'], ROLE[role]['color'])}<div><h3>{esc(r['name'])}</h3>
        <div>{chips}</div><span class="muted">{why} {team}</span></div></div></div>""")

def page_players():
    st.title("Players")
    role_label = st.radio("Role", ["Batters", "Bowlers"], horizontal=True, label_visibility="collapsed")
    role = "bat" if role_label == "Batters" else "bowl"
    pool = get_pool(role)
    if pool is None:
        return
    q = st.text_input("Search player", placeholder="Type part of a name")
    sub = pool[pool["name"].str.contains(q, case=False, na=False)].reset_index(drop=True) if q else pool
    cols = {"final_rank": "Final #", "name": "Player", "p_rating": "Pressure rating", "base_rating": "Usual rating",
            "delta": "Gap (pressure - usual)", "consistency": "Consistency", "p_n": "Pressure balls", "p_matches": "Pressure matches",
            "ipl_share_pct": "IPL share (%)", "p_top10_pct": "P(top 10) (%)", "p_top5_pct": "P(top 5) (%)", "rank_range": "Rank range"}
    cfg = {"P(top 10) (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
           "Pressure rating": st.column_config.NumberColumn(format="%.1f"), "Usual rating": st.column_config.NumberColumn(format="%.1f"),
           "Gap (pressure - usual)": st.column_config.NumberColumn(format="%.1f"), "Consistency": st.column_config.NumberColumn(format="%.2f"),
           "IPL share (%)": st.column_config.NumberColumn(format="%.0f"), "P(top 5) (%)": st.column_config.NumberColumn(format="%.0f")}
    ev = table(sub[list(cols)].rename(columns=cols), f"pl_{role}", f"{role}_pool.csv", cfg, select=True)
    st.caption("Click a row's checkbox to open that player, or pick from the list below. Click a header to sort.")
    chosen = st.selectbox("Player profile", pool["name"].tolist())
    rows = getattr(getattr(ev, "selection", None), "rows", []) or []
    if rows and rows[0] < len(sub):
        chosen = sub.iloc[rows[0]]["name"]
    r = pool[pool["name"] == chosen].iloc[0]
    archetype_card(r, role)
    kpis([("Final rank", f"#{int(r['final_rank'])}", f"of {len(pool)} ranked"), ("P(top 10)", f"{r['p_top10']:.0%}", f"P(top 5) {r['p_top5']:.0%}"),
          ("Consistency", f"{r['consistency']:.2f}", "share of positive pressure matches"),
          ("Pressure sample", f"{int(r['p_n'])} balls", f"{int(r['p_matches'])} pressure matches")])
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=["Usual (normal pressure)", "Under pressure"], y=[r["base_rating"], r["p_rating"]],
                               marker_color=[ROLE[role]["color"], ROLE[role]["light"]],
                               text=[f"{r['base_rating']:+.1f}", f"{r['p_rating']:+.1f}"], textposition="outside", textfont=dict(color=T["text"]),
                               hovertemplate=f"{esc(r['name'])}<br>%{{x}}: %{{y:.1f}} runs per 100 balls<extra></extra>"))
        fig.update_yaxes(title="Runs above average per 100 balls")
        fig.update_xaxes(title="Situation")
        show(style_fig(fig, "Pressure vs usual", 340))
    with c2:
        fig = go.Figure()
        if pd.notna(r["rank_p05"]):
            fig.add_trace(go.Scatter(x=[r["rank_p05"], r["rank_p95"]], y=[0, 0], mode="lines", name="5th–95th percentile",
                                     line=dict(color=ROLE[role]["color"], width=12), hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[r["mean_rank"]], y=[0], mode="markers", name="Mean rank",
                                     marker=dict(color=ROLE[role]["light"], size=16, symbol="diamond", line=dict(color=T["bg"], width=1)),
                                     hovertemplate=f"Mean rank %{{x:.1f}} (range {r['rank_range']})<extra></extra>"))
        fig.update_xaxes(title="Rank in pool (1 = best)", range=[0.5, len(pool) + 0.5])
        fig.update_yaxes(visible=False)
        show(style_fig(fig, "Rank range across bootstrap resamples", 340))
    tot = r["pn_ipl"] + r["pn_major"] + r["pn_assoc"]
    if tot:
        fig = go.Figure()
        for lab, v, col in zip(["IPL", "T20I major", "T20I associate"], [r["pn_ipl"], r["pn_major"], r["pn_assoc"]], ROLE[role]["tints"]):
            fig.add_trace(go.Bar(y=[""], x=[v / tot * 100], orientation="h", name=lab, marker_color=col,
                                 text=[f"{lab} {v / tot:.0%}" if v / tot > 0.08 else ""], textposition="inside", textfont=dict(color=on_color(col)),
                                 hovertemplate=f"{lab}: %{{x:.0f}}% ({int(v)} balls)<extra></extra>"))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title="Share of this player's pressure balls (%)", range=[0, 100])
        show(style_fig(fig, "Where the pressure evidence comes from", 200))
    with st.expander("Pool map: pressure vs normal performance"):
        show_img(f"chart_{role}_pressure_vs_normal.png", "Above the dashed line = better under pressure than usual. Orange = top 10 by main rank.")
    footer()

VAL_LABELS = {"deliveries": "Deliveries", "mean_over": "Average over", "wkts_down": "Wickets down", "win_prob_closeness": "Closeness to 50/50",
              "runs_vs_par": "Runs vs par", "wkts_last_12_balls": "Wickets, last 12 balls", "runs_last_12_balls": "Runs, last 12 balls",
              "mean_leverage": "Leverage per ball", "runs_needed": "Runs needed", "balls_left": "Balls left", "req_rate": "Required rate"}

def compare_card(title, df, innings, role):
    d = df[df["innings"] == innings].set_index("pressure").drop(columns="innings").dropna(axis=1, how="all")
    rows = ""
    for col, lab in VAL_LABELS.items():
        if col not in d.columns: continue
        n, h = d.loc["normal", col], d.loc["HIGH", col]
        rows += (f"<tr><td>{lab}</td><td>{n:,.0f}</td><td>{h:,.0f}</td><td></td></tr>" if col == "deliveries"
                 else f"<tr><td>{lab}</td><td>{n:.2f}</td><td>{h:.2f}</td><td>{h - n:+.2f}</td></tr>")
    md(f"""<div class="card pad">{chip(title, role)}<div class="xs"><table class="cmp"><tr><th>Measure</th><th>Normal</th><th>High pressure</th><th>Change</th></tr>{rows}</table></div></div>""")

def page_pressure():
    st.title("Pressure and validation")
    md(f"""<div class="card pad"><h4>How pressure is defined</h4>
      <p><b>Chases:</b> win-probability leverage, the expected swing in win probability on the next ball (gradient-boosted classifier,
      log-loss {FACTS['wp_logloss']} vs {FACTS['coin_logloss']} for a coin flip).<br>
      <b>First innings, by role:</b> batters face collapse and runs below par; bowlers are attacked and concede runs above par.<br>
      <b>High pressure</b> = the top 25% within each innings and role.</p></div>""")
    if not need("pressure_validation_batting", "pressure_validation_bowling"):
        return
    vb, vw = load("pressure_validation_batting"), load("pressure_validation_bowling")
    st.subheader("High vs normal pressure")
    st.caption("Change = high pressure minus normal. These checks show the flagged moments really are different: more wickets down, "
               "more runs needed per ball, and closer games in chases.")
    a, b, c = st.columns(3)
    with a: compare_card("1st innings: batters", vb, 1, "bat")
    with b: compare_card("1st innings: bowlers", vw, 1, "bowl")
    with c: compare_card("Chase: both roles", vb, 2, "bat")
    show_img("chart_pressure_by_over.png", "Share of deliveries flagged high-pressure, by over (IPL + T20Is).")
    footer()

def heatmap(df, role):
    d = df.set_index("name").rename(columns={"base": "Main"})
    z = d.values.astype(float)
    scale = ([[0, "#C9D6EE"], [0.5, T["blue"]], [1, "#16233A"]] if role == "bat" else [[0, "#FFD3E3"], [0.5, T["pink"]], [1, "#3A1A2C"]])
    fig = go.Figure(go.Heatmap(z=z, x=list(d.columns), y=list(d.index), colorscale=scale, zmin=1, zmax=np.nanmax(z), hoverongaps=False,
                               text=[["–" if pd.isna(v) else f"{v:.0f}" for v in row] for row in z], texttemplate="%{text}",
                               colorbar=dict(title="Rank"), hovertemplate="%{y}<br>%{x}: rank %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Player (top 10 by main rank)")
    fig.update_xaxes(side="top", title="Model variant")
    return style_fig(fig, f"{ROLE[role]['label']}: rank under each variant (brighter = better)", 470)

def boot_chart(pool, role, n):
    d, c = pool.head(n).iloc[::-1], ROLE[role]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=("Chance of finishing in top 10 / top 5", "Rank range (5th–95th percentile)"))
    fig.add_trace(go.Bar(y=d["name"], x=d["p_top10_pct"], orientation="h", name="P(top 10)", marker_color=c["light"]), 1, 1)
    fig.add_trace(go.Bar(y=d["name"], x=d["p_top5_pct"], orientation="h", name="P(top 5)", marker_color=c["color"]), 1, 1)
    fig.add_trace(go.Scatter(x=d["mean_rank"], y=d["name"], mode="markers", name="Mean rank", marker=dict(color=c["light"], size=9),
                             error_x=dict(type="data", symmetric=False, thickness=2, color=T["muted"],
                                          array=np.clip(d["rank_p95"] - d["mean_rank"], 0, None), arrayminus=np.clip(d["mean_rank"] - d["rank_p05"], 0, None))), 1, 2)
    fig.update_xaxes(title_text="Probability (%)", range=[0, 100], row=1, col=1)
    fig.update_xaxes(title_text="Rank in pool (1 = best)", row=1, col=2)
    fig.update_yaxes(categoryorder="array", categoryarray=list(d["name"]))
    fig.update_layout(barmode="group")
    return style_fig(fig, f"{c['label']}: how sure are we?", 150 + 30 * len(d))

def page_reliability():
    st.title("Reliability")
    kpis([("Split-half, batters", f"{FACTS['split_bat']:.2f}", "correlation between data halves"),
          ("Split-half, bowlers", f"{FACTS['split_bowl']:.2f}", "correlation between data halves"),
          ("Bootstrap resamples", f"{FACTS['bootstrap_n']}", "match-level"), ("Stability variants", "8", "cut-offs, weights, shrinkage")])
    md("<div class='note'>Values near 0 mean pressure form in one half of the data says little about the other. Treat the lists as shortlists, not verdicts.</div>")
    st.subheader("Rank stability across 8 variants")
    if need("bat_rank_stability", "bowl_rank_stability"):
        tabs = st.tabs(["Batters", "Bowlers"])
        for tab, role in zip(tabs, ("bat", "bowl")):
            with tab:
                df = load(f"{role}_rank_stability")
                show(heatmap(df, role))
                st.caption("Numbers are ranks; '–' = not ranked in that variant. Variants: other pressure cut-offs, chase-only, equal weights, no gap term, shrinkage k/2 and 2k.")
                table(df, f"rs_{role}", f"{role}_rank_stability.csv")
    st.subheader("Bootstrap uncertainty")
    label = st.radio("Role", ["Batters", "Bowlers"], horizontal=True, key="boot_role", label_visibility="collapsed")
    role = "bat" if label == "Batters" else "bowl"
    pool = get_pool(role)
    if pool is not None:
        n = st.slider("Players shown (by final order)", 5, len(pool), min(15, len(pool)))
        show(boot_chart(pool, role, n))
    footer()

def page_rr():
    st.title("Auction fit for RR")
    md("<div class='note warn'><b>Needs are our reading, not RR statements.</b> Roles and nationalities below are our own classification, not from the data files.</div>")
    kpis([(a, b, "our reading") for a, b in RR_FACTS])
    st.markdown("**Released:** Hetmyer, Milne, Maphaka, Deshpande, Kuldeep Sen. **Trade:** Shami in, Curran out.")
    bat, bowl = get_pool("bat"), get_pool("bowl")
    if bat is None or bowl is None:
        return
    entries = [(row, "bat") for _, row in bat.head(10).iterrows()] + [(row, "bowl") for _, row in bowl.head(10).iterrows()]
    for need_name in NEED_ORDER:
        lines = "".join(
            f"<div class='rk' style='grid-template-columns:2.4rem 1fr'>{avatar(r['name'], ROLE[role]['color'])}<div class='rk-name'><b>{esc(r['name'])}</b>"
            f"<span>{ROLE[role]['label'][:-1]} #{int(r['final_rank'])} | pressure {r['p_rating']:+.1f} | P(top 10) {r['p_top10']:.0%}</span></div></div>"
            for r, role in entries if need_name in NEEDS.get(r["name"], []))
        if lines:
            md(f"<div class='card'><h4 style='margin-top:.6rem'>{esc(need_name)}</h4>{lines}</div>")
    missing = sorted({r["name"] for r, _ in entries if r["name"] not in NEEDS})
    if missing:
        st.caption("Top-10 players not yet classified for this page (add them to NEEDS in app.py): " + ", ".join(missing))
    md("<div class='note'>Umran Malik is the only Indian in the top 20 of our ranked pool and uses no overseas slot.</div>")
    footer()

def _tokens(s): return re.sub(r"[^a-z\s]", " ", s.lower()).split()

def same_person(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb or ta[-1] != tb[-1]: return False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return all(any(t == s or (len(s) <= 2 and t.startswith(s[0])) for t in long_[:-1]) for s in short[:-1])

def page_released():
    st.title("Released Indian players")
    bat, bowl = get_pool("bat"), get_pool("bowl")
    if bat is None or bowl is None:
        return
    rows = []
    for nm in RELEASED_INDIANS:
        found = False
        for role, pool in (("Batter", bat), ("Bowler", bowl)):
            hit = pool[[same_person(nm, x) for x in pool["name"]]]
            if len(hit) == 1:
                r, found = hit.iloc[0], True
                rows.append({"Player": nm, "Pool": role, "Final #": int(r["final_rank"]), "Pressure rating": round(r["p_rating"], 1),
                             "P(top 10) (%)": round(r["p_top10_pct"]), "Status": "Top 10" if r["final_rank"] <= 10 else "Ranked"})
        if not found:
            rows.append({"Player": nm, "Pool": "-", "Final #": None, "Pressure rating": None, "P(top 10) (%)": None, "Status": "Not ranked"})
    df = pd.DataFrame(rows)
    top = df[df["Status"] == "Top 10"]
    md(f"<div class='note'>{'None of these players is in the top 10 of either list.' if top.empty else 'In the top 10: ' + esc(', '.join(top['Player'] + ' (' + top['Pool'].str.lower() + 's)')) + '. The others are ranked lower or not ranked.'}</div>")
    table(df, "released", "released_indians.csv", {"P(top 10) (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})
    st.caption("Matched by name to the ranked pools. 'Not ranked' = outside the pool or below the minimum sample (120 high-pressure balls, 10 pressure matches).")
    footer()

def page_method():
    st.title("Method and limitations")
    wc = FACTS["wicket_cost"]
    md(f"""<div class="card pad"><h4>Pool</h4><p>Players outside every IPL 2027 squad after the 17 Nov releases. Retired, opted-out, Pakistani and Bangladeshi players are excluded.
      Minimum 120 high-pressure balls and 10 pressure matches. IPL and T20Is from 2022 onward; super overs dropped.</p></div>
      <div class="card pad"><h4>How the ranking works</h4><ol>
      <li><b>Pressure.</b> Chases use win-probability leverage (log-loss {FACTS['wp_logloss']} vs {FACTS['coin_logloss']} for a coin flip). The first innings is role-specific: batters face collapse and runs below par, bowlers are attacked and concede runs above par. High pressure = top 25% within innings and role.</li>
      <li><b>Ball value.</b> Runs above context expectation plus a wicket cost (powerplay {wc['Powerplay']}, middle {wc['Middle']}, death {wc['Death']} runs).</li>
      <li><b>Pressure rating.</b> Runs above expectation per 100 high-pressure balls, shrunk toward a prior from the player's own usual level (weight {FACTS['shrinkage_k']}).</li>
      <li><b>Consistency.</b> Share of pressure matches with positive value, smoothed.</li>
      <li><b>Score</b> = 0.55 x z(pressure rating) + 0.15 x z(pressure minus usual) + 0.30 x z(consistency).</li>
      <li><b>Final order</b> = mean rank over {FACTS['bootstrap_n']} match-level bootstrap resamples, with P(top 10) and a 5th–95th percentile rank range.</li>
      <li><b>Reliability.</b> Split-half {FACTS['split_bat']:.2f} (batters) and {FACTS['split_bowl']:.2f} (bowlers); 8 stability variants.</li></ol></div>
      <div class="card pad"><h4>Limitations</h4><ul>
      <li>Extras are assumed at 1 run. Rebuilt scores match official margins within 2 runs in {FACTS['within2']} and within 5 runs in {FACTS['within5']} of {FACTS['score_matches']} matches.</li>
      <li>No opposition or venue adjustment.</li><li>The pool is thin and some samples are small.</li>
      <li>Availability is not guaranteed. Players known to be unavailable for IPL 2027 (for example Moeen Ali) were removed from the pool.</li>
      <li>Pressure form is noisy (see Reliability), so these are shortlists, not verdicts.</li></ul></div>""")
    footer()

inject_css()
with st.sidebar:
    md("<div class='brand' style='padding:.2rem 0 .8rem'><b>SupeRR Selector S2</b><span>Who holds up under pressure?</span><div style='margin-top:.7rem'><span class='tag'>Provisional</span></div><p class='muted' style='margin:.7rem 0 0'>Built for the SupeRR Selector, Rajasthan Royals</p></div>")
PAGES = [st.Page(page_overview, title="Overview", icon=":material/home:", url_path="overview", default=True),
         st.Page(page_players, title="Players", icon=":material/groups:", url_path="players"),
         st.Page(page_pressure, title="Pressure and validation", icon=":material/speed:", url_path="pressure"),
         st.Page(page_reliability, title="Reliability", icon=":material/verified:", url_path="reliability"),
         st.Page(page_rr, title="Auction fit for RR", icon=":material/target:", url_path="rr-fit"),
         st.Page(page_released, title="Released Indian players", icon=":material/flag:", url_path="released"),
         st.Page(page_method, title="Method and limitations", icon=":material/menu_book:", url_path="method")]
st.navigation(PAGES).run()