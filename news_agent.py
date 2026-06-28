"""
News Agent — Market Intelligence
=================================
Fetches news for open positions + macro + AI/semi sector.
Sources: yfinance (per-symbol) + RSS feeds (Reuters/MarketWatch/CNBC — same as MSN aggregates)
Output:  latest_news.md committed to repo — readable on GitHub anytime

Run on demand:  python news_agent.py
Deps:           pip install yfinance alpaca-py pytz feedparser
"""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from email.utils import parsedate_to_datetime

import feedparser
import yfinance as yf
import pytz

from alpaca.trading.client import TradingClient

# ── Config ─────────────────────────────────────────────────────────────────────
HOURS_BACK   = 24
OUT_FILE     = "latest_news.md"   # committed to repo — readable on GitHub

MACRO_SYMS  = ["SPY", "QQQ", "^VIX", "SMH", "XLK"]
SECTOR_SYMS = ["NVDA", "AMD", "TSM", "MRVL", "AVGO"]

# RSS feeds — same sources MSN aggregates (free, no API key)
RSS_FEEDS = {
    "Reuters":    "https://feeds.reuters.com/reuters/businessNews",
    "MarketWatch":"http://feeds.marketwatch.com/marketwatch/topstories/",
    "CNBC":       "https://www.cnbc.com/id/100003114/device/rss/rss.html",
}

ACCOUNTS = {
    "momentum":  (os.environ.get("ALPACA_ACC1_KEY", ""), os.environ.get("ALPACA_ACC1_SECRET", "")),
    "dynamick":  (os.environ.get("ALPACA_ACC2_KEY", ""), os.environ.get("ALPACA_ACC2_SECRET", "")),
    "minervini": (os.environ.get("ALPACA_ACC3_KEY", ""), os.environ.get("ALPACA_ACC3_SECRET", "")),
}

CT = pytz.timezone("America/Chicago")


# ── Data fetchers ──────────────────────────────────────────────────────────────

def get_open_positions():
    """Returns {symbol: (account_name, entry_price, unreal_pl_pct)}. Skips if no keys set."""
    positions = {}
    for name, (k, s) in ACCOUNTS.items():
        if not k or not s:
            continue
        try:
            client = TradingClient(k, s, paper=True)
            for p in client.get_all_positions():
                positions[p.symbol] = {
                    "account":  name,
                    "entry":    float(p.avg_entry_price),
                    "pl_pct":   float(p.unrealized_plpc) * 100,
                    "pl_dollar": float(p.unrealized_pl),
                }
        except Exception as e:
            print(f"  {name} fetch error: {e}")
    return positions


def get_price_context(symbols):
    """Returns {symbol: {price, change_pct}}."""
    ctx = {}
    for sym in symbols:
        try:
            info  = yf.Ticker(sym).fast_info
            price = info.last_price
            prev  = info.previous_close
            chg   = (price / prev - 1) * 100 if prev else 0.0
            ctx[sym] = {"price": round(price, 2), "change_pct": round(chg, 2)}
        except Exception:
            pass
    return ctx


def fetch_news(symbols, hours_back=24):
    """Returns {symbol: [articles]} deduped by title, filtered to last N hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    seen   = set()
    result = defaultdict(list)

    for sym in symbols:
        try:
            raw = yf.Ticker(sym).news or []
            for item in raw:
                # yfinance v0.2.x+ nests everything under 'content'
                c = item.get("content", item)

                # parse publish time — ISO string or unix int
                pub_raw = c.get("pubDate") or c.get("providerPublishTime", 0)
                if isinstance(pub_raw, str):
                    pub = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
                else:
                    pub = datetime.fromtimestamp(int(pub_raw), tz=timezone.utc)

                if pub < cutoff:
                    continue

                title = (c.get("title") or "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)

                # link: nested canonicalUrl or clickThroughUrl
                link = (
                    (c.get("canonicalUrl") or {}).get("url")
                    or (c.get("clickThroughUrl") or {}).get("url")
                    or c.get("link", "#")
                )
                source = (
                    (c.get("provider") or {}).get("displayName")
                    or c.get("publisher", "")
                )

                result[sym].append({
                    "title":  title,
                    "link":   link,
                    "pub":    pub.astimezone(CT).strftime("%I:%M %p CT").lstrip("0"),
                    "source": source,
                })
        except Exception as e:
            print(f"  News fetch failed {sym}: {e}")

    return result


# ── RSS fetcher (Reuters / MarketWatch / CNBC) ────────────────────────────────

def fetch_rss_news(hours_back=24):
    """Returns list of {title, link, pub, source} from RSS feeds, last N hours."""
    cutoff  = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    articles = []
    seen     = set()

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                # parse published date
                pub = None
                if hasattr(entry, "published"):
                    try:
                        pub = parsedate_to_datetime(entry.published)
                    except Exception:
                        pass
                if pub is None:
                    continue
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                if pub < cutoff:
                    continue
                title = (entry.get("title") or "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                articles.append({
                    "title":  title,
                    "link":   entry.get("link", "#"),
                    "pub":    pub.astimezone(CT).strftime("%I:%M %p CT").lstrip("0"),
                    "source": source,
                })
        except Exception as e:
            print(f"  RSS fetch failed {source}: {e}")

    articles.sort(key=lambda x: x["pub"], reverse=True)
    return articles


# ── Markdown builder ───────────────────────────────────────────────────────────

def _price_str(sym, ctx):
    if sym not in ctx:
        return ""
    p    = ctx[sym]
    sign = "+" if p["change_pct"] >= 0 else ""
    return f"${p['price']}  {sign}{p['change_pct']:.2f}%"


def build_markdown(positions, price_ctx, pos_news, macro_news, sector_news):
    now_str   = datetime.now(CT).strftime("%A %B %d %Y — %I:%M %p CT").lstrip("0")
    total_pl  = sum(p["pl_dollar"] for p in positions.values())
    pl_sign   = "+" if total_pl >= 0 else ""
    vix       = price_ctx.get("^VIX", {}).get("price", "—")
    regime    = "CAUTION ⚠" if float(vix or 0) > 20 else "OK"

    lines = []
    lines.append(f"# Market News — {now_str}")
    lines.append(f"\n**SPY** {_price_str('SPY', price_ctx)}  |  "
                 f"**VIX** {vix}  |  **Regime** {regime}  |  "
                 f"**Portfolio P&L** {pl_sign}${total_pl:.0f}  |  "
                 f"**Positions** {len(positions)}")

    # Alerts
    alerts = [f"{s} {price_ctx[s]['change_pct']:+.1f}%"
              for s in positions if price_ctx.get(s, {}).get("change_pct", 0) <= -3]
    if alerts:
        lines.append(f"\n> **ALERT — Down >3% today:** {', '.join(alerts)}")

    # Positions
    lines.append("\n---\n## Your Open Positions")
    for sym in sorted(positions.keys()):
        info = positions[sym]
        p    = price_ctx.get(sym, {})
        chg  = p.get("change_pct", 0)
        flag = " ⚠" if chg <= -3 else (" ↓" if chg <= -1 else "")
        lines.append(f"\n### {sym}{flag}  —  {_price_str(sym, price_ctx)}  "
                     f"| P&L: {info['pl_dollar']:+.0f} ({info['pl_pct']:+.1f}%)  | {info['account']}")
        arts = pos_news.get(sym, [])
        if arts:
            for a in arts[:5]:
                lines.append(f"- [{a['title']}]({a['link']}) — *{a['source']} {a['pub']}*")
        else:
            lines.append("- *No recent news*")

    # Macro
    lines.append("\n---\n## Market Pulse")
    for sym in MACRO_SYMS:
        display = sym.replace("^", "")
        lines.append(f"\n### {display}  {_price_str(sym, price_ctx)}")
        for a in macro_news.get(sym, [])[:3]:
            lines.append(f"- [{a['title']}]({a['link']}) — *{a['source']} {a['pub']}*")

    # Sector
    lines.append("\n---\n## AI / Semiconductor Sector")
    found = False
    for sym in SECTOR_SYMS:
        arts = sector_news.get(sym, [])
        if not arts:
            continue
        found = True
        lines.append(f"\n### {sym}  {_price_str(sym, price_ctx)}")
        for a in arts[:3]:
            lines.append(f"- [{a['title']}]({a['link']}) — *{a['source']} {a['pub']}*")
    if not found:
        lines.append("\n*No sector news in last 24h*")

    return "\n".join(lines)


# Keywords that flag a story as market-moving — shown at top in ALERTS block
PRIORITY_KEYWORDS = [
    # macro structure
    "s&p 500", "s&p500", "index change", "index rebalance", "added to s&p", "removed from s&p",
    # fed / rates
    "federal reserve", "fed rate", "interest rate", "powell", "rate cut", "rate hike", "inflation",
    # geopolitical risk-off
    "iran", "russia", "china tariff", "war", "strike", "sanctions", "missile",
    # tech breakthroughs that move semis/AI
    "quantum", "10x faster", "100x faster", "nvidia killer", "nvidia challenger",
    "breakthrough", "new chip", "new model", "ipo", "fda approval",
    # earnings / guidance
    "earnings miss", "earnings beat", "guidance cut", "guidance raise", "revenue miss",
    # sector-specific
    "semiconductor", "ai chip", "data center", "anthropic", "openai", "deepseek",
]

def _is_priority(title):
    t = title.lower()
    return any(kw in t for kw in PRIORITY_KEYWORDS)


def add_rss_section(lines, rss_articles):
    """Append RSS section with priority alerts at top."""
    if not rss_articles:
        lines.append("\n---\n## Market Headlines  *(Reuters · MarketWatch · CNBC)*")
        lines.append("\n*No RSS news in last 24h*")
        lines.append(f"\n---\n*Generated {datetime.now(CT).strftime('%Y-%m-%d %H:%M CT')} "
                     f"| last {HOURS_BACK}h | sources: yfinance + Reuters/MarketWatch/CNBC*")
        return

    priority = [a for a in rss_articles if _is_priority(a["title"])]
    regular  = [a for a in rss_articles if not _is_priority(a["title"])]

    # Priority block — top of file if anything flagged
    if priority:
        lines.insert(3, "\n---")
        lines.insert(3, "")
        for a in reversed(priority[:8]):
            lines.insert(3, f"- [{a['title']}]({a['link']}) — *{a['source']} {a['pub']}*")
        lines.insert(3, f"\n## MARKET ALERTS — {len(priority)} priority stories")

    # Regular headlines section
    lines.append("\n---\n## Market Headlines  *(Reuters · MarketWatch · CNBC)*")
    for a in regular[:12]:
        lines.append(f"- [{a['title']}]({a['link']}) — *{a['source']} {a['pub']}*")
    lines.append(f"\n---\n*Generated {datetime.now(CT).strftime('%Y-%m-%d %H:%M CT')} "
                 f"| last {HOURS_BACK}h | sources: yfinance + Reuters/MarketWatch/CNBC*")


# ── File writer ────────────────────────────────────────────────────────────────

def save_markdown(md):
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved → {OUT_FILE}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print(f"\nNews Agent — {datetime.now(CT).strftime('%Y-%m-%d %H:%M CT')}")

    print("Fetching open positions...")
    positions = get_open_positions()
    print(f"  {len(positions)} positions: {list(positions.keys())}")

    all_syms  = list(positions.keys()) + MACRO_SYMS + SECTOR_SYMS

    print("Fetching price context...")
    price_ctx = get_price_context(all_syms)

    print(f"Fetching yfinance news (last {HOURS_BACK}h)...")
    all_news    = fetch_news(all_syms, hours_back=HOURS_BACK)
    total_arts  = sum(len(v) for v in all_news.values())
    print(f"  {total_arts} yfinance articles")

    print("Fetching RSS news (Reuters / MarketWatch / CNBC)...")
    rss_articles = fetch_rss_news(hours_back=HOURS_BACK)
    print(f"  {len(rss_articles)} RSS articles")

    pos_news    = {s: all_news.get(s, []) for s in positions}
    macro_news  = {s: all_news.get(s, []) for s in MACRO_SYMS}
    sector_news = {s: all_news.get(s, []) for s in SECTOR_SYMS}

    lines = build_markdown(positions, price_ctx, pos_news, macro_news, sector_news).split("\n")
    add_rss_section(lines, rss_articles)
    md = "\n".join(lines)

    save_markdown(md)
    print(md)
    print("Done.\n")


if __name__ == "__main__":
    main()


# ── GitHub Actions snippet ─────────────────────────────────────────────────────
# Add to .github/workflows/trading.yml:
#
#   news-agent:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4
#       - uses: actions/setup-python@v5
#         with: { python-version: "3.11" }
#       - run: pip install yfinance alpaca-py pytz
#       - run: python news_agent.py
#         env:
#           GMAIL_USER: ${{ secrets.GMAIL_USER }}
#           GMAIL_PASS: ${{ secrets.GMAIL_PASS }}
#
# Schedule (add under 'on:'):
#   schedule:
#     - cron: '30 13 * * 1-5'   # 8:30 AM CT
#     - cron: '0  17 * * 1-5'   # 12:00 PM CT midday check
