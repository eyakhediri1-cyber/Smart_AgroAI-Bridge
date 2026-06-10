"""
download_corpus.py — Download real olive corpus PDFs (FAO, EPPO, CIHEAM)
No hardcoded answers. Everything comes from the actual documents.

Usage:
    python download_corpus.py

After running:
    curl -X POST http://localhost:8000/admin/index-corpus
"""

import urllib.request
import urllib.error
import sys
from pathlib import Path

CORPUS_DIR = Path("corpus")
CORPUS_DIR.mkdir(exist_ok=True)

# ── Real public PDFs — no auth required ───────────────────────────────────────
SOURCES = [
    {
        "name": "FAO_olive_production_manual.pdf",
        "url":  "https://www.fao.org/3/y4890e/y4890e.pdf",
        "desc": "FAO Olive Production Manual (world reference)"
    },
    {
        "name": "FAO_olive_y4555e.pdf",
        "url":  "https://www.fao.org/3/y4555e/y4555e.pdf",
        "desc": "FAO — Olive growing (technical guide)"
    },
    {
        "name": "FAO_ecocrop_olive.pdf",
        "url":  "https://www.fao.org/3/i3144e/i3144e.pdf",
        "desc": "FAO ECOCROP — Climate & soil requirements"
    },
    {
        "name": "CIHEAM_olive_mediterranean.pdf",
        "url":  "https://om.ciheam.org/om/pdf/a56/04600027.pdf",
        "desc": "CIHEAM — Olive growing in the Mediterranean basin"
    },
    {
        "name": "CIHEAM_olive_diseases.pdf",
        "url":  "https://om.ciheam.org/om/pdf/a56/04600028.pdf",
        "desc": "CIHEAM — Olive diseases and pests"
    },
    {
        "name": "CIHEAM_olive_tunisia.pdf",
        "url":  "https://om.ciheam.org/om/pdf/a56/04600030.pdf",
        "desc": "CIHEAM — Olive sector in Tunisia"
    },
    {
        "name": "EPPO_cycloconium_oleaginum.pdf",
        "url":  "https://gd.eppo.int/download/doc/1135_DS_SPICCR.pdf",
        "desc": "EPPO — Spilocaea oleagina (peacock eye) data sheet"
    },
    {
        "name": "EPPO_colletotrichum_olive.pdf",
        "url":  "https://gd.eppo.int/download/doc/1222_DS_COLLAC.pdf",
        "desc": "EPPO — Colletotrichum acutatum (anthracnose) data sheet"
    },
    {
        "name": "EPPO_verticillium_dahliae.pdf",
        "url":  "https://gd.eppo.int/download/doc/1036_DS_VERTDA.pdf",
        "desc": "EPPO — Verticillium dahliae data sheet"
    },
    {
        "name": "IOC_olive_world_report.pdf",
        "url":  "https://www.internationaloliveoil.org/wp-content/uploads/2019/12/COI-T.20-Doc.-30-Rev.14-2019-World-Olive-Production.pdf",
        "desc": "International Olive Council — World olive production report"
    },
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; olive-corpus-downloader/1.0)"
}

def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if not data[:4] == b"%PDF":
            print(f"    ⚠️  Not a valid PDF (got {data[:20]})")
            return False
        dest.write_bytes(data)
        size_kb = len(data) // 1024
        print(f"    ✅ {dest.name} ({size_kb} KB)")
        return True
    except urllib.error.HTTPError as e:
        print(f"    ❌ HTTP {e.code}: {url}")
        return False
    except Exception as e:
        print(f"    ❌ {e}")
        return False

def main():
    print("📥  Downloading olive corpus (real PDFs only)")
    print("=" * 55)

    ok, failed = [], []

    for src in SOURCES:
        dest = CORPUS_DIR / src["name"]
        print(f"\n  {src['desc']}")
        if dest.exists():
            print(f"    ✅ Already exists — skipping")
            ok.append(src["name"])
            continue
        success = download(src["url"], dest)
        (ok if success else failed).append(src["name"])

    # ── Scan for user-supplied PDFs already in /corpus/ ──────────────────────
    all_pdfs = list(CORPUS_DIR.glob("*.pdf"))
    auto_names = {s["name"] for s in SOURCES}
    user_pdfs = [p for p in all_pdfs if p.name not in auto_names]
    if user_pdfs:
        print(f"\n📂  Found {len(user_pdfs)} user-supplied PDF(s) in ./corpus/:")
        for p in user_pdfs:
            print(f"    📄 {p.name} ({p.stat().st_size // 1024} KB)")
    else:
        print(f"\n📂  Tip: drop your own PDFs into ./corpus/ — they will be indexed too.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"✅  Downloaded : {len([x for x in ok if x not in [p.name for p in user_pdfs]])}/{len(SOURCES)}")
    print(f"❌  Failed     : {len(failed)}")
    if failed:
        print("\n  Failed (add PDFs manually to ./corpus/):")
        for f in failed:
            print(f"    • {f}")

    total = len(list(CORPUS_DIR.glob("*.pdf")))
    print(f"\n📚  Total PDFs in ./corpus/ : {total}")
    if total == 0:
        print("\n  ⚠️  No PDFs found! The RAG will refuse all questions.")
        print("      Add at least one PDF to ./corpus/ and re-run.")
        sys.exit(1)

    print("\n🔧  Next step — index the corpus:")
    print("    curl -X POST http://localhost:8000/admin/index-corpus")

if __name__ == "__main__":
    main()
