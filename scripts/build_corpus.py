"""
build_corpus.py — assemble the training text.

BEGINNER NOTE
-------------
A language model can only learn from the text you give it. This script
builds that text in two parts:

    1. data/notes/*.txt  — hand-written notes in this repo (ours).
    2. Wikipedia extracts — large public reference articles (CC BY-SA 4.0).

Part 2 needs the internet, but only ONCE: the merged result is saved to
data/cybersecurity_corpus.txt, and training reads that file offline.

Usage:
    .venv/bin/python scripts/build_corpus.py            # notes + wikipedia
    .venv/bin/python scripts/build_corpus.py --offline  # notes only
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# Where things live, relative to the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES_GLOB = os.path.join(ROOT, "data", "notes", "*.txt")
OUT_PATH = os.path.join(ROOT, "data", "cybersecurity_corpus.txt")

# Wikipedia's API endpoint. `action=query&prop=extracts` returns article text.
API = "https://en.wikipedia.org/w/api.php"

# Sites require a descriptive User-Agent; anonymous default agents get blocked.
HEADERS = {
    "User-Agent": "og-ai-corpus-builder/1.0 (educational ML project; contact: local user)"
}

# Articles chosen to cover the main areas of cybersecurity.
TITLES = [
    "Computer security", "Information security", "Security hacker", "Cyberattack",
    "Cryptography", "Advanced Encryption Standard", "RSA (cryptosystem)",
    "Elliptic-curve cryptography", "Diffie-Hellman key exchange", "Hash function",
    "Digital signature", "Public key certificate", "Public key infrastructure",
    "Transport Layer Security", "Hardware security module",
    "Multi-factor authentication", "Access control", "Zero trust security model",
    "Password", "Password cracking", "Rainbow table", "Salt (cryptography)",
    "Kerberos (protocol)", "Active Directory", "OAuth", "OpenID Connect",
    "JSON Web Token", "Single sign-on",
    "Firewall (computing)", "Intrusion detection system", "Virtual private network",
    "Demilitarized zone (computing)", "Network segmentation",
    "Security information and event management", "Honeypot (computing)",
    "Denial-of-service attack", "Man-in-the-middle attack", "ARP spoofing",
    "DNS spoofing", "Domain Name System", "Wi-Fi Protected Access",
    "Bluetooth", "Internet of things", "Operational technology",
    "Malware", "Ransomware", "Computer virus", "Trojan horse (computing)",
    "Rootkit", "Botnet", "Spyware", "Keylogger", "Fileless malware",
    "Phishing", "Social engineering (security)", "Business email compromise",
    "SQL injection", "Cross-site scripting", "Cross-site request forgery",
    "Server-side request forgery", "XML external entity", "Buffer overflow",
    "Privilege escalation", "Vulnerability (computing)", "Zero-day (computing)",
    "Common Vulnerabilities and Exposures", "Common Vulnerability Scoring System",
    "Penetration test", "Red team", "Bug bounty program",
    "Computer forensics", "Incident response", "Chain of custody",
    "Threat model", "STRIDE model", "Kill chain", "MITRE ATT&CK",
    "Threat intelligence", "Indicator of compromise",
    "Endpoint detection and response", "Antivirus software", "Application firewall",
    "Cloud computing security", "Kubernetes", "Container (computing)",
    "Software supply chain", "Software bill of materials",
    "General Data Protection Regulation", "ISO/IEC 27001",
    "NIST Cybersecurity Framework", "Payment Card Industry Data Security Standard",
    "Backup", "Disaster recovery", "Business continuity planning",
    "Security awareness", "Post-quantum cryptography", "Adversarial machine learning",
    "Prompt injection",
]


def fetch_extract(title, retries=5):
    """Download the plain-text extract for ONE Wikipedia article.

    IMPORTANT QUIRK: the API lowers full-article extracts to one article per
    request, so we must ask for a single title at a time. Asking for many
    titles in one call silently returns only the first one.

    Second quirk: Wikipedia rate-limits aggressive clients with HTTP 429.
    We honour its Retry-After header and pace requests instead of hammering.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",     # plain text, no wiki markup
        "redirects": "1",       # follow "Foo" -> "Bar" redirects
        "exlimit": "1",
        "titles": title,
    }
    url = API + "?" + urllib.parse.urlencode(params)

    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            # pages is a dict keyed by page id; we asked for one title.
            for page in data.get("query", {}).get("pages", {}).values():
                text = page.get("extract", "")
                if text:
                    return text
            return None  # page exists but has no extract (e.g. a redirect stub)

        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                # Server tells us how long to wait; default to 20s if silent.
                wait = int(exc.headers.get("Retry-After", 20)) if exc.headers else 20
                print(f"  … rate limited on {title!r}, waiting {wait}s",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            print(f"  ! HTTP {exc.code} for {title!r}", file=sys.stderr, flush=True)
            time.sleep(2 ** attempt)

        except Exception as exc:  # noqa: BLE001 - network errors are expected
            print(f"  ! {type(exc).__name__} for {title!r}", file=sys.stderr, flush=True)
            time.sleep(2 ** attempt)

    return None


def fetch_extracts(titles):
    """Download extracts for a list of titles, one request per title."""
    results = {}

    for index, title in enumerate(titles, start=1):
        text = fetch_extract(title)
        if text:
            results[title] = text
        print(f"  [{index}/{len(titles)}] fetched {len(results)} articles",
              end="\r", flush=True)

        # Politeness delay: about one request per second is comfortable.
        time.sleep(1.0)

    print()
    return results


def clean(text):
    """Normalise whitespace so the model sees tidy, consistent text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Keep printable ASCII only. Wikipedia maths articles contain rare Unicode
    # symbols (𝑓, ⊕, zero-width spaces, ...) that would waste vocabulary slots
    # and appear as random noise in generated text.
    text = text.encode("ascii", "ignore").decode("ascii")

    text = re.sub(r"[ \t]+", " ", text)        # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)     # max one blank line
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="only use local notes, skip Wikipedia")
    args = parser.parse_args()

    parts = []

    # ------------------------------------------------------------------
    # 1. Local notes, in filename order (01_, 02_, ...).
    # ------------------------------------------------------------------
    note_files = sorted(glob.glob(NOTES_GLOB))
    for path in note_files:
        with open(path, "r", encoding="utf-8") as f:
            body = clean(f.read())
        if body:
            parts.append(body)
    print(f"Local notes: {len(note_files)} files")

    # ------------------------------------------------------------------
    # 2. Wikipedia extracts (optional).
    # ------------------------------------------------------------------
    if not args.offline:
        print(f"Fetching {len(TITLES)} Wikipedia articles...")
        extracts = fetch_extracts(TITLES)
        for title in TITLES:
            # Wikipedia may redirect, so look up the returned title too.
            text = extracts.get(title)
            if text is None:
                for returned_title, candidate in extracts.items():
                    if returned_title.lower() == title.lower():
                        text = candidate
                        break
            if not text:
                print(f"  ! no extract for {title!r}", file=sys.stderr)
                continue

            # Cap very long articles so one page cannot dominate the corpus.
            text = clean(text)[:15000]
            parts.append(
                f"WIKIPEDIA ARTICLE: {title}\n\n"
                f"{text}\n\n"
                "Source: Wikipedia, licensed under CC BY-SA 4.0."
            )

    # ------------------------------------------------------------------
    # 3. Write one big file.
    # ------------------------------------------------------------------
    corpus = "\n\n".join(parts)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"Wrote {OUT_PATH}")
    print(f"Corpus size: {len(corpus):,} characters")


if __name__ == "__main__":
    main()
