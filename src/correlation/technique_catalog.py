import json
from pathlib import Path

# Text descriptions the Correlation Subagent embeds and searches. The ids must
# match the techniques in src/response/playbook.json (tests/test_mitigation.py
# checks this), so Mitigation and the Judge work from the same technique list.
TECHNIQUE_CATALOG = {
    "T1110": "Brute Force: adversary attempts to gain access via repeated login attempts.",
    "T1110.001": "Password Guessing: automated attempts to guess account passwords.",
    "T1566": "Phishing: adversary sends malicious messages to gain initial access.",
    "T1190": "Exploit Public-Facing Application: exploits a weakness in an internet-facing system.",
    "T1041": "Exfiltration Over C2 Channel: data stolen via the existing command-and-control channel.",
    "T1567": "Exfiltration Over Web Service: data stolen using a legitimate external web service.",
    # Added so Detection's DoS, DDoS and PortScan categories have a technique to map to.
    "T1498": "Network Denial of Service: flooding a network or service with traffic from one or many sources.",
    "T1499": "Endpoint Denial of Service: exhausting a specific service's resources with floods of requests.",
    "T1046": "Network Service Discovery: scanning hosts for open ports and running services.",
}

# Which techniques fit each Detection category (most likely first). Read from the
# Judge's playbook rather than copied, so the two can never drift apart.
PLAYBOOK_PATH = Path(__file__).resolve().parents[1] / "response" / "playbook.json"

with open(PLAYBOOK_PATH, encoding="utf-8") as _f:
    CATEGORY_TECHNIQUES = {
        category: list(ids) for category, ids in json.load(_f)["category_techniques"].items()
    }
