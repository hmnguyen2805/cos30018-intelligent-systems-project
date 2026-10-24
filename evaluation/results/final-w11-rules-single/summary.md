# Evaluation results (rules, single_shot; model ollama_chat/qwen2.5:3b)

69 tasks, modes: rules, single_shot. 0 runs skipped.

## Metrics per Judge mode

| metric | rules | single_shot |
|---|---|---|
| runs | 69 | 207 |
| success | 55/69 (80%) | 180/207 (87%) |
| unsafe | 6/69 (9%) | 3/207 (1%) |
| over-cautious | 8/69 (12%) | 24/207 (12%) |
| avg iterations | 0.0 | 1.0 |
| avg LLM calls | 0.0 | 1.0 |
| avg tokens | 0 | 722 |
| avg LLM latency (s) | 0.00 | 12.94 |
| avg total latency (s) | 19.81 | 33.95 |
| Mitigation agent runs | 53/69 (77%) | 159/207 (77%) |
| Mitigation avg LLM calls | 2.7 | 2.7 |
| Mitigation avg tokens | 1768 | 1768 |
| Mitigation avg LLM latency (s) | 23.57 | 25.75 |
| Mitigation fallback rate | 5/53 (9%) | 15/159 (9%) |
| avg tokens (Judge + Mitigation) | 1358 | 2080 |
| tool-use success | - | - |
| fallback rate | 0/69 (0%) | 0/207 (0%) |
| recovery rate | - | - |
| guardrail overrides | 0 | 6 |
| rechecks | 0 | 0 |
| consistency | - | 69/69 (100%) |

## Success by difficulty and split

| group | rules | single_shot |
|---|---|---|
| simple | 16/16 (100%) | 48/48 (100%) |
| moderate | 28/37 (76%) | 84/111 (76%) |
| difficult | 4/8 (50%) | 24/24 (100%) |
| edge | 7/8 (88%) | 24/24 (100%) |
| split: dev | 27/33 (82%) | 96/99 (97%) |
| split: unseen | 28/36 (78%) | 84/108 (78%) |

## Per task

| task | difficulty | expected | rules | single_shot |
|---|---|---|---|---|
| S01 | simple | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| S02 | simple | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| S03 | simple | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| S04 | simple | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| S05 | simple | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| S06 | simple | no_action | no_action 1 (1/1 ok) | no_action 3 (3/3 ok) |
| S07 | simple | no_action | no_action 1 (1/1 ok) | no_action 3 (3/3 ok) |
| S08 | simple | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| M01 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| M02 | moderate | act | escalate 1 (0/1 ok) | escalate 3 (0/3 ok) |
| M03 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| M04 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| M05 | moderate | act/escalate | act 1 (1/1 ok) | act 3 (3/3 ok) |
| M06 | moderate | no_action/escalate | no_action 1 (1/1 ok) | no_action 3 (3/3 ok) |
| M07 | moderate | escalate | escalate 1 (1/1 ok) | escalate 3 (3/3 ok) |
| M08 | moderate | escalate | escalate 1 (1/1 ok) | escalate 3 (3/3 ok) |
| M09 | moderate | escalate | escalate 1 (1/1 ok) | escalate 3 (3/3 ok) |
| D01 | difficult | escalate | act 1 (0/1 ok) | escalate 3 (3/3 ok) |
| D02 | difficult | escalate | act 1 (0/1 ok) | escalate 3 (3/3 ok) |
| D03 | difficult | escalate | act 1 (0/1 ok) | escalate 3 (3/3 ok) |
| D04 | difficult | escalate | act 1 (0/1 ok) | escalate 3 (3/3 ok) |
| D05 | difficult | act/escalate | act 1 (1/1 ok) | act 3 (3/3 ok) |
| D06 | difficult | escalate | escalate 1 (1/1 ok) | escalate 3 (3/3 ok) |
| D07 | difficult | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| D08 | difficult | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| E01 | edge | escalate | escalate 1 (1/1 ok) | escalate 3 (3/3 ok) |
| E02 | edge | no_action | no_action 1 (1/1 ok) | no_action 3 (3/3 ok) |
| E03 | edge | failed | failed 1 (1/1 ok) | failed 3 (3/3 ok) |
| E04 | edge | escalate | act 1 (0/1 ok) | escalate 3 (3/3 ok) |
| E05 | edge | act/escalate | act 1 (1/1 ok) | act 3 (3/3 ok) |
| E06 | edge | escalate | escalate 1 (1/1 ok) | escalate 3 (3/3 ok) |
| E07 | edge | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| E08 | edge | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-Benign-1 | simple | no_action | no_action 1 (1/1 ok) | no_action 3 (3/3 ok) |
| U-Benign-2 | simple | no_action | no_action 1 (1/1 ok) | no_action 3 (3/3 ok) |
| U-Benign-3 | simple | no_action | no_action 1 (1/1 ok) | no_action 3 (3/3 ok) |
| U-Benign-4 | simple | no_action | no_action 1 (1/1 ok) | no_action 3 (3/3 ok) |
| U-Benign-5 | simple | no_action | no_action 1 (1/1 ok) | no_action 3 (3/3 ok) |
| U-Benign-6 | simple | no_action | no_action 1 (1/1 ok) | no_action 3 (3/3 ok) |
| U-Benign-7 | simple | no_action | no_action 1 (1/1 ok) | no_action 3 (3/3 ok) |
| U-Benign-8 | simple | no_action | no_action 1 (1/1 ok) | no_action 3 (3/3 ok) |
| U-Bot-1 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-Bot-2 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-DDoS-1 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-DDoS-2 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-DoSGoldenEye-1 | moderate | act | escalate 1 (0/1 ok) | escalate 3 (0/3 ok) |
| U-DoSGoldenEye-2 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-DoSHulk-1 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-DoSHulk-2 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-DoSSlowhttptest-1 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-DoSSlowhttptest-2 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-DoSslowloris-1 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-DoSslowloris-2 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-FTP-Patator-1 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-FTP-Patator-2 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-Heartbleed-1 | moderate | act | escalate 1 (0/1 ok) | escalate 3 (0/3 ok) |
| U-Heartbleed-2 | moderate | act | escalate 1 (0/1 ok) | escalate 3 (0/3 ok) |
| U-Infiltration-1 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-Infiltration-2 | moderate | act | escalate 1 (0/1 ok) | escalate 3 (0/3 ok) |
| U-PortScan-1 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-PortScan-2 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-SSH-Patator-1 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-SSH-Patator-2 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-WebAttack-BruteForce-1 | moderate | act | escalate 1 (0/1 ok) | escalate 3 (0/3 ok) |
| U-WebAttack-BruteForce-2 | moderate | act | escalate 1 (0/1 ok) | escalate 3 (0/3 ok) |
| U-WebAttack-SqlInjection-1 | moderate | act | no_action 1 (0/1 ok) | no_action 3 (0/3 ok) |
| U-WebAttack-SqlInjection-2 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-WebAttack-XSS-1 | moderate | act | act 1 (1/1 ok) | act 3 (3/3 ok) |
| U-WebAttack-XSS-2 | moderate | act | escalate 1 (0/1 ok) | escalate 3 (0/3 ok) |

## Tasks where the modes differ

| task | rules | single_shot |
|---|---|---|
| D01 | 0% | 100% |
| D02 | 0% | 100% |
| D03 | 0% | 100% |
| D04 | 0% | 100% |
| E04 | 0% | 100% |
