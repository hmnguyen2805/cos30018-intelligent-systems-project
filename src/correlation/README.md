# CTI Correlation Agent

**Owner:** Callum Fennessy

Takes a `DetectionResult` and maps it to known MITRE ATT&CK techniques from the seed
catalog (T1110, T1110.001, T1566, T1190, T1041, T1567).

## Your contract
run(detection: DetectionResult) -> CorrelationResult


## Important: this needs to be a real agent, not a flat lookup

A rule-based "check if X matches Y" lookup by itself doesn't count as an agent per the
grading criteria — it needs to reason and revise, not just return a fixed answer.

A good fit here: build this as **Agentic RAG** — exactly what Week 3's lecture/tutorial
covered (vector DB + RAG + agentic workflow). Store the technique descriptions in a
small vector DB, have the agent query it, and if the match confidence is low, let it
refine the query or search a different angle before settling on an answer (or
explicitly returning "no confident match" rather than forcing one). That reasoning step
is what makes it a loop, and it's a nice way to show you're applying that week's
material directly.

Call `self.log_step(...)` (inherited from `BaseAgent`) at each reasoning/query step so
your agent's actions are visible to the pipeline/UI and satisfy the "observable
iterations" requirement.

## Suggested next steps

1. Start simple: get a basic vector-DB lookup working against the 6 seed techniques.
2. Add the confidence check + retry/refine step — this is the part that makes it a
   genuine agent rather than a lookup table.
3. If it fits your interests, the Gradio UI (task submission + progress view) is also
   up for grabs — ask the group if you want to pick that up too.

## Dependencies

Add anything beyond the shared `requirements.txt` to a `requirements-correlation.txt`
in this folder, and note it here.