# Chapter 14 — sensor-driven handoffs

Two projects preserve the C# handoff topology:

```text
EnvironmentAgent ── fire ──> FireDetectorAgent ──┐
                 ├─ rain ──> RainDetectorAgent ─┼─> MotorsAgent
                 └─ safe ───────────────────────┘
```

The environment agent reads simulated sensors, then the model chooses exactly
one route. Fire and rain specialists use their own tools and hand off to the
sole motor authority. Motors stops after a hazard, or executes the mission when
safe. No physical sensors, alarms, sprinklers, wipers or motors are controlled.

```powershell
python .\14.Handoff\HandoffOrchestration\main.py
python .\14.Handoff\HandoffOrchestrationAndHandoffReasons\main.py
python .\14.Handoff\HandoffOrchestration\main.py --scenario fire
python .\14.Handoff\HandoffOrchestrationAndHandoffReasons\main.py --scenario rain --provider azure
```

Configure the root `.env` as for prior chapters. Both examples default to OpenAI
and random simulated readings. `--scenario safe|fire|rain` fixes the readings for
repeatable demonstrations; routing is still the model's decision, not a Python
if/else replacement for handoff. Fire routing requires temperature **above 60°C**;
rain routing requires droplet level **High**. Random ranges and source tool-result
phrases are preserved. Source sensor/fire delays are 0.5 seconds, rain delays one
second, and motor delays one second.

## Python API adaptations

- Uses the real `HandoffBuilder` from the pinned orchestration package. All
  participant agents enable per-service-call history persistence as required
  by the Python handoff runtime. IDs are stable names, so synthetic tools are
  `handoff_to_FireDetectorAgent`, `handoff_to_RainDetectorAgent`, and
  `handoff_to_MotorsAgent`.
- In `HandoffOrchestration`, routing criteria live in the router instructions.
  In `HandoffOrchestrationAndHandoffReasons`, they live in each handoff tool's
  description, configured with separate `add_handoff` calls.
- Python deliberately strips tool-call/result protocol artifacts when sharing
  a conversation. This would otherwise hide the completed sensor/hazard results
  that the next C# agent expects. A function middleware records actual tool
  outcomes in per-run session state; an agent middleware adds plain-text evidence
  before the handoff. It preserves the final routing result position and does
  not copy synthetic handoff signals into evidence. The next agent sees completed
  outcomes without unmatched tool calls.
- Only MotorsAgent outputs are selected. The workflow terminates after its
  assistant response; Motors has no outgoing edge by design. The builder may
  issue a generic warning about no outgoing handoff, but the termination predicate
  makes this intentional terminal branch safe.
- If an agent responds without handing off, the workflow pauses with native
  request-info rather than silently invoking another agent. The script reports
  that pause and exits with code 1 if there is no terminal Motors output; it does
  not auto-approve or fabricate a route. Specialist instructions explicitly require
  invoking `handoff_to_MotorsAgent`: announcing a handoff in text is not a tool call.
- Mermaid is printed to the console instead of overwriting a root Markdown file.

Routing instructions and sensor scenarios are examples, not enforceable
physical-safety guarantees. Provider/model behavior still requires live testing
for a production application.

## Offline regression tests

```powershell
python -m unittest discover -s tests -p test_handoff.py -q
```

The tests run real framework handoff workflows against mocked provider responses
for safe/fire/rain branches, native streaming, route descriptions, terminal output,
plain-text tool evidence, separated tool authority, and SDK lifecycle. They do
not call live providers or operate hardware.
