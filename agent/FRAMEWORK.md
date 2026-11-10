# The climb agent's framework (the owner's master plan, 2026-11-01)

Goal: the best placement, aiming for Master, with one fixed comp per game (agent/config.json "comp"). The master
plan page (the owner's artifact) draws the same as this file. The player is GPT-6.1 Terra high (Sol medium on odd
games), every role GPT-6.1 Sol high (agent/config.json).

Two phases: during the game, and outside it, where independent panels hand their output to the next through files.

## During the game

Every model call the player makes is: the fixed instructions (rules as numbers, reply format) + the comp strategy
(agent/memory/comps/<comp>.md: the owner's rules for the comp first, then what the comp is - final board, carry and
items, unit jobs, positions - and its code rules, which the executor enforces whenever their condition holds) + the
STAGE n STRATEGY of that comp
(agent/stages.py; stage 1 is 1-1 to 1-4, stages 2, 3 and 4 each have their own, 5 to 7 share one; marked as learned
from strong players' games and our earlier games; each text sent once per conversation, in its stage's first
message) + the player's game plan and notes + the round state (current facts, as text: no screenshot reaches the
model since 2026-11-02; a state section the conversation was already sent unchanged is named, not repeated). The
player plays the game in two conversations (the owner, 2026-11-02): the first from the preparation through the 4-2
fight; in that fight it writes a handover, and the second, from 4-3 to the end, opens with the digest (comp brief,
trait choices, game plan, notes, last state), the handover and stage 4's strategy. The stage strategies are read once at game start and written to
runs/<run>/stage_strategies.json, so the review knows which versions played. At an augment screen each card read is
looked up first in data/reference/augment-harness.json (every augment of the set rated by the mechanisms it needs
against what the harness can do; the owner, 2026-11-03): a card the harness plays poorly is marked lower priority
with its reason, and a comp's banned augments (agent/expert/rules) are marked never picked.

## Outside the game

**Learn** (agent/comp_coach.py, agent/study_queue.py, agent/roles/comp_coach.md): starts on a new comp, a new patch,
a study request, or a comp without stage strategies. MetaTFT's page, strong players' games found and read (HUD every
2 s, a keyframe a round, their words from captions), the material round by round; about 30% of the comp's strong
players' games are kept out of it for the Judge (agent/comps/<comp>/heldout.json). The comp coach writes the comp
notes, the comp's code rules (each new one goes to the Editor, after the owner's yes) and version 1 of each stage
strategy, every point with a source code checks.

**Review**, after each game (agent/climb.py review_game, agent/roles/reviewer.md):
1. the stage report, one session per stage (Job 1): every decision of every round graded against the best one the
   state allowed - very good to very bad, or uncertain when the record cannot tell - never by the fight result, HP or
   the place; the packet carries the whole game in numbers and the luck measured (agent/stage_review.py); the
   Reviewer asks the player when it has a question (agent/ask_player.py); it writes what happened and the
   differences from strong players that mattered; it does not grade whether a strategy was followed. Code scores
   each part out of 10 from the grades;
2. the diff report, one session per strategy stage (Job 4): this stage against the same stage of the comp's earlier
   games, with the version, player model and code each played, and the watches due: keep, revise, revert, or not
   enough evidence;
3. the summary (Job 2): what decided the place and in which stage it began, harness faults and missing information
   with evidence;
4. the report page for the owner (agent/model/reviews/<run>-report.html): screenshots, scores, diff verdicts, the
   running average of places;
5. the change review (Job 3, agent/changes.py, recorded in agent/changes.jsonl): each harness change against its
   watch - keep, remove, monitor.

**Strategist** (agent/roles/strategist.md): one session per stage the diff reports mark revise or revert. It reads
the review and decides: rewrite (with a watch: where, expect, risk, how, games), revert, or keep. The Judge
(agent/judge.py stage_gate) asks the stage's recorded strong players' decisions with the current and the new text,
three times each, graded by code against the strong player's level and gold; the new text ships only if it scores no
lower on the games the Strategist may read and on the held-out ones. Code writes the version (agent/stages.py).

**Harness fixes** (agent/roles/harness_fixer.md): the Editor takes the review's fault list as leads, confirms each
fault in the game's log and screenshots, fixes it at its cause with a regression test from that record, and writes
its watch in the commit. agent/check.py must pass. Anything new - a rule, a prompt line, a new behaviour - waits for
the owner's yes.

**Domain expert feedback** (the owner's; agent/expert/, agent/expert.py), first over every other part (the owner,
2026-11-02: "my feedback should override the reviewer as domain expert found truth"; "this should be prioritized, the
other agents could disagree, or point out the problems but need to state the reasons"):
- the owner's rules for a comp (agent/expert/rules/<comp>.json) are the first section of its comp notes ("FIRST
  RULES"), above the code rules and the comp coach's points: where anything says otherwise, they hold. The player gets
  the rules only; the owner's words and reasons stay with the roles (the owner, 2026-10-30: the feedback goes to the
  framework, "not the player");
- the owner's grades of a game's decisions replace the Reviewer's in that game's audits wherever code reads them (the
  stage scores, the diff reports, the report page); the Reviewer's grade stays beside each as "reviewer_grade";
- any role may disagree with an owner rule or grade, or point out a problem in it, only with its reasons and evidence:
  `"owner_disagreements": [{"point": "<rule id or x-point>", "why": "..."}]` in its final JSON line. Code writes them
  to agent/expert/disagreements.jsonl; the Coordinator puts each to the owner. Until the owner changes it, the point
  holds: no role writes, grades or proposes against it.

**You and oversight**: the owner starts games, approves anything new, and gives feedback; the Coordinator (the
owner's chat) turns feedback into framework changes (agent/expert/), keeps the master plan, and scores games against
the owner's hidden labels when asked. No Director and no periodic audit.

## Round cases: a fix is shown on the game it was found in

The owner, 2026-11-10 (after game 81: Reviewer findings left open across four meta reviews, and fixes that passed the
check but were never run on the rounds they were for): every fault or wrong decision a Reviewer finds, and every owner
point, becomes a round case when a question of the game shows it - the round's recorded question and a check of what
should come out (agent/round_cases.py; agent/round_cases/cases.jsonl). A case must fail on the round as played (the
game's answer through the game's code), else it does not catch the problem.
- harness cases run the game's own answer through the current code; decision cases ask the question again of the game
  model three times (a majority decides).
- A fix lands only when its case flips: the climb runs every case before and after an Editor session; a case that
  passed before and fails after takes the change back, and a request with cases closes only when they pass.
- Failing cases stay on one list until they pass (agent/round_cases/results.json): a failing harness case is an Editor
  request, costliest first, and the Editor works the list again after a batch while its last session made cases pass;
  a failing decision case makes its stage due for the Strategist.
- The Judge decides a stage text on the stage's cases first: rejected if it breaks a passing case, accepted if it makes
  a failing one pass and breaks none; the strong players' decisions decide only when the cases do not.
- After each batch the decision cases are asked again on the notes as they are, and the owner gets the pass count.

## Roles and what they may write

| role | writes | may not |
|---|---|---|
| Player | the game's actions | anything else |
| Reviewer | agent/model/reviews/, agent/reviews/ | anything else |
| Strategist | agent/comps/<comp>/candidates/, agent/model/strategist/ | the stage strategies (code writes them after the Judge), notes, code |
| Comp coach | agent/comps/<comp>/proposed.json | the stage strategies once written, the memory |
| Editor | tft_player/, tools/, tests/, data/ | strategy, the memory, the Judge |
| Judge | code: verdicts in the stage strategy's version | - |

A guard (agent/climb.py guard) takes back any change a role made outside its files; stages.json and heldout.json
are written only by code. The Codex usage mark and the roles' disagreements with the owner are written by the loop's
code around a session, not by the role, and stay. runs/ (gitignored) is the only copy of the recorded games: a role
session after which a run folder is gone writes agent/STOP, so no game runs until the Coordinator has looked
(2026-11-03: an Editor session deleted all of runs/ through a junction in a scratch worktree).

## Rules

1. The owner's rules ship as stated; changing their wording or scope needs the owner. The owner's feedback comes
   first: a role disagrees only with reasons, and the point holds until the owner changes it.
2. No content limits on a strategy (the owner, 2026-11-01: "if it finds out forcing a level 8 is better, then it uses
   it"); the Judge and the diff reports keep what helps and drop what does not.
3. Nobody grades their own output: the Judge is code; the Strategist cannot write a stage strategy; the Reviewer
   grades decisions, not compliance; the held-out games are read by no role.
4. Places are not the measure of a decision; HP never is. Luck is measured by code and reasoned over by the model.
5. Harness faults and information gaps are fixed after each game; every change carries a watch.
6. Live games start only when the owner starts them. Never push.
