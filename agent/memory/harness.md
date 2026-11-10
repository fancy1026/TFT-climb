# Harness notes

- An item moves to its "final" holder by itself only with 10+ s left, the holder fielded and a free slot;
  otherwise move it yourself (a benched holder keeps its items). To keep an item: item_final = its holder.
- A next_round plan written in a fight IS the next phase's plan: put that phase's XP and roll in it (P7).
- A roll buys in the order you list; a list fills each unit to its auto_buy star (else 2): {"Cassiopeia": 3}
  for 3-star copies. A paid roll stops while the gold still buys the FIRST-listed champion: list first the unit
  you need most.
- XP runs before a roll in the same plan: level and roll in ONE plan. A roll sent with < 5 s left may not run.
- XP that reaches a new level needs the new slot's unit in "board" in the same plan, or it is rejected. A
  full board refuses a new unit: level up with it in "board", or place it on an occupied hex (a swap).
- Moves and items aimed at a "?" or "(missing)" unit often fail: "identify" it first. Give to fielded units.
- An Alpha Mark or a potion: {"use": item, "on": unit id}, no "final".
- Wisp and choice screens take 3-15 s of a phase. Augment rounds (2-1, 3-2, 4-2) give the phase to the pick
  (3 answers, ~40 s; the plan starts after it). So a level and roll-down wanted at 4-2 has to be planned before
  the pick - in the 4-1 fight's next_round, which runs as 4-2 opens - or it lands at 4-3 (g21: none was planned).
- In combat, gives to fielded units still run (~6 in 10 land; the rest go first next phase): keep them in
  execution_plan, not next_round.
- A board change sent with < 5 s left may not finish. A new unit on a full board first takes the replaced
  unit's hex, then moves: late, replace a unit standing where it belongs.
- Thief's Gloves fills 3 slots (gives to its holder are refused).
- Items can read short after a merge or a give, and HP can lag the screen (the state marks a stale HP): a later read corrects them. The Item Remover
  fails on a fielded unit in combat: move items in planning (g18 4-1).

## What a human player sees or does that the controller does not (capability inventory)

- Other players' HP, who is out and your place: in the state's Players line (read off the player list; when a unit
  panel covers it, the last read with its round).
- Scouting - other players' boards, levels, items, positions, and who you fight next: not available. Positions are
  chosen without seeing the board you meet.
- Everything else a planning phase needs is in the state: own board, bench, shop, HUD, item bench,
  each unit's items, the trait panel, augments, Wisps, carousel, loot.
