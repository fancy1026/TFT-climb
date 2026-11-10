# Patch 18.6 changes (2026-11-03; on OC1 since the 2026-11-04 maintenance)

Source: [Riot's patch 18.6 notes](https://teamfighttactics.leagueoflegends.com/en-us/news/game-updates/teamfight-tactics-patch-18-6/),
published 2026-11-03. Latest update first; no later update is present in the supplied notes.
Game-play and in-game interface changes only; no cosmetics. Today's snapshot is
`cdragon-tft-2026-11-05.json`, but some of its fields still contain earlier values.
Where the game-data files disagree with this list, this list is current. Format: old => new.

## Traits
- Blossom AD/AP: 12/30/50/55% => 12/30/55/65%
- Elderwood (9) bonus health: 25% => 10%
- Sprykin BFF ranged AD/AP: 15/28/33% => 15/30/38%

## Champions
- Kobuko (1) spell healing: 285/335/480 => 300/350/550 (AP)
- Rek'Sai (1) spell healing: 10/14/20 => 10/16/24 (AP)
- Murkwolf (2) empowered attack damage: 65/100/160 => 65/100/180 AD
- Malphite (4) shield: 735/885/2000 => 750/925/2000 (AP); spell damage: 40/60/2000 => 100/150/2000 (AP)
- Nidalee (4) AD form heal: 300/450/2000 => 325/550/2000 (AP); third cast bonus damage: 125% => 140%
- Nidalee (4) AP form empowered attack: 185/275/2000 => 195/295/2000 (AP); third attack: 330/500/3000 =>
  350/525/3000 (AP)
- Sentinel (4) shield: 10% => 12% of max health

## Augments
- Clockwork Accelerator attack speed per proc: 10% => 12%
- Embiggen bonus Pixie AD/AP: 50% => 35%
- Plot Armor base resists: 8 => 10
- Shopping Spree gold at once: 6 => 7
- Unrivaled: Rengar heal share 25% => 50%; Kha'Zix mana share 50% => 70%

## Wisps (shop)
- Curio Cart emblem cost: 15 => 17
- Middle Path cost: 4 => 5; upgrade cost: 3 => 4

## Rules and fixes that change play
- Carousel release wait: longer => slightly shorter (Riot gives no number).
- Calculated Loss gold timing: incorrect => beginning of the next planning phase.
- Nesting Anvils with a full bench: could fail to grant the component anvil and gold => grants both.
- Consumables after use: some could switch item-bench places => no unintended switching.
- Cooking Pot exclusions: Coronation and Spreading Roots exclusions could fail => properly excluded.
- Flexible and U.R.F.: could appear together => mutually exclusive.
- Gambler's Blade: could drop gold in PvE => no gold drops in PvE.
- Hard Commit: could grant ineligible emblems => ineligible emblems excluded.
- Moonrise and Solar Gift Wisp upgrades: could appear without active Blossom => appear only with active Blossom.
- Clear Mind and Cluttered Mind: could grant erroneous XP in some circumstances => erroneous grants fixed.
- Blighting Jewel: could grant excess AP => grants intended AP (Riot gives no corrected number).
- Slightly Magic Roll: HP and size bonuses could fail => bonuses apply correctly.
- Worth the Wait and Worth the Wait II tooltips: selected champion's name could be missing => name displayed.
- Projectiles while both units move: inconsistent movement => movement fixed (Riot gives no further mechanics).
- Choncc or Treat Solo Leveling: enabled => disabled due to a bug (event-specific).
- Various crashes: present => fixed (Riot does not enumerate them).

## In-game interface
- Chat: fewer lines and previous sizing/background => more lines, better-sized text, lighter window.
- Streamer Mode: unavailable => returned.
- /Deafen: unavailable => restored as an in-chat command.
- Buy XP and Reroll button hover states on PC: harder to see => easier to see; no price change announced.
- Mobile item borders: could be missing => missing borders fixed.
- Tethers: could fail to connect visually => connection display fixed (visual only).

## Unchanged structure
- No champion, trait, item or emblem addition/removal is announced in 18.6.
- No champion price, XP purchase price, reroll price, shop odds, round sequence or augment-round change is announced.
  The carousel release wait above changes timing within a carousel, not its round.
