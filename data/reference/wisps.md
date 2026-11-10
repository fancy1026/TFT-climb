# Wisps

Non-champion shop cards. Buying one applies the effect below. N = a number the game data
does not give; the card in the shop shows the real value and the cost.
Upgraded = the Radiant version (e.g. from the Blossom trait).
Kind = what the effect changes: combat (our fights) or economy (gold, XP, rerolls, the shop, champions, items
kept) - the player's Wisp flag buys or skips by kind (tft_player/wisp_flag.py).

| Wisp | Effect | Upgraded | Kind |
|---|---|---|---|
| Abandon Ship | Sell your board and bench, gain 6 gold and 8 rerolls. | Sell your board and bench, gain N gold and N rerolls. | economy |
| Aftershock | After 8 seconds, stun all enemies for 1.5 seconds. | After 8 and N seconds, stun all enemies for N seconds. | combat |
| All Fives | Reroll your shop with all 5-cost champions. | Reroll your shop with all 5-cost champions. Gain N gold. | economy |
| All Fours | Reroll your shop with all 4-cost champions. | Reroll your shop with all 4-cost champions. Gain N gold. | economy |
| All Ones | Reroll your shop with all 1-cost champions. | Reroll your shop with all 1-cost champions. Gain N Gold. | economy |
| All Threes | Reroll your shop with all 3-cost champions. | Reroll your shop with all 3-cost champions. Gain N gold. | economy |
| All Twos | Reroll your shop with all 2-cost champions. | Reroll your shop with all 2-cost champions. Gain N gold. | economy |
| Animate Shop | For 18 seconds, your shop rerolls for free every 2 seconds. | For N seconds, your shop rerolls for free every 2 seconds. | economy |
| Apprentice | Gain a random 2-star 1-cost champion. | same text, stronger numbers | economy |
| Artifactinate | Lose 3 components, including ones built into items. Gain an Artifact anvil. | Lose N components, including ones built into items. Gain an Artifact anvil and N Reforger. | economy |
| Backrow Star | A random back row champion gains 100% Attack Speed for 7 seconds. | A random back row champion gains N% Attack Speed for N seconds. | combat |
| Bark Armor | Your champions gain 150 shield for 10 seconds. | Your champions gain N shield for N seconds. | combat |
| Barrier | Allies gain 1250 Shield, decaying over 6 seconds. | Allies gain 1250 Shield, decaying over N seconds. | combat |
| Barter | The next champion sold this round sells for N% gold. | same text, stronger numbers | economy |
| Bear's Visit | Gain the Bear Primal buff for this combat. |  | combat |
| Beggar's Wisp | The wilds couldn't find a wisp for you this roll. Gain 2 gold. | The wilds couldn't find a wisp for you this roll. Gain N gold. | economy |
| Big Boom | The first ally to die explodes, dealing N% Max Health magic damage. | The first N allies to die explode, dealing N% Max Health magic damage. | combat |
| Big Guns | Gain 2 random 1-star 4-cost champions. | same text, stronger numbers | economy |
| Blast Potion | Gain a temporary equippable Blast Potion. | Gain a temporary equippable Radiant Blast Potion. | combat |
| Blaze | Fire spreads across the board, dealing true damage to enemies. | Fire spreads across the board, dealing more true damage to enemies. | combat |
| Blood Money | Gain N gold for every champion you kill. | Gain N gold for every champion you kill. Gain N gold. | economy |
| Blood and Iron | After N champion deaths, gain a component anvil. | same text, stronger numbers | economy |
| Booster Shot | Up to N 1-star champions temporarily star up during combat. | same text, stronger numbers | combat |
| Border Village | Reroll your shop with all 2-star 2-cost champions. | same text, stronger numbers | economy |
| Borrowed Gear | At the start of combat, a champion gains a temporary recommended item. | At the start of combat, a champion gains a temporary recommended item and 250 Health. | combat |
| Bronze Spoon | Gain 4 XP. | Gain N XP. | economy |
| Bulwark | Your Tanks gain a 20% Max Health shield upon falling below 50% Health. | Your Tanks gain a N% Max Health shield upon falling below 50% Health. | combat |
| Bunch-o'-Belts | Your allies gain N temporary, non-removable Giant's Belts. | same text, stronger numbers | combat |
| Circle of Elders | Gain one of each 5-cost champion. | same text, stronger numbers | economy |
| Clone Companion | Clone your highest damage champion. The clone deals 20% damage. | Clone your highest damage champion. The clone deals N% damage. | combat |
| Coin Flip | Flip a coin, gain N gold if it's heads. | Flip N coins, gain N gold if it's heads. | economy |
| Combust | Your champions explode on death, dealing N% Max Health magic damage. | same text, stronger numbers | combat |
| Counterspell | Increase all enemies' Max Mana by 15 until their next cast. | Increase all enemies' Max Mana by N until their next cast. | combat |
| Crystal Ball | Reveal your next opponent. Your team gains 12% Attack Speed. | Reveal your next opponent. Your team gains N% Attack Speed. | combat |
| Curio Cart | Open an item store. Emblems cost 17 gold (patch 18.6); other item prices are shown in the store. | Open an item store. Emblems cost 17 gold (patch 18.6); other item prices are shown in the store. Gain 1 Gold. | economy |
| Cutpurse | Every 5 seconds, your champions have a N% chance to drop N Gold. | Every N seconds, your champions have a N% chance to drop N Gold. | economy |
| Die Roll | Roll a six-sided die. Gain gold equal to the result. | Roll two six-sided dice. Gain gold equal to the result. | economy |
| Diversified | Your team gains N Health for each active non-unique trait. | same text, stronger numbers | combat |
| Doodad Bag | Gain N consumables that give permanent Armor and Magic Resist or Health. | same text, stronger numbers | economy |
| Doodad Jar | Gain N consumable that gives permanent Armor and Magic Resist or Health. | Gain N consumables that give permanent Armor and Magic Resist or Health. | economy |
| Doodad Sack | Gain N consumables that give permanent Armor and Magic Resist or Health. | same text, stronger numbers | economy |
| Downpour | Your team gains 2 Mana Regen. | Your team gains N Mana Regen. | combat |
| Drizzle | After 8 seconds of combat, your team gains N Mana. | same text, stronger numbers | combat |
| Early Fix | Gain 1 Item Reforger. | Gain N Item Reforgers. | economy |
| Essence Theft | Gain 50% bonus Coven Essence this combat. | Gain N% bonus Coven Essence this combat. | economy |
| Experienced | Gain 2 XP. | Gain N XP. | economy |
| Fellowship | Champions gain 4% Max Health Shield and 4% Damage Amp per adjacent ally. | Champions gain N% Max Health Shield and N% Damage Amp per adjacent ally. | combat |
| Fertilize | Gain N XP for each enemy that dies. | Gain N XP for each enemy that dies. Gain N XP. | economy |
| Flash Fire | Reroll your shop. Inferno lights N shop slots on fire, upgrading them. | same text, stronger numbers | economy |
| Flip Frenzy | Flip a coin, gain N gold if it's heads. Continue until you flip tails. | same text, stronger numbers | economy |
| Flow | Your champions are unstoppable and gain 15% Attack Speed for 10 seconds. | Your champions are unstoppable and gain 15% Attack Speed for N seconds. | combat |
| Forest Guide | Gain a random 4-cost champion. | same text, stronger numbers | economy |
| Forest Mage | Fill your shop with Wisps. You can only buy one. | Fill your shop with upgraded Wisps. You can only buy one. | economy |
| Forest Twins | Gain 2 copies of a 2-cost champion you don't have. | Gain 3 copies of a 2-cost champion you don't have. | economy |
| Found Friend | Gain a temporary Training Dummy with a Giant's Belt. | Gain a Training Dummy with a Sunfire Cape. | combat |
| Freeroller | Gain 2 rerolls. | Gain N rerolls. | economy |
| Giant Growth | Your highest Health unit grows, gaining N Health. | Your N highest Health units grow, gaining N Health. | combat |
| Giant's Aura | Your highest Health ally deals 2% max Health damage per second in an area. | Your highest Health ally deals N% max Health damage per second in an area. | combat |
| Golden Goose | Gain N gold for every N damage your highest damage champion deals. | same text, stronger numbers | economy |
| Grandmaster | Gain a random 2-star 5-cost champion. | same text, stronger numbers | economy |
| Greater Chaos | Cast N random Wisps. | Cast N random Wisps worth N or more gold. | economy |
| Grove Armor | Your highest Health champion gains 15 Armor and 15 Magic Resist. | Your highest Health champion gains N Armor and N Magic Resist. | combat |
| Grow Up | Gain 10 XP. | Gain N XP. | economy |
| Hand Of Baron | Your team gains Baron Buff, increasing their stats. | Your team gains Baron Buff, greatly increasing their stats. | combat |
| Health Potion | Gain a temporary equippable Health Potion. | Gain a temporary equippable Radiant Health Potion. | combat |
| Heated Rivalry | Gain a Rengar or a Kha'Zix. They gain N takedowns. | same text, stronger numbers | economy |
| Hero Of Prophecy | Gain a 5-cost champion. Gain them again each round FOREVER. | same text, stronger numbers | economy |
| Hero's Entrance | After N seconds, the leftmost benched champion joins combat and gains N recommended items. | same text, stronger numbers | combat |
| Heroic Sacrifice | Your 3 strongest Champions gain HEROIC POWER, but other allies are sacrificed. | same text, stronger numbers | combat |
| Hireling | A temporary N-star 5-cost with N recommended items joins your team. | same text, stronger numbers | combat |
| Homing Fireflies | Your Casters spawn a firefly per N Mana spent. Each deals N magic damage. | same text, stronger numbers | combat |
| Hugify | Your highest Health unit grows, gaining 350 Health. | Your highest Health unit grows, gaining N Health. | combat |
| Hummingbird | Your highest Attack Speed unit shrinks, gaining 20% Attack Speed. | Your highest Attack Speed unit shrinks, gaining N% Attack Speed. | combat |
| Idle Craftsman | A component on your bench turns into a completed item it builds. | same text, stronger numbers | economy |
| Improved Reach | Your Fighters gain 1 range. | Your Fighters gain N range. | combat |
| Infliction | Slow, Burn, Shred, and Sunder enemies for 8 seconds. | Slow, Burn, Shred, and Sunder enemies for N seconds. | combat |
| Iron Core | Your front center champion gains N% Max Health per front-row unit. | same text, stronger numbers | combat |
| Ironwood | Enemies attacking your highest health champion deal 14% less damage. | Enemies attacking your highest health champion deal N% less damage. | combat |
| Journeyman | Gain a random 2-star 2-cost champion. | same text, stronger numbers | economy |
| Jungling | Overrun the shop with Riftbeasts. | same text, stronger numbers | economy |
| Killer's Regret | Your first 3 champions to die stun their killer for 1.5 seconds. | Your first N champions to die stun their killer for N second. | combat |
| Killing Frenzy | Fighters execute targets under 12% Health. They gain N% Attack Speed on kill. | Fighters execute targets under 15% Health. They gain N% Attack Speed on kill. | combat |
| Knick-Knack Bag | Gain N consumables that give permanent Attack Damage or Attack Speed. | same text, stronger numbers | economy |
| Knick-Knack Jar | Gain N consumable that gives permanent Attack Damage or Attack Speed. | Gain N consumables that give permanent Attack Damage or Attack Speed. | economy |
| Knick-Knack Sack | Gain N consumables that give permanent Attack Damage or Attack Speed. | same text, stronger numbers | economy |
| Late Bloomer | A temporary N-star 1-cost joins your team. | A temporary N-star 1-cost with Nrecommend item joins your team. | combat |
| Lesser Chaos | Cast a random Wisp. | Cast a random Wisp worth N or more gold. | economy |
| Life Debt | Gain N gold for every N missing Player Health. | Gain N gold for every N missing Player Health. Then heal N Health. | economy |
| Lightning Storm | Every 1 second deal 5% Max Health true damage to an enemy. | Every N seconds deal N% Max Health true damage to an enemy. | combat |
| Lightning Strike | Deal 10% Max Health true damage to all enemies. | Deal N% Max Health true damage to all enemies. | combat |
| Living Soil | Every N seconds, roots stun an enemy for 1 second. | Every N seconds, roots stun N enemies for 1 second. | combat |
| Lost Travelers | Gain 3 gold worth of random champions. | Gain 4 gold worth of random champions. | economy |
| Lucky 7 | Gain 7 shop rerolls. | Gain N shop rerolls. | economy |
| Major Polymorph | A random 4-cost you own permanently transforms into a 5-cost. | The highest star level 4-cost you own permanently transforms into a 5-cost. | economy |
| Mana Potion | Gain a temporary equippable Mana Potion. | Gain a temporary equippable Radiant Mana Potion. | combat |
| Mana-Rich Soil | Your Casters have N% reduced Max Mana. | same text, stronger numbers | combat |
| Marksmen's Gale | Your Marksmen's attacks grant 3% stacking Attack Speed. | Your Marksmen's attacks grant N% stacking Attack Speed. | combat |
| Marksmen's Marks | Your Marksmen have a 30% chance to double attack. | Your Marksmen have a N% chance to double attack. | combat |
| Mercenary Force | Gain 3% Damage Amp for every N gold you have. | Gain N% Damage Amp for every N gold you have. | combat |
| Middle Path | Reroll your shop with all 2-star 3-cost champions. Cost: 5 gold (patch 18.6). | Reroll your shop with all 2-star 3-cost champions. Cost: 4 gold (patch 18.6). | economy |
| Minor Polymorph | A random 1-cost you own permanently transforms into a 2-cost. | The highest star level 1-cost you own permanently transforms into a 2-cost. | economy |
| Mitosis | Gain a Tiny Champion Duplicator. | same text, stronger numbers | economy |
| Moonlight Ritual | Your lowest star 1-cost or 2-cost champion stars up in combat. | Your two lowest star 1-cost or 2-cost champions star up in combat. | combat |
| Moonrise | After 8 seconds, increase Lunar's effects by N%. | After 8 seconds, increase Lunar's effects by 175%. Offered only while Blossom is active (patch 18.6). | combat |
| Nature's Ally | Every N ally deaths, summon a random Riftbeast. | Every N ally deaths, summon a random Riftbeast that's sometimes 2-star. | combat |
| Nature's Wrath | Kill EVERYONE without an item after N seconds. | same text, stronger numbers | combat |
| Payday | For each ally champion that survives combat, gain N Gold. | same text, stronger numbers | economy |
| Peddler | Open a random item store. Random items cost 0 to N gold. | Open a random item store. Random items cost 0 to N gold. Gain N Gold. | economy |
| Petrified Dummy | Your first champion to die turns into a N Health dummy with their defenses. | same text, stronger numbers | combat |
| Petrify Shields | N% Slow enemies and reduce their shields by 20% for 10 seconds. | N% Slow enemies and reduce their shields by N% for 10 seconds. | combat |
| Phantom Armor | Gain a temporary Warmog's Armor. | Gain a temporary The Indomitable. | combat |
| Phantom Emblem | Gain a temporary emblem of your most active trait. | same text, stronger numbers | combat |
| Phantom Gloves | Gain a temporary Thief's Gloves. | Gain a temporary Mittens. | combat |
| Phantom Splash | Combat start: Up to N inactive traits activate at the lowest level. | Combat Start: Up to N inactive traits activate at the lowest level. | combat |
| Phantom Vest | Gain a temporary Bramble Vest. | Gain a temporary Aegis of Dawn. | combat |
| Plated Shields | Shields are 15% stronger. Allies gain 150 shield for 10 seconds. | Shields are N% stronger. Allies gain N shield for 10 seconds. | combat |
| Pocket Change | Your one-star champions drop N gold on death. | Your one-star champions drop N gold on death. Gain N gold. | economy |
| Polymorph | A random 2-cost you own permanently transforms into a 3-cost. | The highest star level 2-cost you own permanently transforms into a 3-cost. | economy |
| Potioncraft | Gain three temporary equippable Potions. | Gain three temporary equippable Radiant Potions. | combat |
| Potted Lifebloom | Gain a temporary Lifebloom. | same text, stronger numbers | combat |
| Potted Stonebark | Gain a temporary Stonebark Tree. | same text, stronger numbers | combat |
| Preppers | Your stacking items start next combat partially stacked. | Your stacking items start next combat extra stacked. | combat |
| Prolific Power | Your champions gain 4% Attack Damage and Ability Power. | Your champions gain N% Attack Damage and Ability Power. | combat |
| Propagate | Gain a copy of the first allied champion to die. | same text, stronger numbers | economy |
| Quicken | On cast, your champions gain 30% Attack Speed for 4 seconds. | On cast, your champions gain N% Attack Speed for N seconds. | combat |
| Radiantize | After 8 seconds, 1 random equipped item temporarily becomes Radiant. | After N seconds, N random equipped item temporarily becomes Radiant. | combat |
| Rain | Your team gains 1 Mana Regen. | Your team gains N Mana Regen. | combat |
| Refreshing Light | Gain a free reroll per star level of each unique Solar champion. | same text, stronger numbers | economy |
| Regeneration | Your team heals 1% max Health per second. | Your team heals N% max Health per second. | combat |
| Resistant | Tanks reduce instances of incoming damage by N. | same text, stronger numbers | combat |
| Returning Friend | Gain a temporary Training Dummy with two Giant's Belts. | Gain a Training Dummy with Three Giant's Belts. | combat |
| Revenge | Whenever an ally dies, your champions gain 4% stacking Damage Amp. | Whenever an ally dies, your champions gain N% stacking Damage Amp. | combat |
| Ride the Wave | The next 2 rerolls this round will have N+ champions fielded last combat. | The next N rerolls this round will have N+ champions fielded last combat. | economy |
| Rolling Bones | You have a N% chance to get a free reroll with each kill. | N% chance to get a reroll with each kill, and N more if you win. | economy |
| Roly-Polys | For each ally champion alive after N seconds, gain a reroll. | same text, stronger numbers | economy |
| Room for a Shroom | Gain N random mushrooms. | same text, stronger numbers | economy |
| Salvager | Any units you bench have items removed and broken apart. | Any units you bench have items removed and broken apart. Gain N Gold. | economy |
| Scrappy | Combat start: 2 champions without items gain a temporary recommended item. | Combat Start: N champions without items gain a temporary recommended item. | combat |
| Search Party | Each of your shops has a 4-cost in it this round until you buy one. | same text, stronger numbers | economy |
| Slow Study | For each ally champion alive after N seconds, gain N XP. | same text, stronger numbers | economy |
| Smurfing | Gain a random 5 cost champion. | Gain a random 5-cost champion. | economy |
| Snacktime! | The BFF eats enemies he damages under 16% Max Health. | The BFF eats enemies he damages under N% Max Health. | combat |
| Solar Gift | Gain a Leona, Sejuani, and Kayle | Gain a Leona, Sejuani, and Kayle. Offered only while Blossom is active (patch 18.6). | economy |
| Solitude's Cloak | Champions with no adjacent allies gain N% Max Health and 10% Damage Amp. | Champions with no adjacent allies gain N% Max Health and N% Damage Amp. | combat |
| Stand Alone | Champions alone in their row gain 15% Health and 15% Damage Amp. | Champions alone in their row gain N% Health and N% Damage Amp. | combat |
| Starfall | The first N times you reroll your shop this round, the lowest cost champion becomes 2-star. | same text, stronger numbers | economy |
| Starting Town | Reroll your shop with all 2-star 1-cost champions. | same text, stronger numbers | economy |
| Stealthy | Your 3 strongest Fighters or Assassins are untargetable for 8 seconds. | N random champions on the back row are untargetable for N seconds. | combat |
| Sunfire Sorcery | Burn and Wound enemies for 10 seconds. | Burn and Wound enemies for N seconds. | combat |
| Supercritical | Allies with Precision gain 25% Critical Strike Damage. | Allies with Precision gain 40% Critical Strike Damage. | combat |
| Tattered Armor | N% Shred and Sunder enemies for 10 seconds. | N% Shred and Sunder enemies for N seconds. | combat |
| Terraforming | Gain N seeds at start of combat. | same text, stronger numbers | combat |
| Thingamajig Bag | Gain N consumables that give permanent Ability Power or Mana Regen. | same text, stronger numbers | economy |
| Thingamajig Jar | Gain N consumable that gives permanent Ability Power or Mana Regen. | Gain N consumables that give permanent Ability Power or Mana Regen. | economy |
| Thingamajig Sack | Gain N consumables that give permanent Ability Power or Mana Regen. | same text, stronger numbers | economy |
| Three Me | Gain a random 2-star 3-cost champion. | same text, stronger numbers | economy |
| Tiger's Visit | Gain the Tiger Primal buff for this combat. | same text, stronger numbers | combat |
| Training Yard | Gain 2 Training Dummies each with a random tank item. | Gain N Training Dummies each with a random tank item. | economy |
| Treetop Archers | Archers fire from the trees, dealing N physical damage every N seconds. | same text, stronger numbers | combat |
| Tremors | Every N seconds, Stun enemies for 1 second. | Every N seconds, Stun enemies for N seconds. | combat |
| Trophy Hunter | Next player combat, your Rivals' takedowns count twice. | same text, stronger numbers | combat |
| Turtle's Visit | Gain the Turtle Primal buff for this combat. | same text, stronger numbers | combat |
| Ultra Ascension | After 25 seconds, allies gain 300% Damage Amp. | After N seconds, allies gain 300% Damage Amp. | combat |
| Verdant Vitality | Your champions gain 50 Health. | Your champions gain N Health. | combat |
| Wrapped In Thorns | Your Tanks reflect N% of damage taken to adjacent enemies every N seconds. | same text, stronger numbers | combat |
| Yordle Spirit | Allies have a 99% chance to dodge attacks for 4 seconds. | Allies have a N% chance to dodge attacks for N seconds. | combat |
