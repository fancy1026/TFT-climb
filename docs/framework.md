# Architecture

Purple: language model. Grey: code. Green: people. White: hand-over to another part.

## During a game

```mermaid
flowchart LR
  IN1(["Expert rules, comp notes,<br/>stage strategies"])
  IN2(["Harness code"])
  A["Game start · model<br/>writes its game plan"]
  C["Model call · code<br/>rules + notes + stage<br/>strategy + round state"]
  D["Player · model<br/>a plan per round"]
  E["Executor · code<br/>one checked action<br/>per tick"]
  F["Game record · code"]
  IN1 --> A --> C --> D --> E
  IN1 --> C
  E -- next decision --> C
  IN2 -.-> E
  E --> F
  classDef model fill:#ebe8fb,stroke:#6a5fcf,color:#2f2873
  classDef code fill:#eef1f4,stroke:#8a96a6,color:#18202b
  classDef hand fill:#ffffff,stroke:#8a96a6,color:#18202b
  class A,D model
  class C,E,F code
  class IN1,IN2 hand
```

## Between games

### Expert feedback

```mermaid
flowchart LR
  FB(["Domain expert"])
  R1["Rules · code<br/>first in the comp notes"]
  R2["Grades · code<br/>override the Reviewer's"]
  R3(["Harness points:<br/>to Harness fixes"])
  DS["Any agent · model<br/>disagrees only<br/>with reasons"]
  FB --> R1
  FB --> R2
  FB --> R3
  R1 -.-> DS
  R2 -.-> DS
  DS -. reasons .-> FB
  classDef model fill:#ebe8fb,stroke:#6a5fcf,color:#2f2873
  classDef code fill:#eef1f4,stroke:#8a96a6,color:#18202b
  classDef hand fill:#ffffff,stroke:#8a96a6,color:#18202b
  classDef person fill:#e0f2ee,stroke:#23806a,color:#0f4a3d
  class FB person
  class R1,R2 code
  class DS model
  class R3 hand
```

### Learn

```mermaid
flowchart LR
  T(["New comp or patch"])
  M["MetaTFT · code"]
  FD["Find strong players'<br/>games · code"]
  R["Read the videos · code<br/>round by round"]
  PA["Play-along · model"]
  CC["Comp coach · model<br/>comp notes, stage<br/>strategies"]
  O1(["To the game"])
  T --> M --> CC
  T --> FD --> R --> CC
  R --> PA --> CC
  CC --> O1
  classDef model fill:#ebe8fb,stroke:#6a5fcf,color:#2f2873
  classDef code fill:#eef1f4,stroke:#8a96a6,color:#18202b
  classDef hand fill:#ffffff,stroke:#8a96a6,color:#18202b
  class PA,CC model
  class M,FD,R code
  class T,O1 hand
```

### Review, every five games

```mermaid
flowchart LR
  I0(["Five game records"])
  V0["Packets · code<br/>stage by stage,<br/>luck measured"]
  V1["Stage comparison · model<br/>five games side by side,<br/>every decision graded"]
  V5["Meta review · model<br/>what recurs,<br/>what was luck"]
  O4(["To the Strategist"])
  O5(["To Harness fixes"])
  I0 --> V0 --> V1 --> V5
  V5 --> O4
  V5 --> O5
  classDef model fill:#ebe8fb,stroke:#6a5fcf,color:#2f2873
  classDef code fill:#eef1f4,stroke:#8a96a6,color:#18202b
  classDef hand fill:#ffffff,stroke:#8a96a6,color:#18202b
  class V1,V5 model
  class V0 code
  class I0,O4,O5 hand
```

### Strategist

```mermaid
flowchart LR
  I1(["Stages to change"])
  S1["Strategist · model<br/>rewrite, revert or keep"]
  J["Judge · code<br/>round cases, then strong<br/>players' decisions"]
  S2["Versions · code"]
  O6(["Next game"])
  I1 --> S1 --> J -- no worse --> S2 --> O6
  S1 -- revert or keep --> S2
  classDef model fill:#ebe8fb,stroke:#6a5fcf,color:#2f2873
  classDef code fill:#eef1f4,stroke:#8a96a6,color:#18202b
  classDef hand fill:#ffffff,stroke:#8a96a6,color:#18202b
  class S1 model
  class J,S2 code
  class I1,O6 hand
```

### Harness fixes

```mermaid
flowchart LR
  I2(["Faults and<br/>expert's points"])
  H1["Editor · model<br/>fixes at the cause"]
  H2["Round cases · code<br/>its case passes,<br/>none breaks"]
  H3{"New<br/>behaviour?"}
  H4(["Person running it<br/>says yes"])
  O9(["Next game"])
  I2 --> H1 --> H2 --> H3
  H3 -- no --> O9
  H3 -- yes --> H4 --> O9
  classDef model fill:#ebe8fb,stroke:#6a5fcf,color:#2f2873
  classDef code fill:#eef1f4,stroke:#8a96a6,color:#18202b
  classDef hand fill:#ffffff,stroke:#8a96a6,color:#18202b
  classDef person fill:#e0f2ee,stroke:#23806a,color:#0f4a3d
  class H1 model
  class H2,H3 code
  class I2,O9 hand
  class H4 person
```
