# Notes — How This Works

These are your notes. Claude does not write here.

The point is to force you to articulate things in your own words.
If you cannot write it down here, you do not understand it yet.

---

## File conventions

| File | When to write it |
|---|---|
| `project1_design.md` | Before writing any code for Project 1 |
| `project2_design.md` | Before writing any code for Project 2 |
| `project3_design.md` | Before writing any code for Project 3 |
| `daily_YYYY-MM-DD.md` | End of each working session — what you did, what confused you |
| `concepts/` | One file per concept you want to lock in (e.g., `dpo_loss.md`, `lora.md`) |

---

## Design note template (copy this for each project)

```
# Project N Design Note
Date:

## Task definition
What exactly is the task? What does the model receive as input?
What is the expected output format?

## Dataset
Where does the data come from?
How many train / eval examples?
What did I see when I inspected 20 examples manually?

## Eval
What is the single most important metric?
How do I know if training worked?
What does failure look like?

## Architecture decisions
Model: why this checkpoint?
LoRA config: why these ranks / targets?
Prompt template: what is it exactly?

## Open questions before I start
(things I am not sure about yet)
```

---

## Daily note template

```
# Daily Note — YYYY-MM-DD

## What I did
-

## What worked
-

## What confused me / what I still don't understand
-

## What I'll do next session
-
```

---

## Rule
If you find yourself about to ask Claude to explain something — write a draft explanation here first.
Then ask Claude to check it. That way you are learning, not outsourcing.
