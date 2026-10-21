# write-like-myself

This skill helps AI agents write text that I would sign as my own. The aim is that no reader suspects a machine wrote it. It applies to any text that a person reads, from a paper or a grant proposal to a web page or an email. It does not apply to code, data or configuration.

## How it works

Readers notice machine writing because the text has no point to make and no reader in mind. Without a point or a reader, a model falls back on ready-made phrases and familiar habits, and readers soon learn to spot them. A list of habits to avoid treats only these symptoms, and the habits change from one model to the next. So the skill is not a list of rules. It explains what writing is and what good writing asks of the writer. An agent that settles its point and its reader before it writes has no use for these habits.

The skill draws on four works. These are David Lindsay's *Scientific Writing = Thinking in Words*, Joshua Schimel's *Writing Science*, William Strunk Jr's *The Elements of Style* and the ASD-STE100 standard for Simplified Technical English. It holds nothing specific to me or my field, so anyone can use it.

## Install

In a terminal, run this command.

```sh
claude plugin install write-like-myself --marketplace yi-li-yang/write-like-myself
```

Inside a Claude Code session, run these two commands instead.

```text
/plugin marketplace add yi-li-yang/write-like-myself
/plugin install write-like-myself@write-like-myself
```

For another agent, give it the file [SKILL.md](plugins/write-like-myself/skills/write-like-myself/SKILL.md) as instructions.

## Testing

I tested the skill on eight writing tasks, two each from papers, letters, web text and everyday email. Each task was written twice, once with the skill and once with the writing rules my agents followed before. I read the sixteen texts unlabelled and in a shuffled order, and I would use all sixteen without changes. I then compared the two texts for each task side by side. I preferred the skill's text in three pairs and the old rules' text in four, and I could not choose in one. The agent chose to read the skill in all eight runs where it was installed.

The tools folder holds the scripts for these tests and a guard that keeps private material out of this repository. The scripts read texts that stay on my computer, so they will not run elsewhere.

## Licence

MIT. See [LICENSE](LICENSE).
