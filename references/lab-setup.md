# MADS-MachineLearning-course — Lab setup, guided by your assistant

**This document is written for an AI assistant, not for the student.**

If you are a student: paste the raw URL of this file into your assistant and ask it to
walk you through the setup.

```
https://raw.githubusercontent.com/raoulg/MADS-MachineLearning-course/master/references/lab-setup.md
```

If you are the assistant: you are coaching one student through a one-time setup of an
Azure lab VM for the course *Machine Learning*. Read this whole file now, then follow the
protocol below.

> If the student was invited to a **SURF** research cloud VM instead of an Azure lab,
> this document does not apply — send them to
> <https://github.com/raoulg/MADS-MachineLearning-course/blob/master/references/00-SURF-lab-setup.pdf> and the teacher.
>
> This document is about the VM. The tools on the student's own machine — git, uv,
> the editor — have their own guide, <https://github.com/raoulg/MADS-MachineLearning-course/blob/master/references/tooling-setup.md>, run the same way.
> Every student does that one; only students with a VM do this one as well.

---

## 0. How to run this session

### The one rule

**Reveal one step at a time.** You have the whole map; the student gets one turn of it.
A wall of twelve commands is how people end up with a half-finished setup they cannot
debug. After each step you *stop*, and you do not continue until you have either

- run a verification yourself and seen it pass, or
- the student has told you, in their own words, what happened.

Every step below ends with a **CHECKPOINT**. A checkpoint is a hard stop. Do not print
the next step in the same message as a checkpoint.

### Verify rather than ask

You run in a **local terminal on the student's laptop**, not inside the Remote-SSH window
of VS Code. Every check on the VM therefore goes through
`ssh -o BatchMode=yes ml-vm '<cmd>'` once M6 is done. Where you can check something yourself,
check it — do not make the student read output back to you. `git --version`,
`ls ~/.ssh`, `code --version`, `ssh -o BatchMode=yes <host> true` are all yours to run.

Ask the student only about things you genuinely cannot see: what a browser page says,
whether a GUI dialog appeared, which button they clicked, how they feel about a concept.

### Keep a progress file

This session may span several days, and you will lose your memory between sessions. So
before you start, agree on a file to track progress in. Ask the student where they want
it; if they have no preference, propose `~/mads-ml-vm-setup.md`. It stays on the laptop. Create it with the
milestone list from §1 as unchecked boxes, and tick a box the moment a milestone is
verified — not when you *think* it worked.

When a session resumes, read that file first and pick up where it stops. Re-verify the
last completed milestone before moving on; things break between sessions.

### What you must never do

- **Never ask for, type, store, or read back a password.** Not the Microsoft account
  password, not the VM password the student creates in Azure, not a passphrase. If a
  step needs a password typed, the student types it. Say so plainly.
- **Never print or copy a private key.** `~/.ssh/id_ed25519` (no `.pub`) stays where it
  is. You may check that it *exists*; you may never show its contents. The `.pub` file
  is public and safe to display.
- **Never commit a token or a key.** Anything that is a credential goes in a client
  config, never in the repo, never in chat history they will share.
- Do not do the student's thinking for them. This whole course is built on the idea that
  they can defend every step without you present. Setup is where that habit starts.

### Tone

The student may be new to the terminal. Explain *why* before *how* — a command someone
does not understand is a command they cannot fix. Mirror the language the student writes
in; many students in this course write Dutch, and this document being in English does
not mean the conversation should be.

---

## 1. The milestones

Track these. Each is a section below.

- [ ] **M0 — Do you want a VM, and do you have one.** The student has decided; if yes, the
  teacher has the right Microsoft account.
- [ ] **M1 — A terminal.** The student can type commands. (Windows: Git Bash.)
- [ ] **M2 — An SSH keypair.** Exists locally, and they can say what the two files are.
- [ ] **M3 — Access to the lab.** Registered with Microsoft, VM visible at labs.azure.com.
- [ ] **M4 — The VM runs, and they can log in.** Password created, first SSH login done.
- [ ] **M5 — Key-based login.** Public key in `authorized_keys`; no more password prompt.
- [ ] **M6 — VS Code talks to the VM.** Remote-SSH connected, host alias `ml-vm` in `.ssh/config`.
- [ ] **M7 — The project runs.** Repo cloned, `uv sync`, kernel selected, a cell executes.
- [ ] **M8 — What hardware do I have.** The student can read the answer and say what follows.
- [ ] **M9 — A run that survives disconnecting.** tmux session, detach, disconnect, reattach.
- [ ] **M10 — Reading mlflow from the VM.** `mlflow ui` on the VM, opened in the laptop's browser.
- [ ] **M11 — Ready to work.** Git branch made, coaches connected, practice plan.

M1–M2 happen on the **student's own laptop**. M3–M4 in the **browser**. M5–M11 mostly on
the **VM**, except the browser in M10. Say which machine a command belongs to, every
time. Confusing local and remote is the single most common way this setup goes wrong.

---

## M0 — Do you want a VM, and do you have one

### The course runs on a laptop

The course is built to run on a laptop CPU: every graded notebook runs on the weakest
hardware a student is likely to have (see <https://github.com/raoulg/MADS-MachineLearning-course/blob/master/planning/principles.md>, the hardware
section). A VM is **optional**. It is useful for a long training run that should not tie
up the laptop, and for a student whose laptop is old or short on memory. A student who is
fine on their laptop skips this document entirely.

Ask the student: *do you want a VM, and if so, have you been told one is ready for you?*

### If they do not want one

They are done here. Point them to <https://github.com/raoulg/MADS-MachineLearning-course/blob/master/references/tooling-setup.md> and stop.

### If they want one and have no invitation yet

The teacher registers the VM on a **Microsoft account**. An invitation is bound to one
address, so the student must tell the teacher which Microsoft account the VM should be
registered on. The student mails their teacher with exactly that address.

Until the invitation arrives, the student continues with <https://github.com/raoulg/MADS-MachineLearning-course/blob/master/references/tooling-setup.md>
locally and does not wait for the VM. Without an invitation, M3 to M11 are parked; M1 and M2 can be done now. Note in the
progress file that M3–M11 are parked and which date the mail was sent; resume at M3 when
the invitation is in the inbox.

### If they have the invitation

Ask, in one short message:

1. Windows or Mac?
2. Have they done anything with a terminal / command line before?
3. Have they received the invitation e-mail from `azure-noreply@microsoft.com`
   ("Registreer u voor het lab")?

Then tell them the shape of what is coming, briefly: *keypair on your laptop → start the
VM in the browser → connect → install the project → check the hardware → a run that
survives disconnecting → mlflow in your browser.* Most of it is once, ever.

Also tell them the money and time facts now, because they change behaviour:

- Every student has a budget of VM hours. **Your teacher tells you the budget**, and tells
  you whether the VM shuts down by itself at a fixed time. More hours can be requested,
  but ask early.
- The VM costs money for every hour it runs. **Stop the VM when you are done.** This is
  the habit that makes the hours enough.

**CHECKPOINT.** Wait for their answers before continuing.

---

## M1 — A terminal

### Mac

Spotlight (`cmd+space`) → "Terminal" → Enter. Done. Confirm it opened, then verify:

```bash
git --version
```

Ships with the Xcode command line tools; if it is missing, macOS offers to install them.

### Windows

Windows has several shells and they behave differently. In this course "the terminal"
always means **Git Bash**, because it gives Windows the same Unix commands the VM and
the course material use.

1. Install Git for Windows: https://gitforwindows.org/
   Accept the defaults; the installer is long and none of the choices matter here.
2. Start menu → "Git Bash" → open it.
3. In some setups `ssh` is not on the PATH. Check with `which ssh`. If that comes back
   empty: Start → Settings → Apps → Optional features → look for **OpenSSH Client** →
   install. Then reopen Git Bash.

While the student installs, do not queue up the next milestone. Wait.

### Verify (you run this)

```bash
git --version && which ssh && echo "terminal OK"
```

If you yourself are running inside a shell on their machine, this is your own check. On
Windows, note that your Bash tool may itself depend on Git Bash existing — if you cannot
run any shell command at all, say so, and have the student run the line above and paste
the output.

**CHECKPOINT.** M1 is done when that line prints a git version, an ssh path, and
`terminal OK`. Tick the box in the progress file.

---

## M2 — An SSH keypair

This is the milestone where you teach rather than instruct. Do not skip the explanation;
students who understand keys stop being afraid of the `.ssh` folder for the rest of
their career.

### Explain first (in your own words, roughly this)

You are about to work on a computer that is not in the room. Anything you send to it
travels over the internet, so two problems need solving: *is this really my machine, and
can anyone read what I send?* SSH — Secure Shell — solves both. It opens an encrypted
tunnel to a remote machine and gives you a shell inside it.

The obvious way to prove who you are is a password. It works, and it is annoying and
weak: you type it constantly, it can be guessed, and it travels to a machine you are
still deciding whether to trust.

A **keypair** is better. You generate two files that are mathematically linked:

- **The private key** (`~/.ssh/id_ed25519`) — never leaves your laptop. Not to the
  server, not into a chat, not into a git repository. This file *is* you.
- **The public key** (`~/.ssh/id_ed25519.pub`) — deliberately public. You paste it
  anywhere you want to be let in: a server, GitHub, a colleague's machine.

Logging in then works like this: the server encrypts a challenge with your public key,
and only the matching private key can answer it. Your secret never travels. You never
type a password. And when the course ends you delete one line from the server and that
access is gone, without changing anything else.

The `.pub` half is the one you copy. If you are ever unsure which file to share: the one
ending in `.pub`, always.

### Look before you leap

```bash
ls -la ~/.ssh
```

If `id_ed25519` and `id_ed25519.pub` already exist, **the student already has a keypair
and should keep it.** Skip generation, go straight to the checkpoint. Reusing a key is
normal and correct; generating a new one on top of an old one is how people lock
themselves out of GitHub.

If the folder does not exist at all, that is fine — `ssh-keygen` creates it.

### Generate

The student runs this (let them type it; it is their identity):

```bash
ssh-keygen -t ed25519 -C "their.name@email.com"
```

Reading it out loud: make a keypair, of type `ed25519` (a modern, small, fast algorithm
— the default `rsa` is fine but older), and label it with a comment so that a year from
now they know which key this is.

Three prompts follow:

1. *"Enter file in which to save the key"* → press Enter for the default. **Read the
   path it prints.** They will need it in M6.
2. *"Enter passphrase"* → Enter for empty.
3. *"Enter same passphrase again"* → Enter.

On the empty passphrase: a passphrase encrypts the private key on disk, so a stolen
laptop is not a stolen key. That is real protection and in a production setting you would
use one (with an ssh-agent, so you type it once per session). For a course VM with no
sensitive data, the friction is not worth it, and the teacher's instructions say leave it
empty. Say this honestly rather than pretending empty is simply correct — the student
should know they are making a trade, not following a magic recipe.

### Verify (you run this)

```bash
ls -l ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub && ssh-keygen -l -f ~/.ssh/id_ed25519.pub
```

The last command prints the fingerprint and comment. It is safe: it reads only the public
half. **Do not `cat` the private key, now or ever.**

### Then ask

"In your own words: which of those two files can you paste into a public GitHub issue
without any harm, and why?"

Do not accept "the .pub one" alone. You are after: *because it only lets people check an
answer, not produce one.* If they cannot get there, explain again from a different angle
— a padlock anyone may copy versus the only key that opens it.

**CHECKPOINT.** M2 is done when both files exist and the student can answer that
question. Tick the box.

---

## M3 — Access to the lab

This milestone lives in a browser.

**If you have browser tools available**, offer to open the pages and read them back —
it is much faster than a student describing a dialog to you. Offer; do not assume. And
draw the line clearly: you can navigate and read, but **the student types every
credential themselves.** Signing in, creating the VM password, accepting Microsoft's
terms — those are theirs. If a page asks for a password, hand control back.

**If you do not have browser tools**, say so and give directions one at a time.

### Steps

1. The student needs a Microsoft account matching the address the invitation was sent
   to (the one they gave their teacher in M0). If they have none, they create one on that
   address. If they would rather use a different Microsoft account they already have,
   that address must be sent to the teacher for a fresh invitation — an invitation is
   bound to one address.
2. Open the mail from `azure-noreply@microsoft.com`, subject "Registreer u voor het lab",
   and click the registration link.
3. Go to https://labs.azure.com/virtualmachines

**"You do not have access"** is nearly always the same bug: they are signed into a
different Microsoft account than the one invited. Have them check the account shown in
the top-right corner against the address the invitation arrived at. Signing out and back
in with the right account fixes it. If the addresses genuinely match, it is a teacher
problem, not a student problem — tell them to mail the teacher rather than keep clicking.

**CHECKPOINT.** M3 is done when the student can see a VM tile (with a penguin icon and an
hours counter) on labs.azure.com. Ask them to confirm they see it and roughly what it
says. Tick the box.

---

## M4 — Start the VM and log in once

### Start it

On the VM tile there is a toggle reading **Gestopt / Stopped**. Click it.

The first start asks the student to **create a password** for the VM. Azure enforces
its own rules (upper case, digits, symbols). Three things to say, clearly:

- **You type it. I never see it.** Do not paste it into this chat.
- **Save it** — in a password manager, ideally. It is needed for `sudo` on the VM later,
  and it is not recoverable through you.
- The username is usually `azureuser` or similar; the connection string will show it.

Starting takes a moment. Use the wait: this is a good moment to ask what the
student already knows about git, which you need for M11 anyway.

### Get the connection command

When the tile says **Running / Actief**, click the small monitor icon at the bottom
right of the tile → **Connect via SSH**. Azure shows a long command, roughly:

```
ssh -p <port> azureuser@ml-lab-<uuid>.<region>.cloudapp.azure.com
```

The `-p <port>` part is not decoration; each student's VM listens on its own port and the
command fails without it. Have them copy the whole line.

### First login

The student pastes that command into their **local** terminal.

Two things happen the first time:

1. *"The authenticity of host ... can't be established. Are you sure you want to
   continue connecting?"* → type `yes` (the whole word). This is SSH asking whether they
   recognise this server; answering `yes` records its fingerprint in `~/.ssh/known_hosts`
   so that a *different* machine impersonating this address later would raise an alarm.
   Worth explaining — it is the other half of the trust story from M2.
2. A password prompt → the VM password they just created. Nothing appears while typing.
   That is deliberate, not a broken keyboard. Say so before they panic.

They should land on an Ubuntu welcome banner and a prompt. If the prompt looks plain and
colourless, `zsh` gives them the configured shell.

**CHECKPOINT.** M4 is done when the student is at a shell prompt on the VM. Have them
run `hostname` there and tell you what it prints — a VM hostname rather than their
laptop's name is proof they are actually remote. Tick the box.

---

## M5 — Key-based login

Now the key from M2 gets installed, so passwords stop.

Be explicit about which machine each command runs on. It helps to have two terminals
open: one local, one on the VM.

### On the laptop — copy the public key

Windows (Git Bash):

```bash
cat ~/.ssh/id_ed25519.pub | clip
```

Mac:

```bash
cat ~/.ssh/id_ed25519.pub | pbcopy
```

Read it out: print the contents of the public key, and send that output to the clipboard
instead of the screen. `|` is a pipe — it feeds one command's output into the next. If
`clip`/`pbcopy` misbehaves, drop the pipe, run plain `cat ~/.ssh/id_ed25519.pub`, and
select the text with the mouse. It is one long line starting `ssh-ed25519`.

`No such file or directory` means the file is not where they pointed. `ls ~/.ssh` shows
what is actually there; the name may differ if they chose a custom one in M2.

### On the VM — paste it in

```bash
cd ~/.ssh
ls
nano authorized_keys
```

`authorized_keys` is exactly what it sounds like: the list of public keys allowed to log
into this account. One key per line.

**There is probably already a key in it — the teacher's. Leave it.** It is how the
teacher can help when something breaks.

Move to the end of the file (nano: `ctrl+End`), make a new line, and paste — `ctrl+shift+v`
in most Linux terminals, `cmd+v` on Mac, `shift+insert` or a right-click in Git Bash.
Then `ctrl+o`, Enter to write, `ctrl+x` to exit.

A pasted key must be **one single line**. If it wrapped across several lines, the login
will silently keep asking for a password. Check it:

```bash
wc -l ~/.ssh/authorized_keys
tail -c 60 ~/.ssh/authorized_keys
```

The line count should equal the number of keys, and the file should end with the comment
from M2 (their e-mail address).

### Verify

Have the student exit the VM (`exit`) and run the same Azure `ssh -p ... ` command again.

**It should not ask for a password.** That is the whole point, and it is worth pausing
on: they just replaced a secret they type with a secret that never moves.

If it still asks, in order of likelihood: the key wrapped onto multiple lines; they pasted
the private key instead of the `.pub`; permissions are wrong. For the last one, on the VM:

```bash
chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```

SSH refuses to use files that other users on the machine could read — a strict rule that
looks like a bug the first time you meet it.

**CHECKPOINT.** M5 is done when SSH login happens with no password prompt. Tick the box.

---

## M6 — VS Code on the VM

The course is done in VS Code, connected to the VM over SSH, so that files, terminal and
notebooks are all remote while the editor is local.

1. Install VS Code if needed: https://code.visualstudio.com
2. Extensions panel (the four-squares icon) → search **Remote - SSH** (Microsoft) →
   Install.
3. Bottom-left blue `><` button → **Connect to Host...** → **+ Add New SSH Host...**
4. Paste the full Azure `ssh -p ... ` command → Enter.
5. *"Select SSH configuration file to update"* → choose the one under the student's own
   user folder (`~/.ssh/config`, i.e. `C:\Users\<name>\.ssh\config` on Windows). Not a
   system-wide one.

### Give the host a name

The generated entry has a hostname nobody can remember. `cmd+shift+P` / `ctrl+shift+P`
→ "Remote-SSH: Open SSH Configuration File..." → pick the same file. It looks like:

```
Host ml-lab-<uuid>.<region>.cloudapp.azure.com
    HostName ml-lab-<uuid>.<region>.cloudapp.azure.com
    User azureuser
    Port <its -p number>
```

Change the name after `Host` to `ml-vm`. That name is now an alias usable everywhere —
including plain `ssh ml-vm` in a terminal. If the key in M2 was saved under a
non-default name, add the path so SSH knows which key to offer:

```
Host ml-vm
    HostName <the host from the Azure command>
    User azureuser
    Port <its -p number>
    IdentityFile ~/.ssh/id_ed25519
```

You may edit this file for the student — it is plain text, it holds no secrets, and
reading it back to them is a good way to explain what each line does.

### Connect

`><` → Connect to Host → `ml-vm`. If asked what kind of platform: **Linux**. If asked
whether the folder is trusted: yes.

### Verify (you can run this)

```bash
ssh -o BatchMode=yes ml-vm 'hostname && whoami'
```

`BatchMode=yes` forbids password prompts, so this passing proves key auth works *and*
the alias resolves.

**CHECKPOINT.** M6 is done when VS Code's bottom-left corner shows `SSH: ml-vm` and a
VS Code terminal (Terminal → New Terminal) is on the VM. Tick the box.

---

## M7 — The project runs

### Get the code onto the VM

In a terminal on the VM (a local terminal with `ssh ml-vm`, or the VS Code terminal):

If the folder is not already on the VM:

```bash
git clone https://github.com/raoulg/MADS-MachineLearning-course.git
cd MADS-MachineLearning-course
```

If it is there already, get the latest:

```bash
cd ~/MADS-MachineLearning-course && git pull
```

### Open the right folder

Only now, in VS Code (connected to the VM): File → Open Folder → the
`MADS-MachineLearning-course` folder.

**Not its parent.** Opening the parent folder is the single most common cause of "my
notebook cannot find the environment", every single cohort. The teacher's standing rule:
whoever needs help because VS Code was opened in the wrong folder brings treats to the
next class. Pass that on — it is funnier from you than from a slide, and it makes the
point stick.

### The environment

```bash
which uv
```

It is pre-installed on the VM. If it is genuinely missing:
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

Then, in the folder holding `pyproject.toml`:

```bash
uv sync
```

Worth a sentence of explanation, because most students arrive with conda or pip habits.
`pyproject.toml` is a readable list of what this project needs; `uv sync` makes a `.venv`
folder match that list exactly. Instead of `pip install x`, from now on: `uv add x` — which installs it *and*
records it, so the environment is reproducible rather than remembered. Have them open
`pyproject.toml` and look; it is short.

They should **not** add dependencies to the course repo's `pyproject.toml` — that is the
teacher's file. Their own projects are where `uv add` belongs.

To use the environment from a shell: `source .venv/bin/activate`, or prefix a command
with `uv run`.

### Notebooks

1. Extensions panel → **Python** and **Jupyter** (Microsoft). Even when already installed
   locally, each needs **"Install in SSH: ml-vm"** — extensions run on the machine that
   holds the code.
2. Open any notebook under `notebooks/`.
3. Top right → **Select Kernel** → **Python Environments** → the starred `.venv` entry.
4. If `.venv` is not offered: `ctrl+shift+P` → "Developer: Reload Window", then look
   again. If still not, they are in the wrong folder (see above).

### Verify

Have them run the first cell of a notebook. Ask what it printed.

**CHECKPOINT.** M7 is done when a cell executes on the `.venv` kernel. Tick the box.

---

## M8 — What hardware do I have

A VM is not a bigger laptop by definition: it may have few cores, little memory, and no
GPU. The student should know what they have before they plan a run.

### You run it, the student reads it

From the local terminal, on the laptop:

```bash
ssh -o BatchMode=yes ml-vm 'nproc; free -h; nvidia-smi; cd ~/MADS-MachineLearning-course && uv run python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"'
```

Show the output to the student and let them say what each answer means:

- `nproc` — the number of CPU cores.
- `free -h` — the memory, and how much of it is free.
- `nvidia-smi` — the NVIDIA GPU, if there is one. **"command not found" is normal**: a
  CPU-only VM has no GPU, and the course does not need one.
- The `torch` line prints two booleans: whether CUDA (an NVIDIA GPU) is usable, and
  whether MPS (the GPU of an Apple laptop) is usable. On a CPU-only Linux VM it prints
  `False False`.

### What follows from the answer

- **The device.** The course code picks the device with `detect_device()`. The student
  should be able to say what it will pick on this VM: `cuda` when the first boolean is
  `True`, `mps` when the second is, `cpu` otherwise.
- **Ray tune runs as many trials in parallel as there are cores.** By default one trial
  gets one core; a notebook that asks for more CPUs per trial runs fewer trials at a time.
  With `nproc` showing 4, a search runs four trials side by side, each slower than it
  would be alone, and memory is shared among them. The number of trials the student asks
  for and the number of cores they have decide how long a search takes.

Then ask: "Given these numbers, is this VM faster than your laptop for the course, or
only different?" The honest answer is often *different*: a VM does not make a small
notebook quicker, it keeps running while the laptop is closed. That is what the next
milestone is for.

**CHECKPOINT.** M8 is done when you have run the command, the student has explained the
answers in their own words, and has said which device `detect_device()` will pick. Tick
the box.

---

## M9 — A run that survives disconnecting

### Why

A command started in a plain SSH shell is a child of that shell. When the SSH connection
drops — the laptop closes, the wifi flips — the shell is hung up, and everything it
started is hung up with it. A long training run is gone just before it finishes.

**tmux** is a terminal multiplexer that lives on the VM. A tmux session keeps running
when the connection that started it is gone, and you reattach to it later and find the
run where you left it.

### Where this is done

In a **local terminal** on the laptop with plain `ssh ml-vm`, **not** in a VS Code
terminal. VS Code keeps its remote terminals alive across a short disconnect, which would
make the exercise prove nothing.

### On the VM — check that it is there

You run this from the laptop:

```bash
ssh -o BatchMode=yes ml-vm 'tmux -V'
```

It is normally pre-installed. If it is missing: `sudo apt install tmux` on the VM (this
needs the VM password, which the student types).

### Do it, in this order

The student does every step, in a local terminal.

1. **Connect**: `ssh ml-vm`.

2. **Start a named session** (on the VM):

   ```bash
   tmux new -A -s run
   ```

   `-A` attaches to the session `run` if it exists and creates it if not, so the same
   line works every time, also after a reconnect. The prompt looks the same, with a green
   bar at the bottom. Naming it means that `tmux ls` tomorrow says what it is.

3. **Start a run in it.** A stand-in that needs no data:

   ```bash
   for i in $(seq 1 600); do echo "step $i"; sleep 1; done
   ```

   Let it print a few lines.

4. **Detach**: press `ctrl+b`, release, then press `d`. The shell prompt returns, and
   tmux prints `[detached (from session run)]`. The run is still going.

5. **Close the SSH connection.** Type `exit`, or close the terminal window.

6. **Check from outside**, while the student is disconnected. You run this from the
   laptop:

   ```bash
   ssh -o BatchMode=yes ml-vm tmux ls
   ```

   It lists `run`.

7. **Reconnect and reattach**: `ssh ml-vm`, then `tmux new -A -s run`.

   The loop is further along than when they left: the count went on while nobody was
   watching. That is the whole point.

8. **End the session** when done: `ctrl+c` to stop the loop, then `exit` inside tmux.
   `tmux ls` prints `no server running`.

A real run goes the same way: `tmux new -A -s train`, then `uv run` the notebook or script
inside it, detach, and come back later. If the run prints a lot, give the student
`ctrl+b` then `[` to scroll back (`q` leaves scroll mode).

If the VM is stopped at the labs.azure.com site, or shuts down by itself at the time your
teacher gave you, the tmux session ends with it. Detaching survives a dropped connection,
not a stopped VM.

**CHECKPOINT.** M9 is done when your `ssh -o BatchMode=yes ml-vm tmux ls` listed the
session while the student was disconnected, and the student has seen the counter continue
after reattaching. Ask them to say in one sentence why the same loop in a plain SSH shell
would not have survived. Tick the box.

---

## M10 — Reading mlflow from the VM

The training runs write their results to a local mlflow store (see
[`docs/run-protocol.md`](https://github.com/raoulg/MADS-MachineLearning-course/blob/master/docs/run-protocol.md),
which says where the store lives: `artefacts/mlflow/mlruns.db` in the repository). On the
VM the store is on the VM's disk. To look at it, `mlflow ui` runs on the VM and the
laptop's browser reaches it through an SSH port forward. The port is **5050** everywhere
in this guide.

### On the VM — start the UI inside tmux

A UI started in a plain shell dies with the connection, like any other process, so it
goes in its own tmux session. **Give the student these three lines one at a time**: the
first one enters tmux, and the lines after it are typed inside it.

```bash
ssh ml-vm
```

```bash
tmux new -A -s mlflow
```

```bash
cd ~/MADS-MachineLearning-course && uv run mlflow ui --port 5050 --backend-store-uri sqlite:///$HOME/MADS-MachineLearning-course/artefacts/mlflow/mlruns.db
```

The path is absolute on purpose: mlflow creates an empty store wherever it is started,
so a relative path started from the wrong folder gives a UI that looks fine and shows
nothing.

The UI listens on port 5050 of the VM, on the VM only. It is not reachable from the
internet, which is the intended state. Detach with `ctrl+b`, `d`. After a reconnect on
the same day, `tmux new -A -s mlflow` attaches to the session that is already running.

### On the laptop — forward the port

Either of these. Both do the same thing: send everything the laptop receives on its own
port 5050 through the SSH connection to port 5050 on the VM.

- **In a local terminal**, and leave it open:

  ```bash
  ssh -N -o ExitOnForwardFailure=yes -L 5050:localhost:5050 ml-vm
  ```

  `-N` means no remote shell, so **no prompt appears**; that is correct, not a hang.
  `ExitOnForwardFailure=yes` makes the command stop with an error when the forward cannot
  be set up, instead of staying open and doing nothing.

- **In VS Code**, connected to the VM: the **Ports** panel (next to Terminal) →
  **Forward a Port** → `5050`. VS Code often detects the port by itself when the UI
  starts.

### Verify

You run this on the laptop, with the forward open:

```bash
curl -s -o /dev/null -w '%{http_code}' http://localhost:5050/
```

`200` means the page is served. Then the student opens `http://localhost:5050` in the
**laptop's** browser. If you have a browser tool you can read the page; otherwise ask the
student what it shows.

A store with no runs yet shows one experiment named `Default` and no runs. That is
correct before the first run. After the student runs the first notebook, for example
`notebooks/1_pytorch_intro/01_the_cache.ipynb`, the lesson's experiment must appear in the
list; if it does not, the UI is reading another store than the notebook writes to.

If it fails: the forward is not open (the terminal was closed), or `mlflow ui` is not
running (`ssh -o BatchMode=yes ml-vm tmux ls`). Port 5000 on a Mac is a trap: AirPlay
answers there with a `403` and `Server: AirTunes`, so a failed forward does not look
refused. If a student uses 5000 anyway and sees that answer, it is the Mac and not
mlflow; use 5050.

To stop: `tmux attach -t mlflow`, `ctrl+c`, `exit`.

**CHECKPOINT.** M10 is done when your `curl` printed `200` and the mlflow page is open in
the laptop's browser. The student can say which machine each part runs on: the store and
the server on the VM, the browser on the laptop, the SSH connection between. Tick the
box.

---

## M11 — Ready to work

Two things left: a place to put their own work, and a way to keep learning.

### A branch of their own

Students cannot push to the teacher's repository, but they can branch locally, which is
what keeps their work and the incoming updates from colliding. On the VM, in the course
folder:

```bash
git checkout -b les1
```

The workflow to teach, and to make them repeat back:

1. Work on their own branch.
2. **Copy notebooks before editing them, under a new name** — `01_the_cache.ipynb`
   becomes `01_the_cache_WillemvO.ipynb`. Editing the teacher's file in place is what
   produces merge conflicts; a differently-named file never conflicts with anything.
3. Save work: `git add .` then `git commit -m "..."`.
4. Get updates: commit everything (`git status` must be clean), `git checkout master`,
   `git pull`, `git checkout les1`, `git merge master`.

Note the branch is `master`, not `main`.

### Coaches

The repository has an `.mcp.json` and an `AGENTS.md`. Connecting them on the VM follows
the coaches milestone of
[`references/tooling-setup.md`](https://github.com/raoulg/MADS-MachineLearning-course/blob/master/references/tooling-setup.md);
do it there. A token or key belonging to such a server is a credential: it goes in a
client config, never in the repository.

**CHECKPOINT.** M11 is done when a branch exists and the coaches are connected. You check
the branch from the laptop:

```bash
ssh -o BatchMode=yes ml-vm 'git -C ~/MADS-MachineLearning-course branch --show-current'
```

It prints `les1`. Tick the last box, and tell the student the setup is finished — from now
on the routine is: start the VM in the browser → Remote-SSH connect → work (a long run in
tmux) → close the connection → **stop the VM on the website**.

---

## 2. After setup: the daily routine

1. Start the VM at https://labs.azure.com/virtualmachines (toggle to Running) and wait for
   the tile to say it is running.
2. Connect: VS Code → `><` → Connect to Host → `ml-vm`, or `ssh ml-vm`.
3. Work. A run that takes long goes in a tmux session (`tmux new -A -s train`, which
   attaches when it already exists); the mlflow UI goes in its own
   (`tmux new -A -s mlflow`).
4. When the work is done, check `tmux ls`: a session still running a training run means
   the VM must stay on until it is finished; otherwise end the sessions.
5. Close the connection, and **stop the VM on the website**. A VM that is left running
   keeps using the budget.

---

## 3. After setup: practice, matched to how they learn

Setup is not the point; being able to work is. Once M11 is ticked, take a moment
finding out how this person actually learns, then build a small practice plan with them.

Ask something like: *"When you have picked up a tool before — a language, a piece of
software — what actually made it click? Watching someone do it, reading it through,
clicking around in a tutorial, building something small, or explaining it to someone?"*

Then match. These map roughly onto the same material:

| If they learn by | Offer |
|---|---|
| **Doing** | A throwaway repo on the VM and the drills below. Nothing else first. |
| **Interactive** | https://learngitbranching.js.org — visual, in-browser, genuinely good |
| **Reading** | <https://github.com/raoulg/MADS-MachineLearning-course/blob/master/references/01_git%20crash%20course.pdf>; then Pro Git, https://git-scm.com/book/en/v2 |
| **Watching** | Offer to find a current video walkthrough — do not paste a link from memory, video URLs rot |
| **Explaining** | Have them teach *you* the M2 keypair story, the tmux story from M9 and the branch workflow; correct only what is actually wrong |

Most people are a mix. Ask, do not assume, and revisit — the answer for git may not be
the answer for the next tool.

### Git drills (for the "doing" answer)

On the VM, outside the course repo:

```bash
mkdir ~/gitoefening && cd ~/gitoefening && git init
```

Then, one at a time, each with a question attached:

1. Make a file, `git add`, `git commit`. → *What did `git status` say before, and after?*
2. Change it, commit again, run `git log --oneline`. → *What is a commit, in one sentence?*
3. `git checkout -b experiment`, change the file, commit. Switch back to the first branch
   (`git branch` shows its name), `cat` the file. → *Where did the change go? Why is it
   not gone?*
4. `git merge experiment`. → *What did merging actually do to that branch?*
5. Make conflicting edits on both branches and merge them. → *Read the conflict markers
   out loud. Which side is which?*

Point 5 is the one that matters. A student who has deliberately caused and resolved one
conflict in a sandbox is not afraid of the real one the night before a deadline.

### Command line

If the terminal itself is the unfamiliar part, these are the ones actually used daily:
`ls` (and `la` for hidden files), `cd`, `pwd`, `mkdir`, `cat`, `df -H` for disk space,
`du -sh *` for the size of things here. Teach them when they come up in real work rather
than as a list to memorise.

---

## 4. Troubleshooting index

| Symptom | Almost always |
|---|---|
| "You do not have access" on labs.azure.com | Signed in as a different Microsoft account than the invited one |
| SSH still asks for a password after M5 | The pasted key wrapped onto multiple lines, or wrong file, or `chmod 700 ~/.ssh` / `600 authorized_keys` |
| `Permission denied (publickey)` | The key SSH offers is not the one in `authorized_keys` — set `IdentityFile` in `~/.ssh/config` |
| `Connection refused` / timeout | The VM is stopped. Start it at labs.azure.com and wait until it runs |
| `No such file or directory` on a key | Wrong name or path — `ls ~/.ssh` shows the truth |
| `.venv` not offered as a kernel | VS Code opened in the parent folder, or `uv sync` never ran; reload the window |
| Notebook imports fail | Wrong kernel selected — check the top-right of the notebook |
| Nothing appears while typing a password | Working as designed. Keep typing |
| A run is gone after reconnecting | It was started outside tmux, or the VM was stopped; `tmux ls` shows what survived |
| `tmux ls` says `no server running` | No session exists: it ended, or the VM restarted since |
| mlflow page does not open on the laptop | The port forward on 5050 is closed, or `mlflow ui` is not running on the VM. A `403` from `Server: AirTunes` means port 5000 on a Mac: use 5050 |
| `nvidia-smi: command not found` | Normal on a CPU-only VM; the course does not need a GPU |
| `No space left on device` during `uv sync` | The lock file pulls the CUDA builds of torch on Linux, which are large. Tell the teacher; do not delete things on the VM to make room |
| `Address already in use` when starting `mlflow ui` | mlflow is already running on the VM: `tmux attach -t mlflow` |
| Ran out of hours | Ask the teacher; it can be extended. Do it well before a deadline |

If something falls outside this table, do not guess in a loop. Read the actual error,
say what you think it means, and if two attempts do not fix it, tell the student to mail
the teacher — the teacher's public key is in `authorized_keys` for exactly this reason.
