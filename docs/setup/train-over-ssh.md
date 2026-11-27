# Training on a VM over SSH

The models of this course train on a CPU, but not on every CPU. A laptop built for battery
life (an Intel or AMD chip whose name ends in **U**, a thin Windows laptop, an old
MacBook) can be twenty or thirty times slower than a recent one, and slows down further
once it is warm. On such a laptop a search that should run during dinner runs all
evening, and the homework cannot be planned. A VM solves the planning: it keeps running
while your laptop is closed, so you start a run, close the lid, and read the result
later.

This page is the routine once you have a VM. Setting one up (the SSH key, the host alias
`ml-vm`, VS Code Remote-SSH, the clone and `uv sync`, tmux, the Optuna dashboard) is the
lab guide, [`lab-setup.md`](lab-setup.md): give its raw URL to your assistant and let it
walk you through.

## Your 150 hours

Every student has **150 VM hours**. The clock runs for every hour the VM is on, whether it
trains or sits idle: a VM left on overnight with nothing running costs as many hours as
one that trains. So the VM is on while a run goes, and off as soon as the runs are done.

**Only Stop on the website stops the clock.** Shutting the VM down from inside
(`sudo shutdown`, or closing the connection) does not: Azure counts a VM that was shut
down from its own operating system as still running, and it keeps using your hours. Your
assistant cannot stop it for you either; it has no access to your lab account. What it
can do is tell you the moment a run has finished: ask it to check `tmux ls` on the VM
when you come back, and to remind you to press Stop.

## Do you need one?

Let the course measure it. Run Part 1 of a lesson's exercise notebook on your laptop
until it prints how long the rest takes here: lesson 2 prints it after E2
(`04_exercise.ipynb`), lesson 3 after E1 (`05_exercise.ipynb`). When it turns `QUICK` on,
your laptop is slow for this course. `QUICK` gets you through Part 1, but your own study
in Part 2 trains just as slowly: ask your teacher for a VM.

A VM is not automatically faster than your laptop: four cores of a server can be slower
than a recent laptop and faster than a thin one. Run the same cell on the VM and compare
the time it prints. Even at the same speed, the VM is the one that keeps running.

## The routine

Every command below runs **on the VM**, in a terminal on the laptop connected with
`ssh ml-vm`. Use a plain terminal for long runs, not the terminal of VS Code: a tmux
session is what keeps a run alive when the connection drops.

1. **Start the VM** at <https://labs.azure.com/virtualmachines> and connect:

   ```bash
   ssh ml-vm
   ```

2. **Get the latest course material** (with your own work committed first, see M11 of
   the lab guide):

   ```bash
   cd ~/MADS-MachineLearning-course && git switch master && git pull && uv sync
   ```

3. **Open a tmux session** for the run. `-A` reattaches when it exists already:

   ```bash
   tmux new -A -s train
   ```

4. **Run Part 1 of the exercise notebook** without a browser or an editor: Jupyter runs
   every cell and writes the notebook, with its outputs, under a new name, so the
   course's own file stays untouched and `git pull` never conflicts with it. Lesson 2:

   ```bash
   uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
     --output 04_exercise_ran notebooks/2_kernels_state_and_gates/04_exercise.ipynb
   ```

   Lesson 3 is the same with `notebooks/3_scaling_and_search/05_exercise.ipynb` and
   `--output 05_exercise_ran`. The runs go to the Optuna store on the VM, as they would on
   your laptop. Open `04_exercise_ran.ipynb` later in VS Code (Remote-SSH) to read the
   results and answer the questions.

5. **Run your study** in the same session, once Part 1 has finished:

   ```bash
   make study LESSON=2
   ```

   It first asks at the terminal for whatever `STUDY`, `IDEA` or `EXPECTS` in
   `study.py` still lacks, and whether to start. Answer those, see the budget it prints,
   and only then detach.

6. **Detach**: `ctrl+b`, then `d`. The run goes on. Close the connection and the laptop.

7. **Come back later**: `ssh ml-vm`, then `tmux new -A -s train`, and read where it is.
   The search stops when `MINUTES` are up; a study that ran out of time says how many
   trials are left, and the next `make study` runs those.

8. **Read the runs** in the Optuna dashboard, as M10 of the lab guide sets up
   (`make optuna-dashboard` in its own tmux session, the port forwarded to your laptop).

9. **Hand in from the VM**: the runs are in the store on the VM, so `make share` runs
   there.

   ```bash
   make share LESSON=2
   ```

10. **Stop the VM** at <https://labs.azure.com/virtualmachines> as soon as no run is going
    (`tmux ls` lists none, or only the dashboard), and check on the tile that it says
    stopped. A VM that is left on keeps using your 150 hours, also when it does nothing.
