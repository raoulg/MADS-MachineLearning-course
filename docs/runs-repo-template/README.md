# MADS Machine Learning: class runs

The runs everyone in the class submitted, one JSON file per mlflow run:

    lesson<N>/<github-login>/<run_id>.json

Submit through a pull request from your own fork, made by `scripts/submit_runs.py` from
the course repository: the files are generated, not hand-edited. The check `validate`
reads every file a pull request changes. It merges a pull request whose files are valid
runs in its author's own directory, and comments on one that is not: a file in someone
else's directory, a run another student already submitted, a file that does not fit the
format, or any change outside `lesson<N>/`.

How to submit, including the one-time fork: `docs/submit-runs.md` in the course
repository. The format: `scripts/run_submission.py` there.
