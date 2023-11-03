# 🌳 Scaled Trunk Based Git Development 🌳

## Motivation

Trunk Based Git Development aims to simplify the development workflow by promoting a single main branch approach, and feature branches for new features. It encourages continuous integration and collaboration among team members, leading to faster feedback loops and efficient development cycles.

> **NOTE**: there is a key difference between `Trunk Based Git Development` and `Scaled Trunk Based Git Development`. The difference is that in the latter, there are no direct commits into the main branch. Instead, all changes are made in feature branches and merged into the main branch.

## Gitflow and Deployment

Gitflow is a much more complex branching model that is often used in large-scale projects. It is designed to facilitate parallel development and multiple release cycles. However, it can be difficult to maintain and manage, especially for smaller projects. In addition, Gitflow is not suitable for continuous deployment, as it requires a lot of manual work to merge and deploy changes.

In our context (data science courses or small-scale projects), Trunk Based Git Development provides a lightweight alternative that can be equally effective.

## Main Rules of Scaled Trunk Based

1. **Single Main Branch:** Development takes place primarily on a single main branch.
1. **Feature Branches:** New features are created in feature branches, with the naming convention `feature/newfeatuename`.
1. **Continuous Integration:** Frequent integration of code changes into the main branch to detect and resolve conflicts early.
1. **Small, Frequent Commits:** Encourages developers to make small, focused commits to facilitate better collaboration and easier code reviews. Rule of thumb: commit every 30 minutes. Describe your changes such that others understand what is going on (so, DONT do `git commit -m "some stuff"`).

## Example in Bash

Here's an example of using Trunk Based Git Development in a bash terminal:

```bash
# Create and switch to a new branch for a feature
git checkout -b feature/my-feature

# Make changes to the code

# Commit the changes
git commit -m "Implement feature XYZ"

# Push the changes to the remote repository
git push origin feature/my-feature

# send out pull request to others in your team for review

# Once the feature is ready, merge it into the trunk branch
git checkout main
git merge feature/my-feature

# Push the changes to the remote trunk branch
git push origin main
```
