Microsoft provides some [documentation](https://code.visualstudio.com/docs/devcontainers/containers) about using
docker with VScode. Below are the steps I followed for setup. Note that you **need** to add poetry (step 6) to make things work.

0. Install VSCode and Docker Desktop
1. Install the Dev Containers extension from Microsoft in VSCode
2. Open the ML22 directory with VScode
3. Pick the "Dev Conainers: Reopen in Container" option
4. Select "Python 3 devcontainers"
5. Pick 3.10
6. Select "Poetry (via pipx)"
7. Select "Common Utilities"
8. Pick "Keep Defaults". Wait for ~5 minutes

9. In the terminal in VScode, you should be in "/workspaces/ML22$". Run the "poetry install" command. Wait for ~10 minutes.
10. Reload VScode and click "Select Kernel". Select the "deep-learning" environment.


