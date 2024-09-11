# The many faces of Path

There are few topics that seem to cause so much confusion as paths for beginners. This is probably because there are a lot of different situations where we refer to things as "paths", and they all have their own quirks.

# The PATH environment variable

In your terminal, there is an environment variable called PATH. This is a list of directories that the terminal will search through when you type a command. For example, if you type `python` in your terminal, it will search through the directories in PATH to find the `python` executable. If the directory of the `python` executable added to the PATH variable, you van type `python` in any directory, and it will execute. When it is not added to PATH, you will have to type the full path to the executable to run it.

When you start up your terminal, there is a variety of locations your terminal will search for PATHs. For example, using bash, you will have files like `~/.bash_profile` and `~/.bashrc`. If you are using a shell like zsh, you will have `~/.zshrc`. These files are executed when you start up your terminal and they are used to add directories to your PATH.

For example, I am using zhs and my .zshrc file has an entry like this:

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
```

This means that when I start up my terminal, it will add the directory `~/.pyenv/bin` to my PATH. This is where the `pyenv` executable is located. If I would not have this line in my .zshrc file, I would not be able to use `pyenv` in my terminal (it really doesnt matter what pyenv is, it is just an example; but every command you type, for example ls or cd or ssh, it is located somewhere in your $PATH).
You can check the location of commands with `which ssh` for ssh, or `which cd` for cd, etc.

You can see the value of PATH by typing `echo $PATH` in your terminal. If you do that, you will see something like this:

```bash
/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/.pyenv/bin
```

These are all different locations where your terminal looks for executable commands. The last entry in the list is `usr/,pyenv/bin`. This is where the terminal will find the `pyenv` executable when you type `pyenv`.

# PYTHONPATH

There is also something that is called  `PYTHONPATH`. Instead of being a list of global directories where to find executables, `PYTHONPATH` is a Python specific list of directories in wich Python will search for Python modules and packages. When you install `numpy` and use it in your code, how does Python know when you run your code where to find the correct version of `numpy`? It will look in all the directories in `PYTHONPATH`.

VScode will help you by adding the directory of your current project to `PYTHONPATH`. It will also detect folders like .venv that typically contain virtual environments. But if you are not using VScode, you will have to think about how these locations are added. More on virtual environments (.venv), dependencies and package management in the next section.

# Paths as in Filepaths

Then, there is another type of path, which is a filepath. This is a path to a file on your computer. For example, if you have a file called `my_file.txt` in your home directory, the filepath will be `/Users/yourusername/my_file.txt`. If you have a file called `my_file.txt` in a directory called `my_directory` in your home directory, the filepath will be `/Users/yourusername/my_directory/my_file.txt`.

The problem with using paths like this, is that /Users/yourusername/ is different on every computer. So if you want to share your code with someone else, you would have to change the location. That is why it is better to either use the HOME environment variable, which every shell has in its environment. That is why I write `$HOME/.pyenv/bin` (for example) in my .zshrc file, instead of `/Users/yourusername/.pyenv/bin`: this way I dont need to modify my zshrc file if I copy it to another machine.
