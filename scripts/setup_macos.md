# Setup macOS

- [Setup macOS](#setup-macos)
  - [ZSHRC](#zshrc)
    - [NPM](#npm)
    - [Tree](#tree)
    - [P10K](#p10k)
  - [Install Homebrew](#install-homebrew)
    - [Homebrew PATH](#homebrew-path)
  - [Install Java](#install-java)
  - [Install Chrome](#install-chrome)
  - [Install Alt Tab](#install-alt-tab)
  - [VSCode](#vscode)
  - [Docker](#docker)
  - [Node](#node)
  - [Tex Live](#tex-live)
  - [Set up Git](#set-up-git)
    - [Configure Local Git](#configure-local-git)
    - [Cloning](#cloning)
  - [Setting up SSH for GitHub](#setting-up-ssh-for-github)
    - [Checking for existing SSH keys](#checking-for-existing-ssh-keys)
    - [Generating a new SSH key pair](#generating-a-new-ssh-key-pair)
    - [Starting the SSH agent and adding the SSH key](#starting-the-ssh-agent-and-adding-the-ssh-key)
    - [Adding the SSH key to your GitHub account](#adding-the-ssh-key-to-your-github-account)
    - [Testing your SSH connection](#testing-your-ssh-connection)
    - [Configuring Git](#configuring-git)

## ZSHRC

```bash
touch ~/.zshrc
source ~/.zshrc
```

where `source` is a command that runs the commands in the file and will act as
re-running the terminal.

### NPM

```bash
brew install node
```

### Tree

```bash
brew install tree
```

### P10K

Follow closely
https://github.com/romkatv/powerlevel10k#meslo-nerd-font-patched-for-powerlevel10k
here for downloading recommended fonts. Most importantly you need to have
MesloLGS NF Regular font installed.

Then follow configuration with unicode + more icons choice!

## Install Homebrew

Homebrew is a package manager for macOS that simplifies the installation of
various software packages. Run the following command in your terminal:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Homebrew PATH

macOS will prompt you to add Homebrew to your path.

## Install Java

```bash
brew install openjdk@17
```

but not sure if you can do `brew install java` directly.

## Install Chrome

```bash
# Install chrome
brew install google-chrome --cask
```

## Install Alt Tab

```bash
# Install Alt Tab
brew install --cask alt-tab
```

## VSCode

```bash
brew install --cask visual-studio-code
```

Note in settings better put

```bash
"terminal.integrated.env.osx": {
    "PATH": "" }
```

if you are macOS user to make VSCode use the same PATH as the terminal.

See:
https://stackoverflow.com/questions/54582361/vscode-terminal-shows-incorrect-python-version-and-path-launching-terminal-from

Then sign in with your github account to sync settings.

## Docker

```bash
brew install docker --cask
```

## Node

```bash
# Install node and npm
brew install node
```

## Tex Live

```bash
brew install texlive
```

## Set up Git

```bash
brew install git
```

but usually macOS comes with git already installed.

### Configure Local Git

```bash
git config --global user.name "Your Name"
git config --global user.email "youremail@example.com"
```

### Cloning

```bash
git clone git@github.com:gao-hongnan/gaohn-galaxy.git
```

as an example. This is for SSH method.

## Setting up SSH for GitHub

### Checking for existing SSH keys

SSH keys come in pairs, one private and one public. They are used for secure communication with GitHub's servers.

1. Start by opening Terminal.
2. Enter the command `ls -al ~/.ssh` to see if any SSH keys already exist.
3. This command lists the contents of your SSH directory. If files named `id_rsa` (private key) and `id_rsa.pub` (public key) appear, you already have a key pair and can skip to the section about adding the key to the SSH agent.

### Generating a new SSH key pair

If you don't see `id_rsa` and `id_rsa.pub` in your SSH directory:

1. Run this command in Terminal, replacing `<youremail@example.com>` with the email address you use for your GitHub account:

    ```bash
    ssh-keygen -t rsa -b 4096 -C <youremail@example.com>
    ```

2. This command generates a new SSH key pair. It uses `-t rsa` to specify the type of key, `-b 4096` to specify the key length (4096 bits), and `-C <youremail@example.com>` to label the key with your email address.

3. After running this command, you'll be asked where to save the key. Press Enter to accept the default location (`~/.ssh/id_rsa`).

4. You'll also be asked to enter a passphrase for extra security. This is optional, but it can help protect your key if it ever falls into the wrong hands.

### Starting the SSH agent and adding the SSH key

The SSH agent is a program that holds your private keys, so you don't need to re-enter your passphrase every time you use a key.

1. Enter the command `eval "$(ssh-agent -s)"` to start the SSH agent in the background.

2. To ensure the SSH agent can find your key, you'll need to add it:

3. Start by making sure your .ssh directory has the right permissions. Enter `chmod 700 ~/.ssh`.

4. If you're using macOS 10.12.2 or newer, you'll also need to add a few lines to your ~/.ssh/config file to tell the SSH agent to automatically load your keys. You can do this with these commands:

    ```bash
    touch ~/.ssh/config
    open ~/.ssh/config
    ```

5. This creates the config file if it doesn't exist, and opens it in your default text editor. Add the following lines:

    ```bash
    Host *
      AddKeysToAgent yes
      UseKeychain yes
      IdentityFile /Users/<USERNAME>/.ssh/id_rsa
    ```

6. Replace `<USERNAME>` with your actual username, then save and close the file.

7. Finally, run `ssh-add -K ~/.ssh/id_rsa` to add your SSH key to the SSH agent.

    Note that you might see a warning message:

    ```bash
    WARNING: The -K and -A flags are deprecated and have been replaced
            by the --apple-use-keychain and --apple-load-keychain
            flags, respectively.  To suppress this warning, set the
            environment variable APPLE_SSH_ADD_BEHAVIOR as described in
            the ssh-add(1) manual page.
    ```

### Adding the SSH key to your GitHub account

1. To add the key to GitHub, you'll first need to copy it. Run `pbcopy < ~/.ssh/id_rsa.pub` to copy the public key to your clipboard.

2. Then, navigate to GitHub and click on your profile photo, then click "Settings".

3. In the sidebar, click "SSH and GPG keys", then "New SSH key".

4. Paste your key into the "Key" field and give it a descriptive title.

5. Click "Add SSH key" to finish.

### Testing your SSH connection

1. To verify that everything is working, run `ssh -T git@github.com` in Terminal.

2. The first time you do this, you'll see a warning message. Just type `yes` and press Enter.

3. If everything is set up correctly, you'll see this message: "Hi username! You've successfully authenticated, but GitHub does not provide shell access."

### Configuring Git

1. Finally, it's a good idea to configure Git with your name and email address. This information will be included in each commit you make.

2. Replace `Your Name` with your actual name and `<youremail@example.com>` with your actual email address associated with your GitHub account:

    ```bash
    git config --global user.name "Your Name"
    git config --global user.email "<youremail@example.com>"
    ```

    Note that the `user.name` can be arbitrary, but the `user.email` must be the
    same as the one you used to register your GitHub account, or else commits
    you make will not be associated with your account.
