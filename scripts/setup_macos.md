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
    "PATH": ""
    }
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

## Setting up SSH

Secure Shell (SSH) is a cryptographic network protocol that allows you to
securely access network services over an unsecured network.

In the context of GitHub, SSH is used to securely authenticate and communicate
with GitHub servers without needing to provide your username and password every
time. Here's how to set up SSH for GitHub on macOS.

See
[here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
also for guide.

### Check for existing SSH keys

Open Terminal, and check if you already have an SSH key pair:

```bash
ls -al ~/.ssh
```

If you see files like `id_rsa` and `id_rsa.pub`, you already have an SSH key
pair. If not, continue to the next step.

### Generate a new SSH key

In the terminal, run the following command, replacing "youremail@example.com"
with your email address associated with your GitHub account:

```bash
ssh-keygen -t rsa -b 4096 -C <youremail@example.com>
```

Press Enter to accept the default file location, or specify a different location
if desired.

Now if you run `ls -al ~/.ssh`, you should see a new file called `id_rsa` and a
new file called `id_rsa.pub`.

### Enter a passphrase (optional)

You will be prompted to enter a passphrase to secure your SSH key. This is
optional, but recommended for added security. Enter a passphrase or press Enter
to skip.

### Start the SSH agent

Run the following command to start the SSH agent in the background:

```bash
eval "$(ssh-agent -s)"
```

### Add SSH key to the SSH agent

First, ensure your .ssh directory has the correct permissions:

```bash
chmod 700 ~/.ssh
```

If you are using macOS 10.12.2 or newer, modify your ~/.ssh/config file to
automatically load keys into the SSH agent:

```bash
touch ~/.ssh/config
```

```bash
vim ~/.ssh/config
```

Add the following lines to the file, replacing /Users/yourusername/.ssh/id_rsa
with the path to your private key if you specified a different location in step
2:

```bash
Host *
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile /Users/yourusername/.ssh/id_rsa
```

replace `yourusername` with your username.

Save and close the file. Now, add the SSH key to the SSH agent:

```bash
ssh-add -K ~/.ssh/id_rsa
```

### Add the SSH key to your GitHub account

```bash
pbcopy < ~/.ssh/id_rsa.pub
```

this will have copied the SSH key to your clipboard.

Go to GitHub, and navigate to your account's settings page. Click on "SSH and
GPG keys" in the sidebar, then click the "New SSH key" button.

Paste your SSH public key into the "Key" field, give it a descriptive title, and
click "Add SSH key".

### Test SSH connection

In the terminal, run the following command:

```bash
ssh -T git@github.com
```

First time you will see some warning, just type `yes` and press enter.

Then you will see something like this:

```bash
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

means you have successfully set up SSH for GitHub.

### Configure Local Git

```bash
git config --global user.name "Your Name"
git config --global user.email "youremail@example.com"
```
