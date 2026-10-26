# Install and run predictions

Install the software, predict 60 example sites and verify the output. The repository includes the fitted model, example input and expected predictions.

**Choose one starting point:** [RONIN](#ronin) · [Ubuntu or WSL](#ubuntu) · [Windows](#windows)

For an existing installation, [reopen the environment](#reopen-the-environment). See the [project overview](../README.md) for the model and scientific results.

## Hardware

These resources apply to the machine running Python: the Ubuntu instance when using RONIN, or your computer for a local installation.

| Resource | Recommended for the example |
|---|---|
| Processor | 8 CPU cores or vCPUs, Intel or AMD x86-64 |
| Memory | 16 GB RAM |
| Storage | 20 GB free space for the repository, environment and installation caches |
| Software | Ubuntu 22.04/24.04 or Windows 10/11; Python 3.11 |
| GPU | Not required |

For RONIN, use an existing instance with these resources or select **8 vCPUs, 16 GiB RAM and a 64 GB root disk**. An example is AWS [`c7i.2xlarge`](https://aws.amazon.com/ec2/instance-types/c7i/), where available.

For full SG-NEx prediction, allow **32 GB RAM and at least 80 GB working storage**. See the [SG-NEx prediction instructions](<../task 2/README.md#reproduce-predictions-from-the-original-inputs>).

## RONIN

[RONIN LINK](https://ronin.cloud/ronin-link.html) connects Windows, macOS or Linux to an Ubuntu instance, where the software runs and output files are stored.

1. Sign in to your RONIN website and open your project's machine list. Choose an existing **x86-64 Ubuntu** instance, or create one using the resources above. Start it and wait for **RUNNING** and completed status checks.
2. Open [RONIN LINK](https://ronin.cloud/ronin-link.html) on your computer. In **LINK TO YOUR MACHINE**, enter the instance's **MACHINE ADDRESS** without `https://`, its **USER NAME** (normally `ubuntu` for an Ubuntu image), and select its matching private `.pem` key.
3. Click **LINK TO MACHINE**. When its card shows **AVAILABLE**, choose **CONNECT TO MACHINE → TERMINAL**. If the instance is already linked, open its terminal directly.
4. An Ubuntu welcome message and a prompt resembling `ubuntu@ip-…:~$` indicate a successful connection. Continue with [Ubuntu setup](#ubuntu) in that terminal.

See [RONIN's connection guide](https://blog.ronin.cloud/ronin-link/) for connection help.

## Ubuntu

Run these commands in an **x86-64 Ubuntu terminal**, either locally, through RONIN LINK or in Windows Subsystem for Linux (WSL).

Copy each command block into the terminal and press Enter. Wait for it to finish before continuing. If a command fails, see [troubleshooting](#troubleshooting).

### 1. Download the project

Download and extract the repository (approximately 619 MB) into a folder under your home directory:

```sh
mkdir -p "$HOME/context-aware-trees-work"
cd "$HOME/context-aware-trees-work"
curl -L --fail -o project.zip https://github.com/dsa4262-yellow-2026/m6a-context-aware-trees/archive/refs/heads/main.zip
python3 -m zipfile -e project.zip .
cd m6a-context-aware-trees-main
ls
```

The listing should include `README.md`, `task 1 context aware trees` and `task 2`. This folder is the **repository root**.

If the project is already extracted, open a terminal in its repository root and continue below.

### 2. Make Conda available

Conda creates a separate Python environment for this project. If `conda --version` already prints a version in this terminal, continue to [install the packages](#install-the-packages).

Otherwise, install [Miniforge](https://github.com/conda-forge/miniforge#install), which supplies Conda:

```sh
curl -L --fail -o /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

If Miniforge is already installed in `$HOME/miniforge3`, run only the final `source` command. Continue to [install the packages](#install-the-packages).

## Windows

These steps run Python **locally on an Intel or AMD Windows PC**. If you are connecting to RONIN instead, use the [RONIN steps](#ronin).

1. Install [Miniforge for Windows](https://github.com/conda-forge/miniforge#install), keeping its Start menu shortcut. Follow its installation guidance, including choosing an installation path without spaces. If you already use Conda, open your existing Miniforge or Anaconda Prompt.
2. Download the [project ZIP](https://github.com/dsa4262-yellow-2026/m6a-context-aware-trees/archive/refs/heads/main.zip). In File Explorer, right-click it and choose **Extract All**.
3. Open the extracted `m6a-context-aware-trees-main` folder containing `README.md`, `task 1 context aware trees` and `task 2`. Click File Explorer's address bar and copy the full folder path.
4. Open **Miniforge Prompt** from the Windows Start menu. Type `cd /d `, then the copied folder path inside double quotes, and press Enter. For example, if you extracted the project to `C:\Projects`:

```bat
cd /d "C:\Projects\m6a-context-aware-trees-main"
dir
```

Replace the example path with your actual folder path. The listing should include `README.md` and both task folders. Continue below in the **same Miniforge Prompt**.

## Install the packages

Run these commands from the **repository root** in the Ubuntu terminal or Windows Miniforge Prompt:

```sh
conda create -n context-aware-trees --override-channels -c conda-forge python=3.11 pip -y
conda activate context-aware-trees
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "task 1 context aware trees/requirements.txt"
python -m pip check
```

The prompt should now include `(context-aware-trees)`. The final command should print:

```text
No broken requirements found.
```

## Run the example

In the same terminal, enter the model folder and run prediction followed by validation:

```sh
cd "task 1 context aware trees"
python predict.py --data example/input.json --output outputs/example_predictions.csv
python validate_predictions.py --data example/input.json --predictions outputs/example_predictions.csv --expected example/expected_predictions.csv
```

Expected output:

```text
Wrote 60 scores to example_predictions.csv.
PASS: 60 sites; exact columns, input coordinates/order, finite scores in [0,1]; expected scores match.
```

`PASS` confirms that the output rows, columns and scores match the bundled example within numerical tolerance.

### Find and inspect the CSV

Your new file is `outputs/example_predictions.csv` inside `task 1 context aware trees`. It has 60 data rows and a header:

```csv
transcript_id,transcript_position,score
ENST00000000412,496,0.01605253872787089
ENST00000000412,2440,0.5593107649860982
ENST00000007264,108,0.024748834437336574
```

Each row identifies one transcript position. `score` is a continuous value between 0 and 1 indicating predicted m6A site status; it is not a modification percentage.

On **Ubuntu or RONIN**, view the first rows and count the lines:

```sh
head -n 4 outputs/example_predictions.csv
wc -l outputs/example_predictions.csv
```

The count should be **61**: 60 sites plus the header. On RONIN, the file is on the cloud instance.

On **Windows**, open the file in File Explorer, or display it from the same Miniforge Prompt:

```bat
type outputs\example_predictions.csv
```

## Use your own data
From `task 1 context aware trees/`, provide the processed JSON or JSON.gz for a complete sample:

```sh
python predict.py --data "/path/to/data.json.gz" --output outputs/new_sample.csv
python validate_predictions.py --data "/path/to/data.json.gz" --predictions outputs/new_sample.csv
```

Replace the quoted input path with your own. The [input-format description](<../task 1 context aware trees/README.md#input-format>) defines the required signal measurements. For larger datasets, use the [SG-NEx prediction instructions](<../task 2/README.md#reproduce-predictions-from-the-original-inputs>).

## Reopen the environment

Open a terminal, activate the environment and enter the model folder.

**Ubuntu or RONIN**, using the locations in this guide:

```sh
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate context-aware-trees
cd "$HOME/context-aware-trees-work/m6a-context-aware-trees-main/task 1 context aware trees"
```

If you use another Conda installation, open its usual terminal and activate the environment. If you downloaded elsewhere, use your actual project path.

**Windows:** open Miniforge Prompt, then run the following with your own extracted project path:

```bat
conda activate context-aware-trees
cd /d "C:\Projects\m6a-context-aware-trees-main\task 1 context aware trees"
```

Continue with the two Python commands under [run the example](#run-the-example).

### Finish a RONIN session

For a machine used only for your session, return to its RONIN website card, choose **Stop** and wait for **STOPPED**. Closing the terminal or RONIN LINK does not stop the instance. A retained disk can continue to incur storage charges. Coordinate with other users before stopping a shared instance.

## Troubleshooting

| What you see | What to do |
|---|---|
| RONIN LINK does not show **AVAILABLE** | Confirm the instance is **RUNNING** and use its exact address, user name and matching private key. |
| `conda: command not found` or `conda` is not recognized | Ubuntu: run the `source` line under [reopen the environment](#reopen-the-environment). Windows: open **Miniforge Prompt** from Start. |
| Conda says the environment already exists | Run `conda activate context-aware-trees`. If installation was interrupted, continue with the two `python -m pip install` commands and `python -m pip check`. |
| Cannot find `requirements.txt` | The package-install block starts in the **repository root**, where both task folders are visible. Use `ls` on Ubuntu or `dir` on Windows to check. |
| Cannot open `predict.py` or `example/input.json` | Enter `task 1 context aware trees/` using the full path under [reopen the environment](#reopen-the-environment). Keep quotes around paths containing spaces. |
| `No module named …` | Activate the environment. From the repository root, repeat the package-install commands, starting with the PyTorch installation. |
| Download times out or ends early | Check the connection and rerun the failed download or installation command. Wait for it to finish before continuing. |
| No matching package distribution | Check that the environment uses **Python 3.11** and the machine is **x86-64 Ubuntu or Windows**. On a Mac or ARM computer, use the RONIN route. |
| `No space left on device` | Check free space on the machine running Python. Allow 20 GB for this setup; a full SG-NEx run needs more. |
| Validation prints `FAIL` | Read the reason printed after `FAIL`. Re-run prediction with the bundled model and `example/input.json`, then validate the newly generated `outputs/example_predictions.csv` against `example/expected_predictions.csv`. |

If a problem persists, [open an issue](https://github.com/dsa4262-yellow-2026/m6a-context-aware-trees/issues) with your operating system, Python version, command and complete error message.
