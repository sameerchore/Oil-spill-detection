import json
import base64
from pathlib import Path

train_code_bytes = Path("scripts/kaggle_bundle/train.py").read_bytes()
b64_train = base64.b64encode(train_code_bytes).decode("ascii")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# TideTrace Remote GPU Training Pipeline (SIH 26143 / NTRO)\n",
            "## Universal Pascal / Turing GPU Accelerator with Hugging Face Checkpointing\n",
            "\n",
            "**Task**: 3-Class Semantic Segmentation (0: Sea, 1: Look-alike, 2: Mineral Oil Spill)\n",
            "**Hardware Target**: Nvidia Tesla P100 (sm_60) or Nvidia Tesla T4 (sm_75)\n",
            "**Checkpoint Target**: `oil_unet_best.pt` uploaded to Hugging Face Hub `SidMI/TideTrace-weights`"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# CELL 1: Hardware Detection, PyTorch Driver Setup & Script Staging\n",
            "# =============================================================================\n",
            "import sys\n",
            "import os\n",
            "import base64\n",
            "import subprocess\n",
            "from pathlib import Path\n",
            "import torch\n",
            "\n",
            "print(f'Container Python: {sys.version}')\n",
            "print(f'Container PyTorch: {torch.__version__}')\n",
            "\n",
            "if torch.cuda.is_available():\n",
            "    name = torch.cuda.get_device_name(0)\n",
            "    major, minor = torch.cuda.get_device_capability(0)\n",
            "    print(f'Active Accelerator: {name} (Compute Capability sm_{major}{minor})')\n",
            "    \n",
            "    if major < 7:\n",
            "        print(f'Detected Pascal P100 (sm_{major}{minor}). Configuring PyTorch cu118 with native sm_60 support...')\n",
            "        subprocess.run([\n",
            "            sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir',\n",
            "            'torch==2.3.1+cu118', 'torchvision==0.18.1+cu118',\n",
            "            '--index-url', 'https://download.pytorch.org/whl/cu118'\n",
            "        ], check=True)\n",
            "        print('PyTorch cu118 successfully installed for Pascal GPU.')\n",
            "    else:\n",
            "        print(f'Detected modern GPU (sm_{major}{minor}). Native PyTorch {torch.__version__} is fully accelerated!')\n",
            "else:\n",
            "    print('Warning: No GPU detected. Execution will proceed on CPU.')\n",
            "\n",
            "# Write standalone train.py into execution directory\n",
            "train_script_path = Path('/kaggle/working/train.py') if Path('/kaggle/working').exists() else Path('./train.py')\n",
            f"b64_data = '{b64_train}'\n",
            "train_script_path.write_bytes(base64.b64decode(b64_data))\n",
            "print(f'Staged {train_script_path} ({train_script_path.stat().st_size} bytes).')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =============================================================================\n",
            "# CELL 2: Launch High-Performance GPU Training\n",
            "# =============================================================================\n",
            "import subprocess\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "train_script_path = Path('/kaggle/working/train.py') if Path('/kaggle/working').exists() else Path('./train.py')\n",
            "print(f'Executing: {sys.executable} {train_script_path}...')\n",
            "res = subprocess.run([sys.executable, str(train_script_path)])\n",
            "if res.returncode != 0:\n",
            "    raise RuntimeError(f'Training process failed with exit code {res.returncode}')\n",
            "\n",
            "print('=' * 70)\n",
            "print('TRAINING & HUGGING FACE SYNCHRONIZATION FINISHED SUCCESSFULLY!')\n",
            "print('=' * 70)\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "gpu",
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

Path("scripts/kaggle_bundle/train_kaggle.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")
Path("scripts/train_kaggle.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Successfully generated dual-driver train_kaggle.ipynb with base64 staging!")
