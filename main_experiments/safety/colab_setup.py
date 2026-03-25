"""
Colab Setup Script for PGR Safety Experiments.

Run this in a Colab cell to set everything up:
    !python safety/colab_setup.py

Or copy individual sections into Colab cells.
"""

import subprocess
import sys
import os


def run(cmd, check=True):
    print(f'>>> {cmd}')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-500:])  # Last 500 chars
    if result.returncode != 0 and check:
        print(f'STDERR: {result.stderr[-500:]}')
        if check:
            raise RuntimeError(f'Command failed: {cmd}')
    return result


def setup():
    """Full setup sequence for Colab with A100."""

    # 1. Check GPU
    print("=" * 60)
    print("Step 1: Checking GPU")
    print("=" * 60)
    result = run('nvidia-smi', check=False)
    if result.returncode != 0:
        print("WARNING: No GPU detected! Results will be very slow.")
        print("Go to Runtime > Change runtime type > GPU (A100 preferred)")
    else:
        print("GPU available!")

    # 2. Install system dependencies
    print("\n" + "=" * 60)
    print("Step 2: Installing system dependencies")
    print("=" * 60)
    run('apt-get -qq update && apt-get -qq install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf', check=False)

    # 3. Install Python packages
    print("\n" + "=" * 60)
    print("Step 3: Installing Python packages")
    print("=" * 60)

    # Core packages (skip mujoco-py, use newer mujoco)
    packages = [
        'torch',  # Usually pre-installed on Colab
        'numpy',
        'accelerate',
        'einops',
        'ema-pytorch',
        'tqdm',
        'gin-config',
        'mujoco>=2.3.0',
        'dm-control',
        'gymnasium[mujoco]',
        'h5py',
        'sortedcontainers',
        'pyrallis',
        'matplotlib',
    ]
    run(f'{sys.executable} -m pip install -q ' + ' '.join(packages))

    # Install gym 0.23 (required by PGR)
    run(f'{sys.executable} -m pip install -q "gym==0.23.0"')

    # Install dmcgym from the specific commit
    run(f'{sys.executable} -m pip install -q git+https://github.com/conglu1997/dmcgym.git@812905790dd87a448c9544a0beccb8b05ea2a850')

    # Install dm-env and dm-tree
    run(f'{sys.executable} -m pip install -q dm-env dm-tree')

    # 4. Setup REDQ as editable install
    print("\n" + "=" * 60)
    print("Step 4: Installing REDQ submodule")
    print("=" * 60)
    if os.path.exists('synther/REDQ'):
        run(f'{sys.executable} -m pip install -e synther/REDQ')
    else:
        print("WARNING: REDQ submodule not found. Run: git submodule update --init --recursive")

    # 5. Skip d4rl (not needed for online training, causes dependency conflicts)
    print("\n" + "=" * 60)
    print("Step 5: Patching imports (skip d4rl)")
    print("=" * 60)
    # d4rl is only imported in synther/diffusion/utils.py but not used for online training
    utils_path = 'synther/diffusion/utils.py'
    if os.path.exists(utils_path):
        with open(utils_path, 'r') as f:
            content = f.read()
        if 'import d4rl' in content:
            content = content.replace('import d4rl', '# import d4rl  # Not needed for online training')
            with open(utils_path, 'w') as f:
                f.write(content)
            print("Patched d4rl import in utils.py")

    # Also patch ipdb imports
    for filepath in ['synther/diffusion/utils.py', 'synther/diffusion/denoiser_network_cond.py']:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
            if 'from ipdb import' in content:
                content = content.replace('from ipdb import set_trace as st', '# from ipdb import set_trace as st')
                with open(filepath, 'w') as f:
                    f.write(content)
                print(f"Patched ipdb import in {filepath}")

    # 6. Verify setup
    print("\n" + "=" * 60)
    print("Step 6: Verification")
    print("=" * 60)
    verify_script = (
        "import torch; "
        "print('PyTorch:', torch.__version__); "
        "print('CUDA:', torch.cuda.is_available()); "
        "import gym; print('Gym:', gym.__version__); "
        "import dmcgym; print('dmcgym: OK'); "
        "import dm_control; print('dm_control: OK'); "
        "env = gym.make('cheetah-run-v0'); obs = env.reset(); "
        "print('cheetah-run-v0: obs_dim=%d, act_dim=%d' % (obs.shape[0], env.action_space.shape[0])); "
        "env.close(); "
        "import gin; print('gin-config: OK'); "
        "from redq.algos.core import ReplayBuffer; print('REDQ: OK'); "
        "print('All checks passed!')"
    )
    result = run(f"{sys.executable} -c \"{verify_script}\"", check=False)
    if result.returncode != 0:
        print("\nSome checks failed. Review errors above.")
    else:
        print("\nSetup complete! Ready to run experiments.")


if __name__ == '__main__':
    setup()
