from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="nexus-game-ai",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "nexus=nexus.cli.main:main",
            "serpent=nexus.cli.main:serpent_compat",
        ],
    },
)
ENDFILE < /dev/null
