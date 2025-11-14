from setuptools import setup, find_packages

setup(
    name="clrernet",
    version="1.0.0",
    packages=find_packages(include=["libs", "libs.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy",
        "opencv-python",
        "scipy",
        "pyyaml",
        "tqdm",
    ],
)
