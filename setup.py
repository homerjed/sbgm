from setuptools import find_packages, setup

setup(
    name="sgm",
    packages=find_packages(where="sgm"),
    package_dir={"" : "sgm"},
    url="https://github.com/homerjed/sgm/",
    author="Jed Homer",
    author_email="jedhmr@gmail.com",
    license="MIT",
    keywords=[
        "artificial intelligence",
        "machine learning",
        "diffusion",
        "score based diffusion",
        "generative models"
    ],
    install_requires=[
        "jax",
        "equinox",
        "diffrax",
        "optax",
        "einops",
        "numpy",
        "matplotlib",
        "cloudpickle",
        "torch",
        "torchvision",
        "tqdm",
        "powerbox"
    ],
)