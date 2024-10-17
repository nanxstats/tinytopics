from setuptools import setup, find_packages

setup(
    name="tinytopics",
    version="0.1.0",
    description="Fit topic models using neural Poisson non-negative matrix factorization with sum-to-one constraints",
    author="Nan Xiao",
    author_email="me@nanx.me",
    packages=find_packages(),
    install_requires=["torch", "numpy", "matplotlib", "scipy", "tqdm"],
    python_requires=">=3.8",
)
