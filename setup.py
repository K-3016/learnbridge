"""Package configuration; AI assistance attribution is documented in README.md."""
from setuptools import find_packages, setup


def main() -> None:
    """Configure the installable package without running training side effects."""
    setup(name="learnbridge",version="0.1.0",description="Responsible learning resource recommender",package_dir={"":"src"},packages=find_packages("src"),python_requires=">=3.10",install_requires=["numpy>=1.24,<3","pandas>=2.0,<3","scikit-learn>=1.3,<2","scipy>=1.10,<2","joblib>=1.3,<2"])


if __name__ == "__main__":
    main()
