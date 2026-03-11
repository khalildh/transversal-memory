from setuptools import setup, find_packages

setup(
    name="transversal_memory",
    version="0.1.0",
    description="Content-addressable memory via projective geometry and Schubert calculus",
    packages=find_packages(),
    install_requires=["numpy>=1.24", "scipy>=1.10"],
    python_requires=">=3.9",
)
