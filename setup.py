from setuptools import setup, find_packages

setup(
    name="ml_from_scratch",
    version="0.1.0",
    author="Your Name",
    description="Machine Learning algorithms implemented from scratch",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.8+",
    ],
)