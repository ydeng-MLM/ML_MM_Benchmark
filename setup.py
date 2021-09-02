import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AEM-Benchmark",
    version="0.0.2",
    author="Yang Deng, Juncheng Dong, Simiao Ren",
    author_email="yang.deng@duke.edu",
    description="Package for benchmarking deep learning models on AEM problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ydeng-MLM/ML_MM_Benchmark",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)