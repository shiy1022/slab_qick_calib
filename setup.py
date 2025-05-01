from setuptools import setup, find_packages

setup(
    name="slab_qick_calib",
    version="0.1.0",
    description="Qubit calibration tools for QICK",
    author="",
    author_email="",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        # Add your dependencies here
        # e.g., "numpy>=1.18.0",
        # "matplotlib>=3.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
