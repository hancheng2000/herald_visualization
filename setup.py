from setuptools import setup, find_packages

setup(
    name="herald_visualization",
    version="0.1.0",
    description="A package for visualizing data in herald project",
    author="Hancheng Zhao, Chris Eschler",
    author_email="zhaohc@umich.edu",
    # url="",
    packages=["herald_visualization"],  # Automatically find subpackages
    install_requires=['numpy','pandas','navani','matplotlib','xmltodict'],       # Add dependencies here
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)