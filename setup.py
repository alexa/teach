# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="TEACh",
    version="1.0",
    description="Task-driven Embodied Agents that Chat",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexa/TEACh",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        "console_scripts": [
            "teach_download = teach.cli.download:main",
            "teach_eval = teach.cli.eval:main",
            "teach_inference = teach.cli.inference:main",
            "teach_replay = teach.cli.replay:main",
            "teach_api = teach.cli.api:main",
        ]
    },
    include_package_data=True,
    python_requires=">=3.7",
    zip_safe=False,
    install_requires=[
        "ai2thor==3.1.0",
        "boto3==1.15.2",
        "fuzzywuzzy==0.18.0",
        "networkx==2.5",
        "pydub==0.24.1",
        "python-Levenshtein",
        "tqdm",
    ],
)
