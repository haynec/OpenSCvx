from setuptools import setup, find_packages

setup(
    name="OpenSCvx",
    version="0.0.1",
    author="Chris Hayner",
    author_email="haynec@uw.edu",
    description="A general Python-based successive convexification implementation which uses a JAX backend.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/haynec/OpenSCvx",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache 2.0 License",
        "Operating System :: Linux, MacOS, Windows",
    ],
    python_requires='>=3.9',
    install_requires=[
        "jax",
        "numpy",
        "plotly",
        "termcolor",
        "cvxpy",
        "cvxpygen",
        "qoco",
        "diffrax"
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'openscvx=main:run_simulation',
        ],
    },
)