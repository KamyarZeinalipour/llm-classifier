from setuptools import setup, find_packages

setup(
    name="llm-classifier",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"] ),
    install_requires=[
        "pandas>=1.5",
        "openai>=0.27",
        "scikit-learn>=1.2",
        "matplotlib>=3.6"
    ],
    extras_require={
        "deepseek": ["openai>=0.27"]
    },
    entry_points={
        'console_scripts': [
            'llm-classifier=llm_classifier.cli:main',
        ],
    },
    python_requires='>=3.8',
)
