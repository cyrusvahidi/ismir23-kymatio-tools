from setuptools import setup

setup(
    name="scattering-tools",
    version="0.1",
    description="",
    author="Cyrus Vahidi",
    author_email="c.vahidi@qmul.ac.uk",
    include_package_data=True,
    packages=['s1dt'],
    url="https://github.com/cyrusvahidi/scattering-tools",
    install_requires=[
        "torchaudio",
        "torch",
        "numpy",
        "scipy",
        "scikit-learn"
    ],
)
