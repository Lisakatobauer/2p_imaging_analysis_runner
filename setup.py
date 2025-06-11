from setuptools import setup, find_packages

setup(
    name='twop-imaging-analysis-runner',
    version='0.1.0',
    description='A modular pipeline for analyzing 2-photon imaging data using Suite2p and custom components.',
    author='lisa Bauer',
    author_email='lbauer@bi.mpg.de',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.4',
        'tifffile>=2024.8.30',
        'scikit-image>=0.24.0',
        'scipy>=1.13.1',
        'suite2p>=0.14.4',
        'importlib-metadata>=7.0.1',
        'scikit-learn>=1.5.1'
    ],
    python_requires='>=3.8',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    include_package_data=True,
    zip_safe=False
)
