from setuptools import find_packages, setup

setup(
    name='unsupervised-absa',
    packages=find_packages(include=['unsupervised_absa']),
    version='0.1.0',
    description='A natural language processing toolkit for extracting aspect based sentiment analysis dataset.',
    author='Steven Limcorn',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)