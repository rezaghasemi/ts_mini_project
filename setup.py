from setuptools import setup, find_packages

setup(
    name="time_series_forecasting",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],  # add any runtime dependencies if needed
)
