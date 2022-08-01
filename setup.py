import setuptools

setuptools.setup(
    name='mmky',
    version='1.0',
    url="https://github.com/drmj/mechanicalmonkey",
    install_requires=[
        'gym',
        'torch',
        'opencv-python',
        'pyyaml',
        'h5py',
        'roman@git+https://github.com/microsoft/roman.git'
    ],
    packages=setuptools.find_packages(),
) 