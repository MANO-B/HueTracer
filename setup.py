from setuptools import setup, find_packages

setup(
    name='huetracer',
    version='0.0.3',
    description='Analyze cell-cell interaction with spatial transcriptome data.',
    author='Masachika Ikegami',
    author_email='ikegamitky@gmail.com',
    url='https://github.com/MANO-B/HueTracer',
    package_dir={'': 'src'},  # ★ポイント：srcからパッケージを探す
    packages=find_packages(where='src'),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'tqdm',
        'scanpy',
        'scikit-image',
        'opencv-python',
        'pandas',
        'torch',
        'plotly',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.7',
)
