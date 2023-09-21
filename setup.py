from setuptools import setup, find_packages

setup(
    name='Kasthuri_Challenge',
    version='1.0.0',
    url='https://mtneuro.github.io/',
    license='MIT License',
    author='Kasthuri',
    author_email='evadyer@gatech.edu',
    python_requires=">=3.6.0",
    description='Kasthuri et. al. 2015',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "Pillow",
        "intern",
        "umap-learn",
        "torchsummary",
        "torchvision>=0.5.0",
        "torch==1.11.0+cu116",
        "pretrainedmodels==0.7.4",
        "efficientnet-pytorch==0.6.3",
        "timm==0.4.12",
        "rich",
        "opencv-python",
        "einops",
        "kornia==0.6.0",
        "connected-components-3d",
        "segmentation-models-pytorch @ https://github.com/qubvel/segmentation_models.pytorch/archive/740dab561ccf54a9ae4bb5bda3b8b18df3790025.zip#egg=segmentation-models-pytorch-0.3.0-dev"
    ],
    zip_safe=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only"
    ]
    )
