from setuptools import setup, find_packages

setup(
    name="bg_score_extraction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "opencv-python",
        "detectron2"
    ],
    author="Detectron2 User",
    author_email="user@example.com",
    description="Background score extraction for Detectron2 models",
    keywords="detectron2, computer vision, object detection, background score",
    url="https://github.com/yourusername/bg_score_extraction",
) 