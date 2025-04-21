from setuptools import setup, find_packages

EXCLUDE_DIRS  = ["*scripts*", "*web_gui*", "*tests*", "*verification*", "*colourbars*", "*static*"]

setup(name="Integrated-convective-nowcasting",
      version='0.1',
      description="Integrated convective nowcasting",
      author="Yunman",
      author_email=None,
      url="https://github.com/YUN-MAN/Integrated-convective-nowcasting.git",
      packages=find_packages(exclude=EXCLUDE_DIRS),
      #include_package_data=True,
      entry_points={
            'console_scripts': [
                'celltracking = wuct.celltracking.celltracking:main',
                'moctimggen = wuct.imagegen.plthdfclass:main',
                'moctpltadv = wuct.imagegen.pltadv:main'
            ]
        }
      )