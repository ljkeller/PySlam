# PySlam

PySlam is a python SLAM implementation with design decisions driven from ORB SLAM 1&2 [ORB SLAM2](https://github.com/raulmur/ORB_SLAM2). Build using Python3, Pangolin, and OpenCV.

![image](https://user-images.githubusercontent.com/44109284/115982283-79ddfe80-a55f-11eb-85e6-b3e9adadbe21.png)

## Getting Started

With the installed dependencies and prerequisites, running PiSLAM is as simple as
```
python PiSLAM -i /path/to/sequence
```

in local sequence mode, or
```
python PiSLAM -r
```
in network SLAM mode, followed by launching `piVideoStream.py` file with modified PORT and IP addresses, like
```
python rpi/piVideoStream.py
```

**_NOTE:_** To view PiSLAMs command line arguments, run
```
python PiSLAM --help
```

### Prerequisites

Tested on Ubuntu 20.04

Python 3.7.0/3.8.5

Python Pangolin bindings built and installed (see Built With section)

Requirements can be found in requirements.txt

### Installing

pip install -r requirements.txt

## Built With

* [Opencv](https://opencv-python-tutroals.readthedocs.io/en/latest/)
* [Pangolin](https://github.com/uoip/pangolin)

## Authors

* **Lucas Keller & Ian Gluesing** - *Related work* - [ORB SLAM2](https://github.com/raulmur/ORB_SLAM2)

## Acknowledgments

* Inspiration : HCI 575 : Computational Perception final project
