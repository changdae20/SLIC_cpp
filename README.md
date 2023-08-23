# SLIC ![C++](https://img.shields.io/badge/-C++-505050?logo=c%2B%2B&style=flat)

Paper : [Link](https://core.ac.uk/download/pdf/147983593.pdf)

- Superpixel algorithm using simple iterative method
- Segmentation using Normalized Cut

<img src="docs/1.png">

<img src="docs/2.png">
Original Image (left side), Image by 400 Superpixels (right side) 

<img src="docs/3.png">

Bipartited image using Normalized Cut algorithm.


## Usage
```
chmod +x build.sh
./build.sh
./bin/SLIC <image path>
```

## Dependency

### C++17 (gcc >= 8.x)
### Opencv
```sudo apt install libopencv-dev```

## License
MIT License, 2023 @changdae20.