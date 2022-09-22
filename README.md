# patchmatching

![example](https://img.shields.io/badge/Python-3.x-blue.svg)

<center><em>"Once upon a time<br>
You randomly croped some tiles<br>
From some whole slides<br>
Where does the tile lie in slide<br>
Pixel-wise comparison will let you die<br>
Thank God that here is a light<br>
`patchmatching` will save your life"</em></center><br>

`patchmatching` is a tool that can help you efficiently find the location of all patches in one of several WSIs (whole slide images).

# Table of Contents

- [patchmatching](#patchmatching)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
  - [Output](#output)
- [API reference](#api-reference)
  - [`match` function](#match-function)
  - [`save_json` function](#save_json-function)

# Installation

1. Clone or download
    - Use the command bellow in terminal to clone this repo:
    ```
    git clone https://github.com/samson6460/patchmatching.git
    ```

    - Or just download whole files using the **[Code > Download ZIP]** button in the upper right corner.
    
2. Install dependent packages: 
    ```
    cd patchmatching
    pip install -r requirements.txt
    ```

    > If you have problems installing openslide, please refer to the link: https://openslide.org/api/python/.

# Usage

Even you have no idea about which patches belong to which WSI, just put all the patches in one folder and all the possible WSIs in another folder.

Then run the following command in your terminal ... done!

```
python match.py dir_wsi dir_patch [-lv LEVEL] [-ow ORIGINAL_WIDTH] [-oh ORIGINAL_HEIGHT] [-ds DSAMPLE] [-t THRESHOLD] [-fr FINETUNE_RANGE] [-ft FINETUNE_THRESH] [-v] [-o OUTPUT]
```

- **positional arguments**:
  - **dir_wsi**: directory of whole slide images
  - **dir_patch**: directory of patch images

- **optional arguments**:
  - **-h, --help**: show this help message and exit.
  - **-lv, --level**: level number for cropping patches from WSI.
  - **-ow, --original_width**: original width of cropped patches.
  - **-oh, --original_height**: original height of cropped patches.
  - **-ds, --dsample**: downsampling rate for matching.
  - **-t, --threshold**: threshold for SIFT algorithm.
  - **-fr, --finetune_range**: the pixel range for validating matched patch.
  - **-ft, --finetune_thresh**: threshold for fine-tuning.
  - **-v, --verbose**: display cropping and patch image.
  - **-o, --output**: path of output file.

## Output

The output file will be saved as a json file and the format will look like this:

```
{
    "XXXX1.jpg": {
        "WSI name": "aabbcc.svs",
        "start coord": [23311, 8140]
    },
    "XXXX2.jpg": {
        "WSI name": "aabbcc.svs",
        "start coord": [20989, 11200]
    },
    "XXXX3.jpg": {
        "WSI name": "ccbbaa.tiff",
        "start coord": [24820, 9847]
    },
    .
    .
    .
}
```

`"start coord"` refers to the location (top left anchor) of patch in WSI of level 0.

# API reference

You can also use the following API functions in an interactive environment like **jupyter notebook**.

## `match` function

```
match.match(dir_wsi: str, dir_patch: str,
    level=0, ori_crop_size=(3432, 1884),
    dsample_match=8, threshold=0.4,
    finetune_range=60, finetune_thresh=0.9,
    debug_mode=False)
```

- **dir_wsi**: string, directory of whole slide images.
- **dir_patch**: string, directory of patch images.
- **level**: integer, level number for cropping patches from WSI.
- **ori_crop_size**: tuple of two integers,
            original size of cropped patches.
- **dsample_match**: integer, downsampling rate for matching.
- **threshold**: float (0~1), threshold for SIFT algorithm.
- **finetune_range**: integer, the pixel range for validating matched patch.
- **finetune_thresh**: float (0~1), threshold for fine-tuning.
- **debug_mode**: boolean, whether to show cropping and patch image.

***Returns***

A dictionary contains matching information.

## `save_json` function

```
match.save_json(matched_dict: dict, path: str):
```

- **matched_dict**: dictionary, the return from match().
- **path**: string, path of json file.
