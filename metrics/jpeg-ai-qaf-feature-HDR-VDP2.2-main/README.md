# JPEG-AI metrics

JPEG-AI Objective Quality Assessment Framework

## System requirments

Scripts work on Ubuntu 18.04 ro higher. 
Should you have problem with `vmaf`, please manually compile it based on source code from [here](https://github.com/Netflix/vmaf) with tag [v2.2.1](https://github.com/Netflix/vmaf/tree/v2.2.1) and copy executable vile as `vmaf.linux` to directory of JPEG-AI Objective Quality Assessment Framework.
## Structure of directories with results

Bitstreams and reconstructed files of different codecs should be organized as following:

    SUBMISSIONS
        CODEC1
            bit
                CODEC1_00016_TE_1744x2000_8bit_sRGB_200.bits
                …
            rec
                CODEC1_00016_TE_1744x2000_8bit_sRGB_200.png
                …
        ...
        CODEC2
            bit
                CODEC2_00016_TE_1744x2000_8bit_sRGB_200.bits
                …
            rec
                CODEC2_00016_TE_1744x2000_8bit_sRGB_200.png
                …                

Where CODEC1 and CODEC2 are the names of the codecs. ``bit`` and ``rec`` are the directories with bitstreams and reconstructed files.

## How to set up

1) clone the repository:  
2) go to your local copy of jpeg_ai_metrics: ``cd jpeg-ai-qaf``
3) create conda environment: ``conda create -n jpeg_ai_metrics python=3.6.7``
4) activate environment: ``conda activate jpeg_ai_metrics``
5) Upgrade `pip` by a command `python -m pip install --upgrade pip`
6) install dependencies: ``pip install -r requirements.txt``


## How to run

### Downloading

VMAF downloads automatically by scripts. If it doesn't happen, you may download VMAF manually by the following commands:

1) Download [vmaf.linux](https://github.com/Netflix/vmaf/releases/download/v2.2.1/vmaf) version 2.2.1 to the root directory of the project by command:
``wget https://github.com/Netflix/vmaf/releases/download/v2.2.1/vmaf -O vmaf.linux``

2) make it executable: `chmod +x vmaf.linux`

### Iterate over all images

Follow these steps in your local copy of the scripts:

1) activate environment: ``conda activate jpeg_ai_metrics``
2) ``python main.py <PATH_TO_ORIGs> <PATH_TO_BASE_DIR> <CODEC_NAME>``, where `<PATH_TO_ORIGs>` is a path to original files, `<PATH_TO_BASE_DIR>` is a path to base directory with results (`SUBMISSIONS` in a section `Structure of directories with results`), `<CODEC_NAME>` is a name of the codec under test.

By default all supported metrics are performed. If you would like to perform only some of them, set the list of necessary metrics by using command line parameter ``--metrics``. If you would like to calculate only PSNR and MSSSIM, add to command line the following arguments ``--metrics msssim_torch msssim_iqa psnr``.

Format of output summary is tab-separated text file. CSV format is also supported. To have output file in CSV format add `--csv` to command line.

List of the default BPPs to be tested by the framework is ``0.03, 0.06, 0.12, 0.25, 0.50, 0.75``. It's controled by parameter `--rates`. Rates listed there as 1/100 part of bits per pixel. To set rates to ``0.03, 0.06, 0.12`` add command line parameter ``--rates 3 6 12``.

if you need to calculate only distortion metrics without BPP calculation, use command line parameter ``--no-bpp``. It will fill corresponding column by zeros.


See all available options by running: ``python main.py -h``



## Naming format for input files:


Naming convention took from document [`wg1n92048-ICQ-JPEG AI Common Training and Test Conditions`](https://sd.iso.org/documents/ui/#!/browse/iso/iso-iec-jtc-1/iso-iec-jtc-1-sc-29/iso-iec-jtc-1-sc-29-wg-1/library/6/92-Online/OUTPUT%20N-documents/wg1n92048-ICQ-JPEG%20AI%20Common%20Training%20and%20Test%20Conditions) (section 9).

### Original images
`<Name>_<Width>x<Height>.png`

where `<Name>` is a name of the image file, `<Width>` and `<Height>` are width and height of the image

Example of the naming:
`00001_TE_1744x2000.png`

### Reconstrusted images

`<Codec>_<Name>_<Width>x<Height>_<bit>bit_<Format>_<QP>.png`

where `<Codec>` is a name of the codec, `<Name>` is a name of the image file, `<Width>` and `<Height>` are width and height of the image, `<bit>` is a bit-depth, `<Format>` is a format of the data (only YUV444 is currenlty supported), `<QP>` is a quality parameter.

Example of the naming:
`HEVC_00016_TE_1744x2000_8bit_sRGB_025.png`


### Bitrstream files

`<Codec>_<Name>_<QP>.bits`

where `<Codec>` is a name of the codec, `<Name>` is a name of the image file, `<QP>` is a quality parameter.

Example of the naming:
`HEVC_00016_TE_025.bits`

## Reporting template

The file `reporting_template.xlsm` is the official JPEG AI reporting template where BD rates are computed, RD plots are shown for several metrics and decoding complexity can be reported. 
