# HDR-VDP2.2 is a full reference image quality assessment metric.




## How to run
1) The reference input image should be YUV444 or png images 
2) Standard command line:
python main.py dataset codecs HEVC --metrics hdr_vdp --hdr-arg version=VDP3,task=quality,width=3840,height=2160,display_diagonal_in=64.5,viewing_distance=1.32528,EOTF=pq,color_encoding=rgb-bt.709

3) Version can be VDP2 and VDP3 to run VDP2 or VDP3 
4) Task can be quality, side-by-side and flicker if version=VDP3   
    'side-by-side' - side-by-side comparison of two images  
    'flicker' - the comparison of two images shown in the same place and swapped every 0.5 second.  
    'quality' - prediction of image quality (Q_JOD)  
    In task side-by-side and flicker we will save the visualization as .npy file.  
5) Width ,height and display_diagonal_in are the display parameters
6) EOTF can be pq and HLG
7) color encode:  
    In VDP2 we support only BT709   
    In VDP3 we support BT709 and BT2020  
