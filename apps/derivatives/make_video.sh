ls iter*.ps | xargs -n1 -P48 bash ./convert_ps_to_png.sh 
avconv -framerate 30 -f image2 -i iter_%d.png -c:v h264 out.mp4
