echo $1
ps2pdf $1 ${1/ps/pdf}
convert -density 200 ${1/ps/pdf} -background white -alpha remove png24:${1/ps/png}
~/local/ImageStack/bin/ImageStack -load ${1/ps/png} -resample 1620 1080 -save ${1/ps/png}
