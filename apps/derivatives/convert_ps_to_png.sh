echo $1
ps2pdf $1 ${1/ps/pdf}
convert ${1/ps/pdf} -background white -alpha remove -density 100 png24:${1/ps/png}

