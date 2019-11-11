#!/bin/bash
search_dir="$1"
outdir=$search_dir
outdir+="/reconstructions/"
if [ ! -d "$outdir" ]; then
	mkdir $outdir
fi
slicesdir=$search_dir
slicesdir+="/slices"
slicesrecdir=$slicesdir
slicesrecdir+="/reconstructions"
#echo "$search_dir"
for stack in "$search_dir"/*.tif
do
    #echo "$stack"
	#echo $search_dir$entry" "$slicesdir
	if [ -d $slicesdir ]; then
		rm -rf "$slicesdir"
	fi
	mkdir $slicesdir
	~/ivp/Fiji.app/ImageJ-linux64 --headless -macro convert2slices.ijm $stack" "$slicesdir >/dev/null 2>/dev/null
	mkdir $slicesrecdir
	#TODO reconstruction script here
	for slice in "$slicesdir"/*.png
	do
		b=$(basename $slice)
		#echo "$slicesrecdir"/"$b"
		cp "$slice" "$slicesrecdir"/"$b"
	done
	#
	~/ivp/Fiji.app/ImageJ-linux64 --headless -macro convert2stack.ijm $slicesrecdir" "$outdir"rec-"$(basename $stack) >/dev/null 2>/dev/null
done
if [ -d $slicesdir ]; then
	rm -rf "$slicesdir"
fi
