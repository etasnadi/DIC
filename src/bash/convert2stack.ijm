args = split(getArgument(), " ");
run("Image Sequence...", "open="+args[0]+" sort");
saveAs("Tiff", args[1]);
close();