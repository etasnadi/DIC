//arg = getArgument()
//args[0] = input stack image path
//args[1] = output filepath
args = split(getArgument(), " ");
//open("/home/koosk/BRC/images/stack_images/tissues/20170425_1-50/tissue001.tif");
//run("Image Sequence... ", "format=PNG start=0 digits=3 save=/home/koosk/BRC/images/stack_images/tissues/20170425_1-50/slices/");
//print(getArgument());
open(args[0]);
run("Image Sequence... ", "format=PNG start=0 digits=3 save="+args[1]);
close();