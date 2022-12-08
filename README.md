This is Thai Automatic number-plate recognition power by Yolov5 nano. There are 4 models work together.

Car object detection 640px (1 Class, detect car in the image)
License plate object detection 640px (1 Class, detect license plate in the image)
Letter object detection 640px (2 Classes, letter and province, detect x-y coordinate of alphabets on the license plate image)
Letter classification 224px (48 Classes, 38 alphabets and 10 numbers, classify alphabet images)
The main folder is named "yolo" Please "cd" to "yolo" and type "pip install requirements.txt" to install requirement library.

"run_model.ipynb" is full of analysis tools and code comment Please try to understand code in the file "run_model.ipynb" first

the main input image folder is "test_folder" the main output image folder is "my_output" the weight files of 4 models is in "my_weight"

"run_input_from_folder.ipynb" is the model get input from "test_folder" "run_input_from_capture_screen.ipynb" is the model get input from screen capture.

The model will perform faster if you run the model without image processing but it slightly trades with accuracy.

Since yolov5 classification model has no auto resize, we need to resize by ourself. We need to deal with "You must pass torch tensors in BCHW to this model, i.e. shape(1,3,224,224).". However, I've already done that for you.

if you need openCV to access file in your new folder, please make sure that you have already create that new folder otherwise openCV will give you an error.

if you have any question, you can contact me via "sif332@gmail.com".
