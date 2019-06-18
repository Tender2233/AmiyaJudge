# AmiyaJudge
AmiyaJudge


# status  
AmiyaJudhe is now public.We are happy to see you now find this repo.  

# code part  
The model is written by keras in the modol.py(not model.py)  
If you are wandering how we get the data ,just check the python Crawler,but please make share you won't
borther others.


# data process  
before you make your onw tfrecord file.You need to preprocess the imgs into 100x100x1.This part make sure
 the fury's color won't affect the model.the less data image you have ,the color affects even more disappointing.  

# trans the data to model  
after you preprocess the img data.
you can use the python code convert the data into tfrecord.
we recomment you to choose datasetreader.py to trans the data to the model.
In this way ,you will not find the tfrecord way type code we write is rubbish.

# finally  
I hope you have alreay find the amiya.h5 file.
just use it in keras way.