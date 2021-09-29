
import os



# A function to create directory 
def create_dir(path_of_dir):
    try:
        os.makedirs(path_of_dir) # For one directory containing inner/sub directory(ies)    
    except FileExistsError:
        #print("Directory %s already exists" % path_of_dir)   
        pass
    except OSError:
        print ("Creation of the directory %s failed" % path_of_dir)     
    else:
        #print ("Successfully created the directory %s " % path_of_dir) 
        pass
