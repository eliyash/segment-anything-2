# CCR Dataset

Video Dataset of Wild Chimpanzees with Identities from _Count, Crop and Recognise: Fine-Grained Recognition in the Wild_ 

If you want to use the dataset in a publication, please cite the paper:
```
@InProceedings{Bain19,
  author       = "Max Bain and Arsha Nagrani and Daniel Schofield and Andrew Zisserman",
  title        = "Count, Crop and Recognise: Fine-Grained Recognition in the Wild",
  booktitle    = "Workshop on Computer Vision for Wildlife Conservation, ICCV",
  year         = "2019",
}	    
```

Directory Information
---------------------------

 - annotations/
    - face_data.csv   :   Face detections with corresponding identity
    - body_data.csv   :   Body detections and corresponding identity
    - frame_data.csv  :   Timestamps denoting when a given individual is visible within the frame
   
 - lists/
    - classes.txt :   list of categories (individuals)
    - splits.txt  :   training / testing splits of videos
 
 - config.json
    Dataset configuration file
    
 - dataset.py
    Pytorch Dataset classes
    
 - demo_dataset.py Examples of the dataset with annotations
    
 - `extract.sh` Extract video frames


1. Download video files, move to directory where you store data

2. Change `$DATASET_DIR$` in `config.txt` to your dataset directory.

3. `run extract.sh` to extract frames. Note this step requires 700 GB of storage

4. `python demo_dataset.py`

 
 
