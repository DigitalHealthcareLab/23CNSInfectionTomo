# Deep-Learning model for central nervous system infection diagnosis and prognosis using label-free 3D immune-cell morphology in the cerebrospinal fluid

## Overview
![Figure1](img/Figure1.jpg)


## 1. Patient enroll and sample preparation (A)
We performed a prospective cohort study between January and October 2022 at Severance Hospital in Seoul, Republic of Korea. 

Patients who met the following inclusion criteria were enrolled: 
- (1) presenting with fever, headache, and other neurological abnormalities prompting clinical suspicion of encephalitis or meningitis; 
- (2) >10 cells/μL in the CSF sample; and (3) age >18 years. 

Between January and October 2022, <b>14 patients with CNS infections were enrolled in this study.</b>

## 2. 3D Cell Image Acquisition & Preprocessing
### 2-1. Cell Image Acquisition
All 3D cell structural images of the immune cells in the CSF were captured using <b>3D holotomography </b>(HT-2H; Tomocube Inc., Daejeon, Republic of Korea), which reconstructs 3D RI portrayals by merging various 2D quantitative phase images.


### 2-2. Image preprocessing
<b>Exclusion : </b>
- Manual exclusion of sub-par instances (encompassing low-quality, low-resolution, and noisy backdrop depictions)
- Images exhibiting excessive adjacency of two or more cells

<b>Preprocessing :</b>
- <u>Centre-cropping process</u> to a size of 64 × 64 × 64 pixels & <u>centred around the morphological centre</u> co-ordinates determined manually during the previous curation process.


=><b>Through this process, 463 images were excluded from the initial set of 1,890 images, resulting in 1,427 images being used for training.</b> 

### 2-3. Sample data sharing
Here is the Google Drive link for all <u>processed NumPy files and part of Raw data</u> of 3D label-free immune cells from cerebrospinal fluids of the patients with central nervous infection at the <b>[Google Drive](https://drive.google.com/drive/folders/1---UpcPPSYeLeMHsh9VxgRubVkXziMeZ?usp=sharing)</b>



## 3. Deep Learning Model Application
![Figure3](img/Figure3.jpg)

### 3-1. Model and Hyperparameters
- Model : DenseNet
- Loss function : Cross Entropy Loss
- Optimizer : SGD (lr 0.0001) & cosine annealing scheduler 
- Batch size : 8
- Requires VRAM at least 16GB for training

### 3-2. How to Run?
#### 1) <b> Directory Description</b>
<pre>
Root
├── img
│   ├── Figure1.jpg : Overview
│   └── Figure3.jpg : Result
├── src : source codes for training the models and validate the result
│   ├── accuracy.py
│   ├── ...
│   └── weight_initialization.py
├── check_accuracy.py
├── config.py : Configure of the training details
├── create_csf_task_bacteria_virus.py : 
├── draw_gradcam.py : Interpretation
├── main_csf_bacteria_virus.py : Main file for training the file
├── Readme.md : this file
└── task.yaml : User can specify the task for running.
</pre>



#### 2) How to start?
<b> Easy start </b>
Simply, run the code below
```bash
bash run_all.sh
```

or
<b> 2-1) create task</b>
```bash
python create_csf_task_bacteria_virus.py
```

<b> 2-2) train model </b>
```bash
python main_csf_task_bacteria_virus.py
```

<b> 2-3) get results </b>
```bash
python check_accuracy.py
```

<b> 2-4) interpretation </b>
```bash
python draw_gradcam.py
```

