### This project is tested with `python=3.10`

# Pre-requisites
1. Clone the repository with the following command: 
```bash
git clone https://github.com/Emvista/Meta-XAMR-2024.git
```

2. Clone the following repository to the root of the project: 
```bash 
git clone https://github.com/RikVN/AMR.git

# add AMR to the python path  
export PYTHONPATH=${PYTHONPATH}:${PWD}/AMR

# install the required packages for AMR 
pip install -r AMR/requirements.txt
```

3. Install the required packages for the project: 
```bash
pip install -r requirements.txt
``` 

4. Place your AMR data in the following tree structure: 

```bash
data
├── amr
│   ├── en
│   │   ├── train
│   │   │   ├── en-amr.amr 
│   │   │   ├── en-amr.pm
│   │   │   ├── en-amr.en
│   │   ├── dev 
│   │   └── test
│   └── de
│       ├── train
│       │   ├── de-amr.amr
...
```
- `en-amr.amr` contains linearized AMR graphs
- `en-amr.pm` contains AMR graphs in Penman notation, where each graph is separated by an empty line
- `en-amr.en` contains the corresponding English sentences





# Train a model
```bash 
# To train a model using maml, see the script for more options 
python train_maml_xlingual.py 

# To train a model using baseline, see the script for more options 
python train_baseilne_xlingual.py 
```
# Evaluate a model 

```bash 
# To evaluate the model, set the `--max_steps` to 0
# and `--resume_from_checkpoint` to True
# and specify the `--checkpoint_path` 
python train_maml_xlingual.py --max_steps 0 --resume_from_checkpoint True --checkpoint_path <path_to_checkpoint>
```

# GUI 
You can test our best model with gui. This will run a server on your local machine and you can access the page by going to the following link: `http://127.0.0.1:7860/`

1. Download the model from the following [link](https://drive.google.com/file/d/1IwGmlufzDrIKwmOAKMob18VMS1-aKHzj/view?usp=sharing) and place it in the folder `checkpoint/`   
2. Run the following command: 

```bash 
python gui_amr_parser.py 
```

# Citation 
If you use this code, please cite the following paper: 
```bibtex
to be added 
```



