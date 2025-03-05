task=$1
task_name=$2

python create_task.py --task $task --task_name $task_name
python main.py --task $task --task_name $task_name
python check_accuracy.py --task $task --task_name $task_name


### TASK 
# task = 'prognosis'
# task_name = 'csf_good_poor'

# task = 'virus_others'
# task_name = 'csf_vo'