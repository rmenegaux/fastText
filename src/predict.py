import time
import subprocess

# -----------------
# File Paths
# -----------------
# root_path = '/mnt/data40T/rmenegaux/'
root_path = '/Users/romainmenegaux/These/'
data_path = '{}these_romain/data/'.format(root_path)
output_path = '{}these_romain/fasttext/output/'.format(root_path)
fasttext = '{}fastText/fasttext'.format(root_path)

# -----------------
# Data parameters
# -----------------
#FIXME consider using a dict for parameters
dataset = 'small'
# train_labels = '{}train_small-db.gi2taxid'.format(data_path)
test_dataset_path = '{}/{}-DB/simulated-dataset/'.format(data_path, dataset)
# valid_data = '{}/NS2.noerror.200-reads.fasta'.format(test_dataset_path)
valid_data = '{}/test.fragments.fasta'.format(test_dataset_path)
# valid_data = '/Users/romainmenegaux/These/opal-package/data/A1.10.1000/test/A1.10.1000.test.fasta'

#valid_data = '{}/test.fragments.labelled.nosplit.txt'.format(output_path)
#valid_data = '{}test.fasttext.large.txt'.format(output_path)
# train_data = valid_data

# -----------------
# Model parameters
# -----------------
# embedding dimension
d = 10
# epochs
e = 2
# learning rate
lr = .1
# k-mer length
k = 10

model_name = '{}models/fdna_{}_k{}_d{}_e{}_lr{}'.format(output_path, dataset, k, d, e, lr)
model_name = '{}models/fdna_model_k12_d100_e100_lr0.1.ftz'.format(output_path)
pred_path = '{}predictions/fdna_{}_k{}_d{}_e{}_lr{}.pred'.format(output_path, dataset, k, d, e, lr)
# -----------------
# Train model
# -----------------
print('Testing model -k {} -d {} -e {} -lr {}'.format(k, d, e, lr))

# -----------------
# Test model
# -----------------
start = time.time()
subprocess.check_call(
	'''
	{} predict {} {} > {}
	'''.format(fasttext, model_name, valid_data, pred_path),
	shell=True
)
end = time.time()
print('Predicted test set, elapsed time {:d}s'.format(int(end-start)))