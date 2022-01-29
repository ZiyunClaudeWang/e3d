
'''

all_sequences_train = []
all_sequences_val = []

with open(os.path.join(path, "train_split.txt"), 'r') as train_file:
    for f in train_file.readlines():
        f = f.strip("\n")
        path = Path(f.path)
        if "h5" in f:
            all_sequences_train.append(os.path.join(self.path, f))

with open(os.path.join(self.path, "test_split.txt"), 'r') as test_file:
    for f in test_file.readlines():
        f = f.strip("\n")
        path = Path(self.path)
        if "h5" in f:
            all_sequences_val.append(os.path.join(self.path, f))

all_sequences_train, all_sequences_val = self.generate_paths()
all_datasets_train = []
all_datasets_val = []

data_class = EventSegHDF5
for i in range(len(all_sequences_train)):
    dataset_one = data_class(all_sequences_train[i],
                            width=self.width,
                            height=self.height,
                            max_length=-1)
    all_datasets_train.append(dataset_one)
for i in range(len(all_sequences_val)):
    dataset_one = data_class(all_sequences_val[i],
                            width=self.width,
                            height=self.height,
                            max_length=-1)
    all_datasets_val.append(dataset_one)

self.train_dataset = ConcatDataset(all_datasets_train)
self.val_dataset = ConcatDataset(all_datasets_val)

for i, data in enumerate(train_loader):
    pdb.set_trace()
    loss = self.step(data, i, mode='train')
'''
