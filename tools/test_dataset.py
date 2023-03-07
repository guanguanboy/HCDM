#from data.dataset import SSHarmonizationTestDataset
import os
print(os.curdir)

if __name__ == "__main__":
    data_path = "/data1/liguanlin/Datasets/RealHM"
    #ssdataset = SSHarmonizationTestDataset(data_path)
    from torch.utils.data import DataLoader
    print(os.path)
    #dataloader = DataLoader(ssdataset)

    #print(len(dataloader))